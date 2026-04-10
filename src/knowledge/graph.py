import json
import os
import re
import subprocess
import sys
import traceback
import uuid
import warnings
from pathlib import Path

from neo4j import GraphDatabase as GD

from src import config
from src.models import select_embedding_model
from src.utils import logger
from src.utils.datetime_utils import utc_isoformat

warnings.filterwarnings("ignore", category=UserWarning)


UIE_MODEL = None
ACADEMIC_RELATION_MAP = {
    "belongs_to": "BELONGS_TO",
    "part_of": "BELONGS_TO",
    "includes": "HAS_SUB",
    "has_sub": "HAS_SUB",
    "has_method": "HAS_METHOD",
    "implemented_by": "HAS_METHOD",
    "traversed_by": "HAS_METHOD",
    "has_property": "HAS_PROPERTY",
    "supports": "HAS_PROPERTY",
    "has_complexity": "HAS_COMPLEXITY",
    "average_complexity": "HAS_COMPLEXITY",
    "depends_on": "PREREQUISITE",
    "prerequisite": "PREREQUISITE",
    "cause": "CAUSE",
    "triggers": "CAUSE",
    "switches_to": "CAUSE",
    "used_in": "USED_IN",
    "uses": "USED_IN",
    "used_for": "USED_IN",
    "applies_to": "USED_IN",
    "runs_over": "USED_IN",
    "equivalent": "EQUIVALENT",
    "compared_with": "EQUIVALENT",
    "related_to": "EQUIVALENT",
}


class GraphDatabase:
    def __init__(self):
        self.driver = None
        self.files = []
        self.status = "closed"
        self.kgdb_name = "neo4j"
        self.embed_model_name = os.getenv("GRAPH_EMBED_MODEL_NAME") or "siliconflow/BAAI/bge-m3"
        self.embed_model = select_embedding_model(self.embed_model_name)
        self.work_dir = os.path.join(config.save_dir, "knowledge_graph", self.kgdb_name)
        os.makedirs(self.work_dir, exist_ok=True)

        # 尝试加载已保存的图数据库信息
        if not self.load_graph_info():
            logger.debug("创建新的图数据库配置")

        self.start()

    def start(self):
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        username = os.environ.get("NEO4J_USERNAME", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "0123456789")
        logger.info(f"Connecting to Neo4j: {uri}/{self.kgdb_name}")
        try:
            self.driver = GD.driver(f"{uri}/{self.kgdb_name}", auth=(username, password))
            self.status = "open"
            logger.info(f"Connected to Neo4j: {self.get_graph_info(self.kgdb_name)}")
            # 连接成功后保存图数据库信息
            self.save_graph_info(self.kgdb_name)
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}, {uri}, {self.kgdb_name}, {username}, {password}")

    def close(self):
        """关闭数据库连接"""
        assert self.driver is not None, "Database is not connected"
        self.driver.close()

    def is_running(self):
        """检查图数据库是否正在运行"""
        return self.status == "open" or self.status == "processing"

    def get_sample_nodes(self, kgdb_name="neo4j", num=50, subject: str | None = None):
        """获取指定数据库的 num 个节点信息，优先返回连通的节点子图"""
        assert self.driver is not None, "Database is not connected"
        self.use_database(kgdb_name)

        subject_filter = subject or ""

        def query(tx, num, subject_filter):
            """Note: 使用连通性查询获取集中的节点子图"""
            # 首先尝试获取一个连通的子图
            query_str = """
                // 获取高度数节点作为种子节点
                MATCH (seed:Entity)
                WHERE $subject = '' OR $subject IN coalesce(seed.subject_tags, [])
                WITH seed, COUNT{(seed)-[]->()} + COUNT{(seed)<-[]-()} as degree
                WHERE degree > 0
                ORDER BY degree DESC
                LIMIT 5

                // 为每个种子节点收集更多邻居节点
                UNWIND seed as s
                MATCH (s)-[*1..1]-(neighbor:Entity)
                WHERE $subject = '' OR $subject IN coalesce(neighbor.subject_tags, [])
                WITH s, neighbor, COUNT{(s)-[]->()} + COUNT{(s)<-[]-()} as s_degree
                WITH s, s_degree, collect(DISTINCT neighbor) as neighbors
                // 调整限制比例，允许更多的邻居节点
                WITH s, s_degree, neighbors[0..toInteger($num * 0.15)] as limited_neighbors

                // 从邻居节点扩展到二跳节点，形成开枝散叶结构
                UNWIND limited_neighbors as neighbor
                OPTIONAL MATCH (neighbor)-[*1..1]-(second_hop:Entity)
                WHERE second_hop <> s AND ($subject = '' OR $subject IN coalesce(second_hop.subject_tags, []))
                // 增加二跳节点的数量
                WITH s, limited_neighbors, neighbor, collect(DISTINCT second_hop)[0..5] as second_hops

                // 收集所有连通节点
                WITH collect(DISTINCT s) as seeds,
                    collect(DISTINCT neighbor) as first_hop_nodes,
                    reduce(acc = [], x IN collect(second_hops) | acc + x) as second_hop_nodes
                WITH seeds + first_hop_nodes + second_hop_nodes as connected_nodes

                // 确保不会超过请求的节点数量
                WITH connected_nodes[0..$num] as final_nodes

                // 获取这些节点之间的关系，避免双向边
                UNWIND final_nodes as n
                OPTIONAL MATCH (n)-[rel]-(m)
                WHERE m IN final_nodes AND elementId(n) < elementId(m)
                RETURN
                    {id: elementId(n), name: n.name} AS h,
                    CASE WHEN rel IS NOT NULL THEN
                        {type: rel.type, source_id: elementId(n), target_id: elementId(m)}
                    ELSE null END AS r,
                    CASE WHEN m IS NOT NULL THEN
                        {id: elementId(m), name: m.name}
                    ELSE null END AS t
                """

            try:
                results = tx.run(query_str, num=int(num), subject=subject_filter)
                formatted_results = {"nodes": [], "edges": []}
                node_ids = set()
                node_names = set()

                for item in results:
                    h_node = item["h"]

                    # 始终添加头节点
                    if h_node["id"] not in node_ids and h_node.get("name") not in node_names:
                        formatted_results["nodes"].append(h_node)
                        node_ids.add(h_node["id"])
                        node_names.add(h_node.get("name"))

                    # 只有当边和尾节点都存在时才处理
                    if item["r"] is not None and item["t"] is not None:
                        t_node = item["t"]

                        # 避免重复添加尾节点
                        if t_node["id"] not in node_ids and t_node.get("name") not in node_names:
                            formatted_results["nodes"].append(t_node)
                            node_ids.add(t_node["id"])
                            node_names.add(t_node.get("name"))

                        formatted_results["edges"].append(item["r"])

                # 如果连通查询返回的节点数不足，补充更多节点
                if len(formatted_results["nodes"]) < num:
                    remaining_count = num - len(formatted_results["nodes"])

                    # 获取额外的节点来补充
                    supplement_query = """
                    MATCH (n:Entity)
                    WHERE NOT elementId(n) IN $existing_ids
                      AND ($subject = '' OR $subject IN coalesce(n.subject_tags, []))
                    RETURN {id: elementId(n), name: n.name} AS node
                    LIMIT $count
                    """

                    supplement_results = tx.run(
                        supplement_query,
                        existing_ids=list(node_ids),
                        count=remaining_count,
                        subject=(subject or ""),
                    )

                    for item in supplement_results:
                        node = item["node"]
                        if node["id"] not in node_ids and node.get("name") not in node_names:
                            formatted_results["nodes"].append(node)
                            node_ids.add(node["id"])
                            node_names.add(node.get("name"))

                return formatted_results

            except Exception as e:
                # 如果连通查询失败，使用原始查询作为备选
                logger.warning(f"Connected subgraph query failed, falling back to simple query: {e}")
                fallback_query = """
                MATCH (n:Entity)-[r]-(m:Entity)
                WHERE elementId(n) < elementId(m)
                  AND ($subject = '' OR $subject IN coalesce(n.subject_tags, []))
                  AND ($subject = '' OR $subject IN coalesce(m.subject_tags, []))
                RETURN
                    {id: elementId(n), name: n.name} AS h,
                    {type: r.type, source_id: elementId(n), target_id: elementId(m)} AS r,
                    {id: elementId(m), name: m.name} AS t
                LIMIT $num
                """
                results = tx.run(fallback_query, num=int(num), subject=subject_filter)
                formatted_results = {"nodes": [], "edges": []}
                node_ids = set()
                node_names = set()

                for item in results:
                    h_node = item["h"]
                    t_node = item["t"]

                    # 避免重复添加节点
                    if h_node["id"] not in node_ids and h_node.get("name") not in node_names:
                        formatted_results["nodes"].append(h_node)
                        node_ids.add(h_node["id"])
                    node_names.add(h_node.get("name"))
                    if t_node["id"] not in node_ids and t_node.get("name") not in node_names:
                        formatted_results["nodes"].append(t_node)
                        node_ids.add(t_node["id"])
                        node_names.add(t_node.get("name"))

                    formatted_results["edges"].append(item["r"])

                return formatted_results

        with self.driver.session() as session:
            results = session.execute_read(query, num, subject_filter)
            return results

    def create_graph_database(self, kgdb_name):
        """创建新的数据库，如果已存在则返回已有数据库的名称"""
        assert self.driver is not None, "Database is not connected"
        with self.driver.session() as session:
            existing_databases = session.run("SHOW DATABASES")
            existing_db_names = [db["name"] for db in existing_databases]

            if existing_db_names:
                print(f"已存在数据库: {existing_db_names[0]}")
                return existing_db_names[0]  # 返回所有已有数据库名称

            session.run(f"CREATE DATABASE {kgdb_name}")  # type: ignore
            print(f"数据库 '{kgdb_name}' 创建成功.")
            return kgdb_name  # 返回创建的数据库名称

    def use_database(self, kgdb_name="neo4j"):
        """切换到指定数据库"""
        assert kgdb_name == self.kgdb_name, (
            f"传入的数据库名称 '{kgdb_name}' 与当前实例的数据库名称 '{self.kgdb_name}' 不一致"
        )
        if self.status == "closed":
            self.start()

    async def txt_add_vector_entity(
            self, triples, kgdb_name="neo4j", subject: str | None = None, with_embedding: bool = True
    ):
        """添加实体三元组"""
        assert self.driver is not None, "Database is not connected"
        self.use_database(kgdb_name)

        def _index_exists(tx, index_name):
            """检查索引是否存在"""
            result = tx.run("SHOW INDEXES")
            for record in result:
                if record["name"] == index_name:
                    return True
            return False

        def _create_graph(tx, data, default_subject):
            """添加一个三元组"""
            for entry in data:
                relation_subject = str(entry.get("subject") or default_subject or "").strip()
                relation_type = str(entry.get("r", "")).strip()
                normalized_relation = ACADEMIC_RELATION_MAP.get(relation_type.lower(),relation_type.upper() or "EQUIVALENT")
                tx.run(
                    """
                MERGE (h:Entity:Upload {name: $h})
                MERGE (t:Entity:Upload {name: $t})
                MERGE (h)-[r:RELATION {type: $r}]->(t)
                  SET h.subject_tags = CASE
                    WHEN $subject = '' THEN coalesce(h.subject_tags, [])
                    WHEN $subject IN coalesce(h.subject_tags, []) THEN h.subject_tags
                    ELSE coalesce(h.subject_tags, []) + [$subject]
                END
                SET t.subject_tags = CASE
                    WHEN $subject = '' THEN coalesce(t.subject_tags, [])
                    WHEN $subject IN coalesce(t.subject_tags, []) THEN t.subject_tags
                    ELSE coalesce(t.subject_tags, []) + [$subject]
                END
                SET r.subject_tags = CASE
                    WHEN $subject = '' THEN coalesce(r.subject_tags, [])
                    WHEN $subject IN coalesce(r.subject_tags, []) THEN r.subject_tags
                    ELSE coalesce(r.subject_tags, []) + [$subject]
                END
                """,
                    h=entry["h"],
                    t=entry["t"],
                    r=normalized_relation,
                    subject=relation_subject,
                )

        def _create_vector_index(tx, dim):
            """创建向量索引"""
            # NOTE 这里是否是会重复构建索引？
            index_name = "entityEmbeddings"
            if not _index_exists(tx, index_name):
                tx.run(f"""
                CREATE VECTOR INDEX {index_name}
                FOR (n: Entity) ON (n.embedding)
                OPTIONS {{indexConfig: {{
                `vector.dimensions`: {dim},
                `vector.similarity_function`: 'cosine'
                }} }};
                """)

        def _get_nodes_without_embedding(tx, entity_names):
            """获取没有embedding的节点列表"""
            # 构建参数字典，将列表转换为"param0"、"param1"等键值对形式
            params = {f"param{i}": name for i, name in enumerate(entity_names)}

            # 构建查询参数列表
            param_placeholders = ", ".join([f"${key}" for key in params.keys()])

            # 执行查询
            result = tx.run(
                f"""
            MATCH (n:Entity)
            WHERE n.name IN [{param_placeholders}] AND n.embedding IS NULL
            RETURN n.name AS name
            """,
                params,
            )

            return [record["name"] for record in result]

        def _batch_set_embeddings(tx, entity_embedding_pairs):
            """批量设置实体的嵌入向量"""
            for entity_name, embedding in entity_embedding_pairs:
                tx.run(
                    """
                MATCH (e:Entity {name: $name})
                CALL db.create.setNodeVectorProperty(e, 'embedding', $embedding)
                """,
                    name=entity_name,
                    embedding=embedding,
                )

        # 判断模型名称是否匹配
        self.embed_model_name = self.embed_model_name or config.embed_model
        cur_embed_info = config.embed_model_names.get(self.embed_model_name)
        logger.warning(f"embed_model_name={self.embed_model_name}, {cur_embed_info=}")
        assert self.embed_model_name == config.embed_model or self.embed_model_name is None, (
            f"embed_model_name={self.embed_model_name}, {config.embed_model=}"
        )

        logger.info(f"Adding entity to {kgdb_name}")
        with self.driver.session() as session:
            session.execute_write(_create_graph, triples, subject)

        if not with_embedding:
            logger.info(f"Skip embedding for {kgdb_name}, only graph triples are inserted.")
            self.save_graph_info()
            return

        logger.info(f"Creating vector index for {kgdb_name} with {config.embed_model}")
        with self.driver.session() as session:
            session.execute_write(_create_vector_index, cur_embed_info["dimension"])

        # 收集所有需要处理的实体名称，去重
        all_entities = []
        for entry in triples:
            if entry["h"] not in all_entities:
                all_entities.append(entry["h"])
            if entry["t"] not in all_entities:
                all_entities.append(entry["t"])

        with self.driver.session() as session:
             nodes_without_embedding = session.execute_read(_get_nodes_without_embedding, all_entities)
        if not nodes_without_embedding:
            logger.info("所有实体已有embedding，无需重新计算")
            return

        logger.info(f"需要为{len(nodes_without_embedding)}/{len(all_entities)}个实体计算embedding")

        # 批量处理实体
        max_batch_size = 1024  # 限制此部分的主要是内存大小 1024 * 1024 * 4 / 1024 / 1024 = 4GB
        total_entities = len(nodes_without_embedding)

        for i in range(0, total_entities, max_batch_size):
            batch_entities = nodes_without_embedding[i: i + max_batch_size]
            logger.debug(
                f"Processing entities batch {i // max_batch_size + 1}/"
                f"{(total_entities - 1) // max_batch_size + 1} ({len(batch_entities)} entities)"
            )

            #批量获取嵌入向量
            batch_embeddings = await self.aget_embedding(batch_entities)

            # 将实体名称和嵌入向量配对
            entity_embedding_pairs = list(zip(batch_entities, batch_embeddings))

            # 批量写入数据库（每批独立 session，避免长连接在 await 后失效）
            with self.driver.session() as session:
                 session.execute_write(_batch_set_embeddings, entity_embedding_pairs)

        # 数据添加完成后保存图信息
        self.save_graph_info()

    async def jsonl_file_add_entity(self, file_path, kgdb_name="neo4j", with_embedding: bool = True):
        assert self.driver is not None, "Database is not connected"
        self.status = "processing"
        kgdb_name = kgdb_name or "neo4j"
        self.use_database(kgdb_name)  # 切换到指定数据库
        resolved_file_path = self._resolve_local_file(file_path)
        if not resolved_file_path.exists() and resolved_file_path.name in {"cs408_full_kg_triples.jsonl","cs408_expert_seed.jsonl"}:
            self._try_generate_builtin_cs408_dataset()
            resolved_file_path = self._resolve_local_file(file_path)

        if not resolved_file_path.exists():
            if resolved_file_path.name in {"cs408_full_kg_triples.jsonl", "cs408_full_kg_triples.json","cs408_expert_seed.jsonl"}:
                logger.warning(f"内置图谱文件缺失，回退到内置生成数据: {resolved_file_path}")
                triples = self._generate_builtin_cs408_triples()
                await self.txt_add_vector_entity(triples, kgdb_name, with_embedding=with_embedding)
                self.status = "open"
                self.save_graph_info()
                return kgdb_name
            raise FileNotFoundError(f"图谱文件不存在: {file_path} (resolved: {resolved_file_path})")

        logger.info(f"Start adding entity to {kgdb_name} with {resolved_file_path}")

        def read_triples(target_file_path):
            with open(target_file_path, encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        yield json.loads(line.strip())

        triples = list(read_triples(resolved_file_path))

        await self.txt_add_vector_entity(triples, kgdb_name, with_embedding=with_embedding)

        self.status = "open"
        # 更新并保存图数据库信息
        self.save_graph_info()
        return kgdb_name

    def _resolve_local_file(self, file_path: str | os.PathLike) -> Path:
        candidate = Path(file_path)
        project_root = Path(__file__).resolve().parents[2]

        candidates = []
        if candidate.is_absolute():
            candidates.append(candidate)
        else:
            candidates.append(Path.cwd() / candidate)
            candidates.append(project_root / candidate)
            candidates.append(project_root / "examples" / "cs408" / candidate.name)

        for path in candidates:
            if path.exists():
                return path
        return candidates[0] if candidates else candidate

    def _try_generate_builtin_cs408_dataset(self):
        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "examples" / "cs408" / "generate_full_kg_dataset.py"
        if not script.exists():
            logger.warning(f"内置数据生成脚本不存在: {script}")
            return
        try:
            logger.info(f"尝试自动生成内置408图谱数据: {script}")
            subprocess.run([sys.executable, str(script)], check=True, cwd=project_root)
        except Exception as e:
            logger.warning(f"自动生成内置408图谱数据失败: {e}")

    def _generate_builtin_cs408_triples(self):
        """当本地文件不存在时的兜底内置图谱数据。"""
        return [
            # 数据结构（核心语义关系）
            {"h": "二叉树", "r": "属于", "t": "树结构", "subject": "数据结构"},
            {"h": "栈", "r": "基于", "t": "数组", "subject": "数据结构"},
            {"h": "栈", "r": "基于", "t": "链表", "subject": "数据结构"},
            {"h": "链表", "r": "插入", "t": "时间复杂度 O(1)", "subject": "数据结构"},
            {"h": "链表", "r": "删除", "t": "时间复杂度 O(1)", "subject": "数据结构"},
            {"h": "顺序表", "r": "插入", "t": "时间复杂度 O(n)", "subject": "数据结构"},
            {"h": "哈希表", "r": "查找", "t": "平均时间复杂度 O(1)", "subject": "数据结构"},
            {"h": "快速排序", "r": "平均时间复杂度", "t": "O(n log n)", "subject": "数据结构"},
            {"h": "归并排序", "r": "时间复杂度", "t": "O(n log n)", "subject": "数据结构"},
            {"h": "图", "r": "遍历算法", "t": "BFS", "subject": "数据结构"},
            {"h": "图", "r": "遍历算法", "t": "DFS", "subject": "数据结构"},
            # 操作系统
            {"h": "进程", "r": "拥有", "t": "独立地址空间", "subject": "操作系统"},
            {"h": "线程", "r": "共享", "t": "进程地址空间", "subject": "操作系统"},
            {"h": "死锁", "r": "产生条件", "t": "互斥", "subject": "操作系统"},
            {"h": "死锁", "r": "产生条件", "t": "请求与保持", "subject": "操作系统"},
            {"h": "死锁", "r": "产生条件", "t": "不可剥夺", "subject": "操作系统"},
            {"h": "死锁", "r": "产生条件", "t": "循环等待", "subject": "操作系统"},
            # 计算机网络
            {"h": "TCP", "r": "属于", "t": "传输层协议", "subject": "计算机网络"},
            {"h": "UDP", "r": "属于", "t": "传输层协议", "subject": "计算机网络"},
            {"h": "TCP", "r": "提供", "t": "可靠传输", "subject": "计算机网络"},
            {"h": "UDP", "r": "提供", "t": "无连接传输", "subject": "计算机网络"},
            {"h": "HTTP", "r": "基于", "t": "TCP", "subject": "计算机网络"},
            {"h": "DNS", "r": "默认端口", "t": "53", "subject": "计算机网络"},
            # 计算机组成原理
            {"h": "Cache", "r": "命中率提升依赖", "t": "局部性原理", "subject": "计算机组成原理"},
            {"h": "流水线", "r": "可能产生", "t": "结构冒险", "subject": "计算机组成原理"},
            {"h": "流水线", "r": "可能产生", "t": "数据冒险", "subject": "计算机组成原理"},
            {"h": "流水线", "r": "可能产生", "t": "控制冒险", "subject": "计算机组成原理"},
        ]


    async def auto_build_computer_knowledge_graph(
        self,
        content: str,
        kgdb_name="neo4j",
        clear_existing: bool = False,
        source_name: str | None = None,
        subject: str | None = None,
    ):
        """
        从计算机专业文本（JSON/自然语言）中自动抽取三元组并构建知识图谱。
        """
        assert self.driver is not None, "Database is not connected"
        self.status = "processing"
        kgdb_name = kgdb_name or "neo4j"
        self.use_database(kgdb_name)

        triples = self._extract_computer_triples(content=content, source_name=source_name, default_subject=subject)
        if not triples:
            self.status = "open"
            raise ValueError("未从输入内容中抽取到有效三元组，请检查输入格式")

        if clear_existing:
            self.delete_entity(entity_name=None, kgdb_name=kgdb_name)

        await self.txt_add_vector_entity(triples, kgdb_name=kgdb_name, subject=subject)
        self.status = "open"
        self.save_graph_info()
        return {
            "triples_count": len(triples),
            "unique_entity_count": len({item["h"] for item in triples} | {item["t"] for item in triples}),
            "source_name": source_name or "inline_content",
            "subject": subject or "all",
        }

    async def auto_build_cs408_subject_graphs(self, content: str, kgdb_name="neo4j", clear_existing: bool = False):
        """
        从 408 结构化 JSON 内容中按学科自动构建子图谱，同时可用于总图谱检索。
        """
        try:
            payload = json.loads(content)
        except Exception as e:
            raise ValueError(f"408 学科自动构图仅支持 JSON 内容: {e}")

        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            raise ValueError("408 学科自动构图输入必须是 JSON 数组")

        if clear_existing:
            self.delete_entity(entity_name=None, kgdb_name=kgdb_name)

        grouped: dict[str, list[dict]] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            subject = str(item.get("subject", "")).strip() or "未分类"
            grouped.setdefault(subject, []).append(item)

        build_stats = {}
        total_triples = 0
        for subject, items in grouped.items():
            triples = self._extract_json_triples(json.dumps(items, ensure_ascii=False), default_subject=subject)
            if not triples:
                continue
            await self.txt_add_vector_entity(triples, kgdb_name=kgdb_name, subject=subject)
            total_triples += len(triples)
            build_stats[subject] = {
                "triples_count": len(triples),
                "entity_count": len({t["h"] for t in triples} | {t["t"] for t in triples}),
            }

        self.save_graph_info()
        return {
            "subjects": build_stats,
            "subject_count": len(build_stats),
            "triples_count": total_triples,
        }

    def list_subject_tags(self, kgdb_name="neo4j"):
        assert self.driver is not None, "Database is not connected"
        self.use_database(kgdb_name)

        def query(tx):
            result = tx.run(
                """
                MATCH ()-[r:RELATION]->()
                UNWIND coalesce(r.subject_tags, []) AS subject
                RETURN subject, count(*) AS count
                ORDER BY count DESC, subject ASC
                """
            )
            return [{"subject": row["subject"], "count": row["count"]} for row in result if row["subject"]]

        with self.driver.session() as session:
            return session.execute_read(query)

    def _extract_computer_triples(
        self, content: str, source_name: str | None = None, default_subject: str | None = None
    ):
        """
        轻量规则抽取：优先 JSON 结构，其次按中文句式解析。
        """
        triples = []
        normalized = (content or "").strip()
        if not normalized:
            return triples

        json_like = self._extract_json_triples(normalized, default_subject=default_subject)
        if json_like:
            triples.extend(json_like)

        sentence_like = self._extract_sentence_triples(normalized, default_subject=default_subject)
        if sentence_like:
            triples.extend(sentence_like)

        unique = set()
        deduped = []
        for item in triples:
            h = str(item.get("h", "")).strip()
            r = str(item.get("r", "")).strip()
            t = str(item.get("t", "")).strip()
            subject = str(item.get("subject", "")).strip()
            if not (h and r and t):
                continue
            key = (h, r, t, subject)
            if key in unique:
                continue
            unique.add(key)
            triple = {"h": h, "r": r, "t": t}
            if subject:
                triple["subject"] = subject
            deduped.append(triple)

        logger.info(
            f"自动构图抽取完成: source={source_name or 'inline_content'}, triples={len(deduped)}, raw_len={len(normalized)}"
        )
        return deduped

    def _extract_json_triples(self, content: str, default_subject: str | None = None):
        try:
            payload = json.loads(content)
        except Exception:
            return []

        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            return []

        triples = []
        relation_mapping = {
            "related_topics": "相关",
            "algorithms": "采用算法",
            "properties": "具有属性",
            "applications": "应用于",
            "steps": "步骤",
        }
        for item in payload:
            if not isinstance(item, dict):
                continue
            topic = str(item.get("topic", "")).strip()
            subject = str(item.get("subject", "")).strip() or (default_subject or "")
            if topic and subject:
                triples.append({"h": topic, "r": "BELONGS_TO", "t": subject, "subject": subject})

            for field_name, relation_name in relation_mapping.items():
                values = item.get(field_name, [])
                if not topic or not isinstance(values, list):
                    continue
                for value in values:
                    value_text = str(value).strip()
                    if value_text:
                        triple = {"h": topic, "r": relation_name, "t": value_text}
                        if subject:
                            triple["subject"] = subject
                        triples.append(triple)
            # complexities 支持 dict/list，优先输出“语义关系”
            complexity_values = item.get("complexities", {})
            if topic:
                if isinstance(complexity_values, dict):
                    for op_name, op_complexity in complexity_values.items():
                        op = str(op_name).strip()
                        comp = str(op_complexity).strip()
                        if op and comp:
                           triple = {"h": topic, "r": op, "t": f"时间复杂度 {comp}"}
                           if subject:
                               triple["subject"] = subject
                           triples.append(triple)
                elif isinstance(complexity_values, list):
                    for comp in complexity_values:
                        comp_text = str(comp).strip()
                        if comp_text:
                            triple = {"h": topic, "r": "时间复杂度", "t": comp_text}
                            if subject:
                                triple["subject"] = subject
                            triples.append(triple)


        return triples

    def _extract_sentence_triples(self, content: str, default_subject: str | None = None):
        relations = [
            (r"(.+?)是(.+?)的?一种", "BELONGS_TO"),
            (r"(.+?)属于(.+)", "BELONGS_TO"),
            (r"(.+?)实现(.+)", "HAS_METHOD"),
            (r"(.+?)使用(.+)", "USED_IN"),
            (r"(.+?)具有(.+)", "HAS_PROPERTY"),
            (r"(.+?)复杂度[为是]?(.+)", "HAS_COMPLEXITY"),
            (r"(.+?)依赖(.+)", "PREREQUISITE"),
            (r"(.+?)基于(.+)", "BASED_ON"),
            (r"(.+?)导致(.+)", "CAUSE"),
            (r"(.+?)包含(.+)", "HAS_SUB"),
            (r"(.+?)等价于(.+)", "EQUIVALENT"),
            (r"(.+?)对比(.+)", "EQUIVALENT"),
        ]

        triples = []
        candidates = re.split(r"[。；;\n]", content)
        for line in candidates:
            sentence = line.strip()
            if len(sentence) < 4:
                continue
            for pattern, rel in relations:
                found = re.search(pattern, sentence)
                if not found:
                    continue
                h = found.group(1).strip(" ：:，,")
                t = found.group(2).strip(" ：:，,")
                if h and t and h != t:
                    triple = {"h": h, "r": rel, "t": t}
                    if default_subject:
                        triple["subject"] = default_subject
                    triples.append(triple)
                break
        return triples

    def delete_entity(self, entity_name=None, kgdb_name="neo4j"):
        """删除数据库中的指定实体三元组, 参数entity_name为空则删除全部实体"""
        assert self.driver is not None, "Database is not connected"
        self.use_database(kgdb_name)
        with self.driver.session() as session:
            if entity_name:
                session.execute_write(self._delete_specific_entity, entity_name)
            else:
                session.execute_write(self._delete_all_entities)

    def _delete_specific_entity(self, tx, entity_name):
        query = """
        MATCH (n {name: $entity_name})
        DETACH DELETE n
        """
        tx.run(query, entity_name=entity_name)

    def _delete_all_entities(self, tx):
        query = """
        MATCH (n)
        DETACH DELETE n
        """
        tx.run(query)

    def query_node(
            self,
            keyword,
            threshold=0.9,
            kgdb_name="neo4j",
            hops=2,
            max_entities=8,
            return_format="graph",
            subject: str | None = None,
            **kwargs,
    ):
        """知识图谱查询节点的入口:"""
        assert self.driver is not None, "Database is not connected"
        assert self.is_running(), "图数据库未启动"

        self.use_database(kgdb_name)

        # 简单空格分词，OR 聚合
        tokens = [t for t in str(keyword).split(" ") if t]
        if not tokens:
            tokens = [str(keyword)]

        # name -> score 聚合；向量分数累加，模糊命中给予轻权重
        entity_to_score = {}
        for token in tokens:
            # 使用向量索引进行查询
            results_sim = self._query_with_vector_sim(token, kgdb_name, threshold, subject=subject)
            for r in results_sim:
                name = r[0]  # 与下方保持统一的 [0] 取 name 的方式
                score = 0.0
                try:
                    score = float(r["score"])  # neo4j.Record 支持键访问
                except Exception:
                    # 兜底：若无法取到score，给个基础分
                    score = 0.5
                entity_to_score[name] = max(entity_to_score.get(name, 0.0), score)

            # 模糊查询（不区分大小写），命中加一个较小分
            results_fuzzy = self._query_with_fuzzy_match(token, kgdb_name, subject=subject)
            for fr in results_fuzzy:
                # _query_with_fuzzy_match 返回 values()，形如 [name]
                name = fr[0]
                # 给予轻权重，避免覆盖向量高分
                entity_to_score[name] = max(entity_to_score.get(name, 0.0), 0.3)

        # 排序并截断
        qualified_entities = [name for name, _ in sorted(entity_to_score.items(), key=lambda x: x[1], reverse=True)][
            :max_entities
        ]

        logger.debug(f"Graph Query Entities: {keyword}, {qualified_entities=}")

        # 对每个合格的实体进行查询
        all_query_results = {"nodes": [], "edges": [], "triples": []}
        for entity in qualified_entities:
            query_result = self._query_specific_entity(entity_name=entity, kgdb_name=kgdb_name, hops=hops, subject=subject)
            if return_format == "graph":
                all_query_results["nodes"].extend(query_result["nodes"])
                all_query_results["edges"].extend(query_result["edges"])
            elif return_format == "triples":
                all_query_results["triples"].extend(query_result["triples"])
            else:
                raise ValueError(f"Invalid return_format: {return_format}")

        # 基础去重
        if return_format == "graph":
            seen_node_ids = set()
            dedup_nodes = []
            for n in all_query_results["nodes"]:
                nid = n.get("id") if isinstance(n, dict) else n
                if nid not in seen_node_ids:
                    seen_node_ids.add(nid)
                    dedup_nodes.append(n)
            all_query_results["nodes"] = dedup_nodes

            seen_edges = set()
            dedup_edges = []
            for e in all_query_results["edges"]:
                key = (e.get("source_id"), e.get("target_id"), e.get("type"))
                if key not in seen_edges:
                    seen_edges.add(key)
                    dedup_edges.append(e)
            all_query_results["edges"] = dedup_edges

        elif return_format == "triples":
            seen_triples = set()
            dedup_triples = []
            for t in all_query_results["triples"]:
                if t not in seen_triples:
                    seen_triples.add(t)
                    dedup_triples.append(t)
            all_query_results["triples"] = dedup_triples

        return all_query_results

    def _query_with_fuzzy_match(self, keyword, kgdb_name="neo4j", subject: str | None = None):
        """模糊查询"""
        assert self.driver is not None, "Database is not connected"
        self.use_database(kgdb_name)

        def query_fuzzy_match(tx, keyword, subject):
            result = tx.run(
                """
            MATCH (n:Entity)
            WHERE toLower(n.name) CONTAINS toLower($keyword)
              AND ($subject = '' OR $subject IN coalesce(n.subject_tags, []))
            RETURN DISTINCT n.name AS name
            """,
                keyword=keyword,
                subject=subject or "",
            )
            values = result.values()
            logger.debug(f"Fuzzy Query Results: {values}")
            return values

        with self.driver.session() as session:
            return session.execute_read(query_fuzzy_match, keyword, subject)

    def _query_with_vector_sim(self, keyword, kgdb_name="neo4j", threshold=0.9, subject: str | None = None):
        """向量查询"""
        assert self.driver is not None, "Database is not connected"
        self.use_database(kgdb_name)

        def _index_exists(tx, index_name):
            """检查索引是否存在"""
            result = tx.run("SHOW INDEXES")
            for record in result:
                if record["name"] == index_name:
                    return True
            return False

        def query_by_vector(tx, text, threshold, subject):
            # 首先检查索引是否存在
            if not _index_exists(tx, "entityEmbeddings"):
                raise Exception(
                    "向量索引不存在，请先创建索引，或当前图谱中未上传任何三元组（知识库中自动构建的，不会在此处展示和检索）。"
                )

            embedding = self.get_embedding(text)
            result = tx.run(
                """
            CALL db.index.vector.queryNodes('entityEmbeddings', 10, $embedding)
            YIELD node AS similarEntity, score
            WHERE ($subject = '' OR $subject IN coalesce(similarEntity.subject_tags, []))
            RETURN similarEntity.name AS name, score
            """,
                embedding=embedding,
                subject=subject or "",
            )
            return [r for r in result if r["score"] > threshold]

        with self.driver.session() as session:
            results = session.execute_read(query_by_vector, keyword, threshold=threshold, subject=subject)
            return results

    def _query_specific_entity(self, entity_name, kgdb_name="neo4j", hops=2, limit=100, subject: str | None = None):
        """查询指定实体三元组信息（无向关系）"""
        assert self.driver is not None, "Database is not connected"
        if not entity_name:
            logger.warning("实体名称为空")
            return []

        self.use_database(kgdb_name)

        def query(tx, entity_name, hops, limit, subject):
            try:
                query_str = """
                WITH [
                    // 1跳出边
                    [(n {name: $entity_name})-[r1]->(m1) |
                     {h: {id: elementId(n), name: n.name},
                      r: {type: r1.type, source_id: elementId(n), target_id: elementId(m1), subject_tags: coalesce(r1.subject_tags, [])},
                      t: {id: elementId(m1), name: m1.name}}],
                    // 2跳出边
                    [(n {name: $entity_name})-[r1]->(m1)-[r2]->(m2) |
                     {h: {id: elementId(m1), name: m1.name},
                      r: {type: r2.type, source_id: elementId(m1), target_id: elementId(m2), subject_tags: coalesce(r2.subject_tags, [])},
                      t: {id: elementId(m2), name: m2.name}}],
                    // 1跳入边
                    [(m1)-[r1]->(n {name: $entity_name}) |
                     {h: {id: elementId(m1), name: m1.name},
                      r: {type: r1.type, source_id: elementId(m1), target_id: elementId(n), subject_tags: coalesce(r1.subject_tags, [])},
                      t: {id: elementId(n), name: n.name}}],
                    // 2跳入边
                    [(m2)-[r2]->(m1)-[r1]->(n {name: $entity_name}) |
                     {h: {id: elementId(m2), name: m2.name},
                      r: {type: r2.type, source_id: elementId(m2), target_id: elementId(m1), subject_tags: coalesce(r2.subject_tags, [])},
                      t: {id: elementId(m1), name: m1.name}}]
                ] AS all_results
                UNWIND all_results AS result_list
                UNWIND result_list AS item
                RETURN item.h AS h, item.r AS r, item.t AS t
                LIMIT $limit
                """
                results = tx.run(query_str, entity_name=entity_name, limit=limit)

                if not results:
                    logger.info(f"未找到实体 {entity_name} 的相关信息")
                    return {}

                formatted_results = {"nodes": [], "edges": [], "triples": []}

                for item in results:
                    if subject and subject not in item["r"].get("subject_tags", []):
                        continue
                    formatted_results["nodes"].extend([item["h"], item["t"]])
                    formatted_results["edges"].append(item["r"])
                    formatted_results["triples"].append((item["h"]["name"], item["r"]["type"], item["t"]["name"]))

                logger.debug(f"Query Results: {results}")
                return formatted_results

            except Exception as e:
                logger.error(f"查询实体 {entity_name} 失败: {str(e)}")
                return []

        try:
            with self.driver.session() as session:
                return session.execute_read(query, entity_name, hops, limit, subject)

        except Exception as e:
            logger.error(f"数据库会话异常: {str(e)}")
            return []

    def recommend_learning_path(self, concept: str, kgdb_name="neo4j", subject: str | None = None, max_steps: int = 12):
        """基于 PREREQUISITE 关系推荐学习路径（从基础到目标概念）。"""
        assert self.driver is not None, "Database is not connected"
        self.use_database(kgdb_name)

        def query(tx, concept, subject, max_steps):
            result = tx.run(
                """
                MATCH p=(pre:Entity)-[:RELATION {type:'PREREQUISITE'}*1..6]->(target:Entity {name:$concept})
                WHERE ($subject='' OR $subject IN coalesce(target.subject_tags, []))
                RETURN [n IN nodes(p) | n.name] AS path
                ORDER BY length(p) DESC
                LIMIT 1
                """,
                concept=concept,
                subject=subject or "",
            ).single()
            if result and result.get("path"):
                return result["path"][:max_steps]
            return [concept]

        with self.driver.session() as session:
            return session.execute_read(query, concept, subject, max_steps)

    def analyze_knowledge_association(
        self, concept: str, kgdb_name="neo4j", subject: str | None = None, max_neighbors: int = 30
    ):
        """知识关联分析：返回关联概念及关系分布。"""
        assert self.driver is not None, "Database is not connected"
        self.use_database(kgdb_name)

        def query(tx, concept, subject, max_neighbors):
            rows = tx.run(
                """
                MATCH (c:Entity {name:$concept})-[r:RELATION]-(n:Entity)
                WHERE ($subject='' OR $subject IN coalesce(r.subject_tags, []))
                RETURN n.name AS neighbor, r.type AS rel
                LIMIT $max_neighbors
                """,
                concept=concept,
                subject=subject or "",
                max_neighbors=max_neighbors,
            )
            neighbors = []
            rel_dist = {}
            for row in rows:
                neighbors.append({"name": row["neighbor"], "relation": row["rel"]})
                rel_dist[row["rel"]] = rel_dist.get(row["rel"], 0) + 1
            return {"neighbors": neighbors, "relation_distribution": rel_dist}

        with self.driver.session() as session:
            return session.execute_read(query, concept, subject, max_neighbors)

    async def aget_embedding(self, text):
        if isinstance(text, list):
            outputs = await self.embed_model.abatch_encode(text, batch_size=40)
            return outputs
        else:
            outputs = await self.embed_model.aencode(text)
            return outputs

    def get_embedding(self, text):
        if isinstance(text, list):
            outputs = self.embed_model.batch_encode(text, batch_size=40)
            return outputs
        else:
            outputs = self.embed_model.encode([text])[0]
            return outputs

    def set_embedding(self, tx, entity_name, embedding):
        tx.run(
            """
        MATCH (e:Entity {name: $name})
        CALL db.create.setNodeVectorProperty(e, 'embedding', $embedding)
        """,
            name=entity_name,
            embedding=embedding,
        )

    def get_graph_info(self, graph_name="neo4j"):
        assert self.driver is not None, "Database is not connected"
        self.use_database(graph_name)

        def query(tx):
            # 只统计包含Entity标签的节点
            entity_count = tx.run("MATCH (n:Entity) RETURN count(n) AS count").single()["count"]
            # 只统计包含RELATION标签的关系
            relationship_count = tx.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS count").single()["count"]
            triples_count = tx.run("MATCH (n:Entity)-[r:RELATION]->(m:Entity) RETURN count(n) AS count").single()[
                "count"
            ]

            # 获取所有标签
            labels = tx.run("CALL db.labels() YIELD label RETURN collect(label) AS labels").single()["labels"]

            return {
                "graph_name": graph_name,
                "entity_count": entity_count,
                "relationship_count": relationship_count,
                "triples_count": triples_count,
                "labels": labels,
                "status": self.status,
                "embed_model_name": self.embed_model_name,
                "unindexed_node_count": len(self.query_nodes_without_embedding(graph_name)),
            }

        try:
            if self.is_running():
                # 获取数据库信息
                with self.driver.session() as session:
                    graph_info = session.execute_read(query)

                    # 添加时间戳
                    graph_info["last_updated"] = utc_isoformat()
                    return graph_info
            else:
                logger.warning(f"图数据库未连接或未运行:{self.status=}")
                return None

        except Exception as e:
            logger.error(f"获取图数据库信息失败：{e}, {traceback.format_exc()}")
            return None

    def save_graph_info(self, graph_name="neo4j"):
        """
        将图数据库的基本信息保存到工作目录中的JSON文件
        保存的信息包括：数据库名称、状态、嵌入模型名称等
        """
        try:
            graph_info = self.get_graph_info(graph_name)
            if graph_info is None:
                logger.error("图数据库信息为空，无法保存")
                return False

            info_file_path = os.path.join(self.work_dir, "graph_info.json")
            with open(info_file_path, "w", encoding="utf-8") as f:
                json.dump(graph_info, f, ensure_ascii=False, indent=2)

            # logger.info(f"图数据库信息已保存到：{info_file_path}")
            return True
        except Exception as e:
            logger.error(f"保存图数据库信息失败：{e}")
            return False

    def query_nodes_without_embedding(self, kgdb_name="neo4j"):
        """查询没有嵌入向量的节点

        Returns:
            list: 没有嵌入向量的节点列表
        """
        assert self.driver is not None, "Database is not connected"
        self.use_database(kgdb_name)

        def query(tx):
            result = tx.run("""
            MATCH (n:Entity)
            WHERE n.embedding IS NULL
            RETURN n.name AS name
            """)
            return [record["name"] for record in result]

        with self.driver.session() as session:
            return session.execute_read(query)

    def load_graph_info(self):
        """
        从工作目录中的JSON文件加载图数据库的基本信息
        返回True表示加载成功，False表示加载失败
        """
        try:
            info_file_path = os.path.join(self.work_dir, "graph_info.json")
            if not os.path.exists(info_file_path):
                logger.debug(f"图数据库信息文件不存在：{info_file_path}")
                return False

            with open(info_file_path, encoding="utf-8") as f:
                graph_info = json.load(f)

            # 更新对象属性
            if graph_info.get("embed_model_name"):
                self.embed_model_name = graph_info["embed_model_name"]

            # 如果需要，可以加载更多信息
            # 注意：这里不更新self.kgdb_name，因为它是在初始化时设置的

            logger.info(f"已加载图数据库信息，最后更新时间：{graph_info.get('last_updated')}")
            return True
        except Exception as e:
            logger.error(f"加载图数据库信息失败：{e}")
            return False

    def add_embedding_to_nodes(self, node_names=None, kgdb_name="neo4j"):
        """为节点添加嵌入向量

        Args:
            node_names (list, optional): 要添加嵌入向量的节点名称列表，None表示所有没有嵌入向量的节点
            kgdb_name (str, optional): 图数据库名称，默认为'neo4j'

        Returns:
            int: 成功添加嵌入向量的节点数量
        """
        assert self.driver is not None, "Database is not connected"
        self.use_database(kgdb_name)

        # 如果node_names为None，则获取所有没有嵌入向量的节点
        if node_names is None:
            node_names = self.query_nodes_without_embedding(kgdb_name)

        count = 0
        with self.driver.session() as session:
            for node_name in node_names:
                try:
                    embedding = self.get_embedding(node_name)
                    session.execute_write(self.set_embedding, node_name, embedding)
                    count += 1
                except Exception as e:
                    logger.error(f"为节点 '{node_name}' 添加嵌入向量失败: {e}, {traceback.format_exc()}")

        return count

    def submit_graph_edit(self, edit_payload: dict, submitter: str = "unknown"):
        project_root = Path(__file__).resolve().parents[2]
        queue_file = project_root / "saves" / "knowledge_graph" / "review_queue.jsonl"
        queue_file.parent.mkdir(parents=True, exist_ok=True)
        item = {
            "id": str(uuid.uuid4()),
            "status": "pending",
            "submitter": submitter,
            "payload": edit_payload,
            "created_at": utc_isoformat(),
        }
        with open(queue_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return item

    def list_graph_edits(self, status: str | None = None):
        project_root = Path(__file__).resolve().parents[2]
        queue_file = project_root / "saves" / "knowledge_graph" / "review_queue.jsonl"
        if not queue_file.exists():
            return []
        items = []
        for line in queue_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if status and item.get("status") != status:
                continue
            items.append(item)
        return items

    def review_graph_edit(self, edit_id: str, action: str, reviewer: str = "admin"):
        assert action in {"approve", "reject"}, "action must be approve or reject"
        project_root = Path(__file__).resolve().parents[2]
        queue_file = project_root / "saves" / "knowledge_graph" / "review_queue.jsonl"
        if not queue_file.exists():
            raise FileNotFoundError("审核队列为空")

        lines = queue_file.read_text(encoding="utf-8").splitlines()
        updated = []
        target = None
        for line in lines:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("id") == edit_id:
                item["status"] = "approved" if action == "approve" else "rejected"
                item["reviewer"] = reviewer
                item["reviewed_at"] = utc_isoformat()
                target = item
            updated.append(item)

        if target is None:
            raise ValueError(f"未找到编辑记录: {edit_id}")

        queue_file.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in updated) + "\n", encoding="utf-8")
        if action == "approve":
            self._apply_approved_edit(target["payload"])
        return target

    def _apply_approved_edit(self, payload: dict):
        project_root = Path(__file__).resolve().parents[2]
        payload_type = payload.get("type", "add_triple")
        if payload_type == "add_triple":
            seed_file = project_root / "examples" / "cs408" / "cs408_expert_seed.jsonl"
            seed_file.parent.mkdir(parents=True, exist_ok=True)
            triple = payload.get("triple", {})
            with open(seed_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(triple, ensure_ascii=False) + "\n")
        elif payload_type == "add_relation":
            ontology_file = project_root / "examples" / "cs408" / "cs408_relation_ontology.yaml"
            if not ontology_file.exists():
                return
            import yaml

            data = yaml.safe_load(ontology_file.read_text(encoding="utf-8")) or {}
            data.setdefault("relations", {})
            relation_name = payload.get("relation_name")
            relation_desc = payload.get("relation_desc", "")
            if relation_name:
                data["relations"][relation_name] = {"description": relation_desc}
                ontology_file.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def format_general_results(self, results):
        nodes = []
        edges = []

        for item in results:
            nodes.extend([item["h"], item["t"]])
            edges.append(item["r"])

        formatted_results = {"nodes": nodes, "edges": edges}
        return formatted_results

    def _extract_relationship_info(self, relationship, source_name=None, target_name=None, node_dict=None):
        """
        提取关系信息并返回格式化的节点和边信息
        """
        rel_id = relationship.element_id
        nodes = relationship.nodes
        if len(nodes) != 2:
            return None, None

        source, target = nodes
        source_id = source.element_id
        target_id = target.element_id

        # 如果没有提供 source_name 或 target_name，则需要 node_dict
        if source_name is None or target_name is None:
            assert node_dict is not None, "node_dict is required when source_name or target_name is None"
            source_name = node_dict[source_id]["name"] if source_name is None else source_name
            target_name = node_dict[target_id]["name"] if target_name is None else target_name

        relationship_type = relationship._properties.get("type", "unknown")
        if relationship_type == "unknown":
            relationship_type = relationship.type

        edge_info = {
            "id": rel_id,
            "type": relationship_type,
            "source_id": source_id,
            "target_id": target_id,
            "source_name": source_name,
            "target_name": target_name,
        }

        node_info = [
            {"id": source_id, "name": source_name},
            {"id": target_id, "name": target_name},
        ]

        return node_info, edge_info


def clean_triples_embedding(triples):
    for item in triples:
        if hasattr(item[0], "_properties"):
            item[0]._properties["embedding"] = None
        if hasattr(item[2], "_properties"):
            item[2]._properties["embedding"] = None
    return triples


if __name__ == "__main__":
    pass

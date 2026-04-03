import os
import traceback
import asyncio
import re
import json
import inspect
from time import perf_counter

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, setup_logger
from lightrag.prompt import PROMPTS
from neo4j import GraphDatabase
import numpy as np
from openai import AsyncOpenAI

from src.knowledge.base import KnowledgeBase
from src.knowledge.config.domain_entity_config import get_domain_entity_relation_config
from src.knowledge.pipeline import UnstructuredToKGPipeline
from src.knowledge.indexing import process_file_to_markdown, process_url_to_markdown
from src.knowledge.utils.kb_utils import get_embedding_config, prepare_item_metadata
from src.utils import hashstr, logger
from src.utils.datetime_utils import shanghai_now

LIGHTRAG_LLM_PROVIDER = os.getenv("LIGHTRAG_LLM_PROVIDER", "siliconflow")
LIGHTRAG_LLM_NAME = os.getenv("LIGHTRAG_LLM_NAME", "zai-org/GLM-4.5-Air")


async def _safe_ainsert_with_retry(rag: LightRAG, markdown_content: str, file_id: str, item_path: str) -> dict:
    """模块级兜底重试函数，避免实例方法缺失导致上传失败。"""
    max_retries = int(os.getenv("LIGHTRAG_INSERT_MAX_RETRIES", "2"))
    retry_delay = float(os.getenv("LIGHTRAG_INSERT_RETRY_DELAY", "1.5"))
    last_error = None

    for attempt in range(max_retries + 1):
        start = perf_counter()
        try:
            await rag.ainsert(input=markdown_content, ids=file_id, file_paths=item_path)
            return {"attempt": attempt + 1, "duration_sec": round(perf_counter() - start, 4), "status": "done"}
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt >= max_retries:
                break
            await asyncio.sleep(retry_delay * (attempt + 1))

    return {
        "attempt": max_retries + 1,
        "duration_sec": 0.0,
        "status": "failed",
        "error": last_error or "unknown insertion error",
    }


def _get_neo4j_connection_config() -> tuple[str, str, str]:
    """统一获取 Neo4j 连接配置，避免不同分支使用不一致默认值。"""
    return (
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USERNAME", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "12345678"),
    )


class LightRagKB(KnowledgeBase):
    """基于 LightRAG 的知识库实现"""

    def __init__(self, work_dir: str, **kwargs):
        """
        初始化 LightRAG 知识库

        Args:
            work_dir: 工作目录
            **kwargs: 其他配置参数
        """
        super().__init__(work_dir)

        # 存储 LightRAG 实例映射 {db_id: LightRAG}
        self.instances: dict[str, LightRAG] = {}

        # 设置 LightRAG 日志
        log_dir = os.path.join(work_dir, "logs", "lightrag")
        os.makedirs(log_dir, exist_ok=True)
        setup_logger(
            "lightrag",
            log_file_path=os.path.join(log_dir, f"lightrag_{shanghai_now().strftime('%Y-%m-%d')}.log"),
        )

        logger.info("LightRagKB initialized")
        self._kg_pipeline_cache: dict[str, UnstructuredToKGPipeline] = {}
        self._kg_pipeline_cache_sig: dict[str, tuple] = {}

    def _get_db_plugin_config(self, db_id: str) -> dict:
        db_meta = self.databases_meta.get(db_id, {})
        metadata = db_meta.get("metadata", {}) or {}
        addon_params = metadata.get("addon_params", {}) if isinstance(metadata.get("addon_params"), dict) else {}
        return {
            "kg_ner_plugin": addon_params.get("kg_ner_plugin", os.getenv("LIGHTRAG_KG_NER_PLUGIN", "rule")),
            "kg_re_plugin": addon_params.get("kg_re_plugin", os.getenv("LIGHTRAG_KG_RE_PLUGIN", "rule")),
            "kg_ner_model_spec": addon_params.get("kg_ner_model_spec", os.getenv("KG_NER_PLUGIN_MODEL_SPEC", "")),
            "kg_re_model_spec": addon_params.get("kg_re_model_spec", os.getenv("KG_RE_PLUGIN_MODEL_SPEC", "")),
            "kg_ner_llm_enabled": addon_params.get("kg_ner_llm_enabled", False),
            "kg_re_llm_enabled": addon_params.get("kg_re_llm_enabled", False),
        }

    def _get_generic_kg_pipeline(self, db_id: str) -> UnstructuredToKGPipeline:
        config = self._get_db_plugin_config(db_id)
        sig = (
            config["kg_ner_plugin"],
            config["kg_re_plugin"],
            config["kg_ner_model_spec"],
            config["kg_re_model_spec"],
            bool(config["kg_ner_llm_enabled"]),
            bool(config["kg_re_llm_enabled"]),
        )
        if db_id in self._kg_pipeline_cache and self._kg_pipeline_cache_sig.get(db_id) == sig:
            return self._kg_pipeline_cache[db_id]

        pipeline = UnstructuredToKGPipeline(
            ner_plugin=config["kg_ner_plugin"],
            re_plugin=config["kg_re_plugin"],
            ner_kwargs={
                "enabled": bool(config["kg_ner_llm_enabled"]),
                "model_spec": config["kg_ner_model_spec"],
            },
            re_kwargs={
                "enabled": bool(config["kg_re_llm_enabled"]),
                "model_spec": config["kg_re_model_spec"],
            },
        )
        self._kg_pipeline_cache[db_id] = pipeline
        self._kg_pipeline_cache_sig[db_id] = sig
        return pipeline

    @property
    def kb_type(self) -> str:
        """知识库类型标识"""
        return "lightrag"

    def delete_database(self, db_id: str) -> dict:
        """删除数据库，同时清除Chroma和Neo4j中的数据"""
        # ChromaDB 数据会在父类的 delete_database 中自动清理，无需特殊处理

        # Delete Neo4j data
        neo4j_uri, neo4j_username, neo4j_password = _get_neo4j_connection_config()

        try:
            with GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password)) as driver:
                with driver.session() as session:
                    # 删除该知识库相关的所有节点和关系
                    query = """
                    MATCH (n {workspace: $workspace})
                    DETACH DELETE n
                    """
                    session.run(query, workspace=db_id)
                    logger.info(f"Deleted Neo4j data for workspace {db_id}")
        except Exception as e:
            logger.error(f"Failed to delete Neo4j data for {db_id}: {e}")

        # Delete local files and metadata
        return super().delete_database(db_id)

    async def _create_kb_instance(self, db_id: str, kb_config: dict) -> LightRAG:
        """创建 LightRAG 实例"""
        logger.info(f"Creating LightRAG instance for {db_id}")

        if db_id not in self.databases_meta:
            raise ValueError(f"Database {db_id} not found")

        llm_info = self.databases_meta[db_id].get("llm_info", {})
        embed_info = self.databases_meta[db_id].get("embed_info", {})
        # 读取在创建数据库时透传的附加参数（包括语言）
        metadata = self.databases_meta[db_id].get("metadata", {}) or {}
        addon_params = {}
        if isinstance(metadata.get("addon_params"), dict):
            addon_params.update(metadata.get("addon_params", {}))
        # 兼容直接放在 metadata 下的 language
        if isinstance(metadata.get("language"), str) and metadata.get("language"):
            addon_params.setdefault("language", metadata.get("language"))
        # 默认语言从环境变量读取，默认 Chinese
        addon_params.setdefault("language", os.getenv("SUMMARY_LANGUAGE", "Chinese"))

        # 根据领域自动设置实体/关系类型（默认 computer，可选 museum 兼容旧行为）
        domain = addon_params.get("domain") or os.getenv("KG_DOMAIN", "computer")
        domain_config = get_domain_entity_relation_config(domain)
        addon_params.setdefault("domain", domain)
        if not addon_params.get("entity_types"):
            addon_params["entity_types"] = domain_config["entity_types"]
        if not addon_params.get("relation_types"):
            addon_params["relation_types"] = domain_config["relation_types"]
        # 创建工作目录
        working_dir = os.path.join(self.work_dir, db_id)
        os.makedirs(working_dir, exist_ok=True)

        # 创建 LightRAG 实例
        rag = LightRAG(
            working_dir=working_dir,
            workspace=db_id,
            llm_model_func=self._get_llm_func(llm_info),
            embedding_func=self._get_embedding_func(embed_info),
            vector_storage="FaissVectorDBStorage",
            kv_storage="JsonKVStorage",
            graph_storage="Neo4JStorage",
            doc_status_storage="JsonDocStatusStorage",
            log_file_path=os.path.join(working_dir, "lightrag.log"),
            addon_params=addon_params,
        )

        return rag

    async def _initialize_kb_instance(self, instance: LightRAG) -> None:
        """初始化 LightRAG 实例"""
        logger.info(f"Initializing LightRAG instance for {instance.working_dir}")
        await instance.initialize_storages()
        await initialize_pipeline_status()

    async def _get_lightrag_instance(self, db_id: str) -> LightRAG | None:
        """获取或创建 LightRAG 实例"""
        if db_id in self.instances:
            return self.instances[db_id]

        if db_id not in self.databases_meta:
            return None

        try:
            # 创建实例
            rag = await self._create_kb_instance(db_id, {})

            # 异步初始化存储
            await self._initialize_kb_instance(rag)

            self.instances[db_id] = rag
            return rag

        except Exception as e:
            logger.error(f"Failed to create LightRAG instance for {db_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _get_llm_func(self, llm_info: dict):
        """获取 LLM 函数"""
        from src.models import select_model

        # 如果用户选择了LLM，使用用户选择的；否则使用环境变量默认值
        if llm_info and llm_info.get("model_spec"):
            model_spec = llm_info["model_spec"]
            logger.info(f"Using user-selected LLM spec: {model_spec}")
        elif llm_info and llm_info.get("provider") and llm_info.get("model_name"):
            model_spec = f"{llm_info['provider']}/{llm_info['model_name']}"
            logger.info(f"Using user-selected LLM: {model_spec}")
        else:
            provider = LIGHTRAG_LLM_PROVIDER
            model_name = LIGHTRAG_LLM_NAME
            model_spec = f"{provider}/{model_name}"
            logger.info(f"Using default LLM from environment: {provider}/{model_name}")

        model = select_model(model_spec=model_spec)

        async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return await openai_complete_if_cache(
                model=model.model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=model.api_key,
                base_url=model.base_url,
                **kwargs,
            )

        return llm_model_func

    def _get_embedding_func(self, embed_info: dict):
        """获取 embedding 函数"""
        config_dict = get_embedding_config(embed_info)
        embedding_state = {"dimension": int(config_dict["dimension"])}
        embedding_func = None
        embed_client = None
        supports_dimensions = "dimensions" in inspect.signature(openai_embed).parameters
        if not supports_dimensions:
            logger.warning(
                "Current LightRAG openai_embed has no `dimensions` parameter; "
                "using provider default dimension. model={}",
                config_dict["model"],
            )

        async def custom_embed(texts):
            nonlocal embedding_func, embed_client, supports_dimensions
            embed_kwargs = {
                "texts": texts,
                "model": config_dict["model"],
                "api_key": config_dict["api_key"],
                "base_url": config_dict["base_url"].replace("/embeddings", ""),
            }
            if embed_client is None:
                embed_client = AsyncOpenAI(
                    api_key=config_dict["api_key"],
                    base_url=embed_kwargs["base_url"],
                )

            async def _request_embeddings(with_dimensions: bool) -> list[list[float]]:
                request_kwargs = {
                    "model": config_dict["model"],
                    "input": texts,
                }
                if with_dimensions:
                    request_kwargs["dimensions"] = embedding_state["dimension"]
                response = await embed_client.embeddings.create(**request_kwargs)
                return [item.embedding for item in response.data]

            try:
                embeddings = await _request_embeddings(with_dimensions=supports_dimensions)
            except Exception as exc:  # noqa: BLE001
                should_retry_without_dimensions = supports_dimensions and (
                    "dimensions" in str(exc).lower() or "unexpected" in str(exc).lower()
                )
                if should_retry_without_dimensions:
                    logger.warning(
                        "Embedding API rejected `dimensions`; fallback to provider default dimension. model={}",
                        config_dict["model"],
                    )
                    supports_dimensions = False
                    embeddings = await _request_embeddings(with_dimensions=False)
                else:
                    # 兜底回退到 LightRAG 的 openai_embed 适配器
                    if supports_dimensions:
                        embeddings = await openai_embed(
                            **embed_kwargs,
                            dimensions=embedding_state["dimension"],
                        )
                    else:
                        embeddings = await openai_embed(**embed_kwargs)

            if embeddings is None:
                return np.zeros((0, embedding_state["dimension"]), dtype=np.float32)
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            if not embeddings:
                return np.zeros((0, embedding_state["dimension"]), dtype=np.float32)

            first_vector = embeddings[0] if isinstance(embeddings[0], list) else []
            real_dimension = len(first_vector)
            if real_dimension <= 0:
                return np.zeros((0, embedding_state["dimension"]), dtype=np.float32)

            if real_dimension != embedding_state["dimension"]:
                logger.warning(
                    "Embedding dimension mismatch detected (configured={}, actual={}). "
                    "Auto-adjusting to actual dimension for model={}.",
                    embedding_state["dimension"],
                    real_dimension,
                    config_dict["model"],
                )
                embedding_state["dimension"] = real_dimension
                if embedding_func is not None:
                    embedding_func.embedding_dim = real_dimension

            target_dim = embedding_state["dimension"]
            normalized = []
            for vec in embeddings:
                if not isinstance(vec, list):
                    continue
                if len(vec) > target_dim:
                    normalized.append(vec[:target_dim])
                elif len(vec) < target_dim:
                    normalized.append(vec + [0.0] * (target_dim - len(vec)))
                else:
                    normalized.append(vec)
            if not normalized:
                return np.zeros((0, target_dim), dtype=np.float32)
            return np.asarray(normalized, dtype=np.float32)

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_state["dimension"],
            max_token_size=4096,
            func=custom_embed,
        )
        return embedding_func

    async def _ainsert_with_retry(self, rag: LightRAG, markdown_content: str, file_id: str, item_path: str) -> dict:
        """带重试的图谱写入，提升外部 LLM/网络抖动场景稳定性。"""
        return await _safe_ainsert_with_retry(rag, markdown_content, file_id, item_path)

    def _extract_generic_triples(
            self, text: str, pipeline: UnstructuredToKGPipeline | None = None
    ) -> list[tuple[str, str, str]]:
        """通用文本兜底三元组抽取（规则版，保证普通文档也可形成基础图谱）。"""
        clean = re.sub(r"\s+", " ", text or "").strip()
        if not clean:
            return []

        # 优先尝试 JSON 结构抽取（适配通用 JSON 文档）
        json_match = re.search(r"```json\s*(.*?)\s*```", text or "", re.S)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                json_triples = self._extract_json_triples(data)
                if json_triples:
                    return json_triples[:80]
            except Exception:  # noqa: BLE001
                pass

        sentences = re.split(r"[。！？\n;；]", clean)
        triples: list[tuple[str, str, str]] = []
        # 优先走 GraphRAG 风格流水线（规则版）
        pipeline_result = (pipeline or UnstructuredToKGPipeline()).run(clean, max_triples=80)
        if pipeline_result.triples:
            return [(tri.subject, tri.predicate, tri.obj) for tri in pipeline_result.triples]

        patterns = [
            (r"(.{1,30})是(.{1,30})的一种", "IS_A"),
            (r"(.{1,30})包括(.{1,30})", "INCLUDES"),
            (r"(.{1,30})由(.{1,30})组成", "COMPOSED_OF"),
            (r"(.{1,30})用于(.{1,30})", "USED_FOR"),
            (r"(.{1,30})依赖(.{1,30})", "DEPENDS_ON"),
        ]

        for sent in sentences:
            sent = sent.strip(" ，,：:")
            if len(sent) < 6:
                continue
            matched = False
            for regex, rel in patterns:
                m = re.search(regex, sent)
                if m:
                    head = m.group(1).strip(" ，,：:")
                    tail = m.group(2).strip(" ，,：:")
                    if head and tail and head != tail:
                        triples.append((head[:60], rel, tail[:60]))
                        matched = True
                        break
            if not matched and "是" in sent:
                parts = sent.split("是", 1)
                if len(parts) == 2:
                    head, tail = parts[0].strip(), parts[1].strip()
                    if 1 < len(head) <= 30 and 1 < len(tail) <= 30 and head != tail:
                        triples.append((head[:60], "RELATED_TO", tail[:60]))

        # 去重并限制数量，避免过量噪音
        dedup = []
        seen = set()
        for tri in triples:
            if tri in seen:
                continue
            seen.add(tri)
            dedup.append(tri)
            if len(dedup) >= 80:
                break

        # 最后兜底：如果规则没有命中，仍基于文本片段构造最小连通图，保证图谱非空。
        if not dedup:
            segments = [
                seg.strip(" ，,：:()（）[]【】\"'“”")
                for seg in re.split(r"[。！？\n;；,，]", clean)
                if seg and len(seg.strip()) >= 4
            ]
            compact = []
            for seg in segments:
                normalized = re.sub(r"\s+", " ", seg)[:48]
                if normalized and normalized not in compact:
                    compact.append(normalized)
                if len(compact) >= 12:
                    break

            for idx in range(len(compact) - 1):
                head = compact[idx]
                tail = compact[idx + 1]
                if head != tail:
                    dedup.append((head, "RELATED_TO", tail))

        return dedup

    def _extract_json_triples(self, data: object, root_name: str = "Document") -> list[tuple[str, str, str]]:
        triples: list[tuple[str, str, str]] = []

        def _walk(node: object, parent: str):
            if isinstance(node, dict):
                for k, v in node.items():
                    key_name = str(k)[:60]
                    triples.append((parent[:60], "HAS_FIELD", key_name))
                    if isinstance(v, (dict, list)):
                        _walk(v, key_name)
                    else:
                        value_name = str(v)[:80]
                        if value_name:
                            triples.append((key_name, "VALUE_IS", value_name))
            elif isinstance(node, list):
                for idx, item in enumerate(node[:100]):
                    item_name = f"{parent}_item_{idx}"
                    triples.append((parent[:60], "HAS_ITEM", item_name[:60]))
                    if isinstance(item, (dict, list)):
                        _walk(item, item_name)
                    else:
                        value_name = str(item)[:80]
                        if value_name:
                            triples.append((item_name[:60], "VALUE_IS", value_name))

        _walk(data, root_name)
        dedup = []
        seen = set()
        for tri in triples:
            if tri in seen or tri[0] == tri[2]:
                continue
            seen.add(tri)
            dedup.append(tri)
            if len(dedup) >= 80:
                break
        return dedup

    def _fallback_insert_generic_graph(
            self, workspace: str, markdown_content: str, pipeline: UnstructuredToKGPipeline | None = None
    ) -> dict:
        """当 LightRAG 未抽取到图谱节点时，使用规则抽取结果落库 Neo4j。"""
        triples = self._extract_generic_triples(markdown_content, pipeline=pipeline)
        if not triples:
            return {"inserted_nodes": 0, "inserted_edges": 0}

        neo4j_uri, neo4j_username, neo4j_password = _get_neo4j_connection_config()

        with GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password)) as driver:
            with driver.session() as session:
                query = """
                UNWIND $triples AS tri
                MERGE (h:Entity {name: tri.head, workspace: $workspace})
                ON CREATE SET h.entity_type = "GenericConcept"
                MERGE (t:Entity {name: tri.tail, workspace: $workspace})
                ON CREATE SET t.entity_type = "GenericConcept"
                CALL apoc.create.relationship(h, tri.rel, {workspace: $workspace, source: "fallback_rule"}, t) YIELD rel
                RETURN count(rel) AS edge_count
                """
                # 如果 Neo4j 未安装 APOC，使用固定关系类型兜底
                try:
                    record = session.run(
                        query,
                        workspace=workspace,
                        triples=[{"head": h, "rel": r, "tail": t} for h, r, t in triples],
                    ).single()
                    edge_count = record["edge_count"] if record else 0
                except Exception:
                    plain_query = """
                    UNWIND $triples AS tri
                    MERGE (h:Entity {name: tri.head, workspace: $workspace})
                    ON CREATE SET h.entity_type = "GenericConcept"
                    MERGE (t:Entity {name: tri.tail, workspace: $workspace})
                    ON CREATE SET t.entity_type = "GenericConcept"
                    MERGE (h)-[r:RELATED_TO {workspace: $workspace, source: "fallback_rule"}]->(t)
                    RETURN count(r) AS edge_count
                    """
                    record = session.run(
                        plain_query,
                        workspace=workspace,
                        triples=[{"head": h, "tail": t} for h, _, t in triples],
                    ).single()
                    edge_count = record["edge_count"] if record else 0

                node_count = session.run(
                    "MATCH (n {workspace: $workspace}) RETURN count(n) AS node_count",
                    workspace=workspace,
                ).single()
                return {
                    "inserted_nodes": (node_count["node_count"] if node_count else 0),
                    "inserted_edges": edge_count,
                }

    async def add_content(self, db_id: str, items: list[str], params: dict | None = None) -> list[dict]:
        """添加内容（文件/URL）"""
        if db_id not in self.databases_meta:
            raise ValueError(f"Database {db_id} not found")

        rag = await self._get_lightrag_instance(db_id)
        if not rag:
            raise ValueError(f"Failed to get LightRAG instance for {db_id}")

        content_type = params.get("content_type", "file") if params else "file"
        processed_items_info = []

        for item in items:
            # 准备文件元数据
            metadata = prepare_item_metadata(item, content_type, db_id)
            file_id = metadata["file_id"]
            item_path = metadata["path"]

            # 添加文件记录
            file_record = metadata.copy()
            self.files_meta[file_id] = file_record
            self._save_metadata()

            self._add_to_processing_queue(file_id)
            try:
                db_pipeline = self._get_generic_kg_pipeline(db_id)
                parse_start = perf_counter()
                # 根据内容类型处理内容
                if content_type == "file":
                    markdown_content = await process_file_to_markdown(item, params=params)
                    markdown_content_lines = markdown_content[:100].replace("\n", " ")
                    logger.info(f"Markdown content: {markdown_content_lines}...")
                else:  # URL
                    markdown_content = await process_url_to_markdown(item, params=params)
                parse_duration = round(perf_counter() - parse_start, 4)

                # 使用 LightRAG 插入内容
                insert_runner = getattr(self, "_ainsert_with_retry", None)
                if callable(insert_runner):
                    insert_observe = await insert_runner(rag, markdown_content, file_id, item_path)
                else:
                    logger.warning("_ainsert_with_retry not found on instance, falling back to module helper")
                    insert_observe = await _safe_ainsert_with_retry(rag, markdown_content, file_id, item_path)
                fallback_summary = {"inserted_nodes": 0, "inserted_edges": 0}
                if insert_observe.get("status") != "done":
                    # 兜底：LLM 图谱抽取失败时，仍尝试规则抽取
                    if os.getenv("LIGHTRAG_ENABLE_GENERIC_FALLBACK", "true").lower() == "true":
                        try:
                            fallback_summary = self._fallback_insert_generic_graph(
                                db_id, markdown_content, pipeline=db_pipeline
                            )
                        except Exception as fallback_err:  # noqa: BLE001
                            logger.warning(f"Fallback graph extraction failed for {db_id}: {fallback_err}")
                    if fallback_summary.get("inserted_nodes", 0) <= 0:
                        raise RuntimeError(insert_observe.get("error", "插入图谱失败"))

                # 兜底：若模型抽取后仍无节点，则对通用文本执行规则抽取，保证可视化图谱不为空
                if os.getenv("LIGHTRAG_ENABLE_GENERIC_FALLBACK", "true").lower() == "true":
                    try:
                        neo4j_uri, neo4j_username, neo4j_password = _get_neo4j_connection_config()
                        with GraphDatabase.driver(
                                neo4j_uri,
                                auth=(neo4j_username, neo4j_password),
                        ) as driver:
                            with driver.session() as session:
                                node_count_record = session.run(
                                    "MATCH (n {workspace: $workspace}) RETURN count(n) AS node_count",
                                    workspace=db_id,
                                ).single()
                                node_count = node_count_record["node_count"] if node_count_record else 0
                        if node_count == 0:
                            fallback_summary = self._fallback_insert_generic_graph(
                                db_id, markdown_content, pipeline=db_pipeline
                            )
                    except Exception as fallback_err:  # noqa: BLE001
                        logger.warning(f"Fallback graph extraction failed for {db_id}: {fallback_err}")

                logger.info(f"Inserted {content_type} {item} into LightRAG. Done.")

                # 更新状态为完成
                self.files_meta[file_id]["status"] = "done"
                self.files_meta[file_id]["observability"] = {
                    "parse_duration_sec": parse_duration,
                    "insert_duration_sec": insert_observe.get("duration_sec"),
                    "insert_attempt": insert_observe.get("attempt"),
                    "fallback_inserted_nodes": fallback_summary.get("inserted_nodes", 0),
                    "fallback_inserted_edges": fallback_summary.get("inserted_edges", 0),
                }
                self._save_metadata()
                file_record["status"] = "done"
                file_record["observability"] = self.files_meta[file_id]["observability"]

            except Exception as e:
                error_msg = str(e)
                logger.error(f"处理{content_type} {item} 失败: {error_msg}, {traceback.format_exc()}")
                self.files_meta[file_id]["status"] = "failed"
                self.files_meta[file_id]["error"] = error_msg
                self._save_metadata()
                file_record["status"] = "failed"
                file_record["error"] = error_msg
            finally:
                self._remove_from_processing_queue(file_id)

            processed_items_info.append(file_record)

        return processed_items_info

    async def add_image_embeddings(self, db_id: str, items: list[str], params: dict | None) -> list[dict]:
        pass

    async def rebuild_file_graph(self, db_id: str, file_id: str, params: dict | None = None) -> dict:
        """手动重建指定文件的图谱数据（不重新上传文件）。"""
        if db_id not in self.databases_meta:
            raise ValueError(f"Database {db_id} not found")
        if file_id not in self.files_meta:
            raise ValueError(f"File not found: {file_id}")

        file_meta = self.files_meta[file_id]
        if file_meta.get("database_id") != db_id:
            raise ValueError(f"File {file_id} does not belong to database {db_id}")

        item_path = file_meta.get("path")
        if not item_path:
            raise ValueError(f"File path is missing for {file_id}")

        params = params or {}
        content_type = params.get("content_type")
        if content_type not in {"file", "url"}:
            content_type = "url" if str(item_path).startswith(("http://", "https://")) else "file"

        rag = await self._get_lightrag_instance(db_id)
        if not rag:
            raise ValueError(f"Failed to get LightRAG instance for {db_id}")

        self._add_to_processing_queue(file_id)
        if file_id not in self.files_meta:
            raise ValueError(f"File not found during rebuild start: {file_id}")
        self.files_meta[file_id]["status"] = "processing"
        self.files_meta[file_id].pop("error", None)
        self._save_metadata()

        try:
            db_pipeline = self._get_generic_kg_pipeline(db_id)
            parse_start = perf_counter()
            if content_type == "file":
                markdown_content = await process_file_to_markdown(item_path, params=params)
            else:
                markdown_content = await process_url_to_markdown(item_path, params=params)
            parse_duration = round(perf_counter() - parse_start, 4)

            insert_runner = getattr(self, "_ainsert_with_retry", None)
            if callable(insert_runner):
                insert_observe = await insert_runner(rag, markdown_content, file_id, item_path)
            else:
                logger.warning("_ainsert_with_retry not found on instance, falling back to module helper")
                insert_observe = await _safe_ainsert_with_retry(rag, markdown_content, file_id, item_path)
            fallback_summary = {"inserted_nodes": 0, "inserted_edges": 0}
            if insert_observe.get("status") != "done":
                if os.getenv("LIGHTRAG_ENABLE_GENERIC_FALLBACK", "true").lower() == "true":
                    fallback_summary = self._fallback_insert_generic_graph(
                        db_id, markdown_content, pipeline=db_pipeline
                    )
                if fallback_summary.get("inserted_nodes", 0) <= 0:
                    raise RuntimeError(insert_observe.get("error", "重建图谱失败"))

            if os.getenv("LIGHTRAG_ENABLE_GENERIC_FALLBACK", "true").lower() == "true":
                neo4j_uri, neo4j_username, neo4j_password = _get_neo4j_connection_config()
                with GraphDatabase.driver(
                        neo4j_uri,
                        auth=(neo4j_username, neo4j_password),
                ) as driver:
                    with driver.session() as session:
                        node_count_record = session.run(
                            "MATCH (n {workspace: $workspace}) RETURN count(n) AS node_count",
                            workspace=db_id,
                        ).single()
                        node_count = node_count_record["node_count"] if node_count_record else 0
                if node_count == 0:
                    fallback_summary = self._fallback_insert_generic_graph(
                        db_id, markdown_content, pipeline=db_pipeline
                    )

            if file_id not in self.files_meta:
                raise ValueError(f"File deleted during rebuild: {file_id}")
            self.files_meta[file_id]["status"] = "done"
            self.files_meta[file_id]["observability"] = {
                "parse_duration_sec": parse_duration,
                "insert_duration_sec": insert_observe.get("duration_sec"),
                "insert_attempt": insert_observe.get("attempt"),
                "fallback_inserted_nodes": fallback_summary.get("inserted_nodes", 0),
                "fallback_inserted_edges": fallback_summary.get("inserted_edges", 0),
                "manual_rebuild": True,
            }
            self._save_metadata()
            return {"file_id": file_id, "status": "done", "observability": self.files_meta[file_id]["observability"]}
        except Exception as e:  # noqa: BLE001
            error_msg = str(e)
            logger.error(f"重建图谱失败 {db_id=}, {file_id=}: {error_msg}, {traceback.format_exc()}")
            if file_id in self.files_meta:
                self.files_meta[file_id]["status"] = "failed"
                self.files_meta[file_id]["error"] = error_msg
                self._save_metadata()
            raise
        finally:
            self._remove_from_processing_queue(file_id)

    async def aquery(self, query_text: str, db_id: str, **kwargs) -> str:
        """异步查询知识库"""
        rag = await self._get_lightrag_instance(db_id)
        if not rag:
            raise ValueError(f"Database {db_id} not found")

        try:
            # 设置查询参数
            params_dict = {
                              "mode": "mix",
                              "only_need_context": True,
                              "top_k": 10,
                          } | kwargs
            param = QueryParam(**params_dict)

            # 执行查询
            response = await rag.aquery(query_text, param)
            logger.debug(f"Query response: {response}")

            return response

        except Exception as e:
            logger.error(f"Query error: {e}, {traceback.format_exc()}")
            return ""

    async def delete_file(self, db_id: str, file_id: str) -> None:
        """删除文件"""
        rag = await self._get_lightrag_instance(db_id)
        if rag:
            try:
                # 使用 LightRAG 删除文档
                await rag.adelete_by_doc_id(file_id)
            except Exception as e:
                logger.error(f"Error deleting file {file_id} from LightRAG: {e}")

        # 删除文件记录
        if file_id in self.files_meta:
            del self.files_meta[file_id]
            self._save_metadata()

    async def get_file_basic_info(self, db_id: str, file_id: str) -> dict:
        """获取文件基本信息（仅元数据）"""
        if file_id not in self.files_meta:
            raise Exception(f"File not found: {file_id}")

        return {"meta": self.files_meta[file_id]}

    async def get_file_content(self, db_id: str, file_id: str) -> dict:
        """获取文件内容信息（chunks和lines）"""
        if file_id not in self.files_meta:
            raise Exception(f"File not found: {file_id}")

        # 使用 LightRAG 获取 chunks
        content_info = {"lines": []}
        rag = await self._get_lightrag_instance(db_id)
        if rag:
            try:
                # 获取文档的所有 chunks（兼容不同 LightRAG 存储实现）
                all_chunks = {}
                if hasattr(rag.text_chunks, "get_all"):
                    all_chunks = await rag.text_chunks.get_all()  # type: ignore[attr-defined]
                elif hasattr(rag.text_chunks, "all_keys"):
                    keys = await rag.text_chunks.all_keys()  # type: ignore[attr-defined]
                    for key in keys or []:
                        try:
                            value = await rag.text_chunks.get_by_id(key)  # type: ignore[attr-defined]
                            all_chunks[key] = value
                        except Exception:  # noqa: BLE001
                            continue
                else:
                    logger.warning("LightRAG text_chunks storage does not expose get_all/all_keys methods")
                    return content_info

                # 筛选属于该文档的 chunks
                doc_chunks = []
                for chunk_id, chunk_data in all_chunks.items():
                    if isinstance(chunk_data, dict) and chunk_data.get("full_doc_id") == file_id:
                        chunk_data["id"] = chunk_id
                        chunk_data["content_vector"] = []
                        doc_chunks.append(chunk_data)

                # 按 chunk_order_index 排序
                doc_chunks.sort(key=lambda x: x.get("chunk_order_index", 0))
                content_info["lines"] = doc_chunks
                return content_info

            except Exception as e:
                logger.error(f"Failed to get file content from LightRAG: {e}")
                content_info["lines"] = []
                return content_info

        return content_info

    async def get_file_info(self, db_id: str, file_id: str) -> dict:
        """获取文件完整信息（基本信息+内容信息）- 保持向后兼容"""
        if file_id not in self.files_meta:
            raise Exception(f"File not found: {file_id}")

        # 合并基本信息和内容信息
        basic_info = await self.get_file_basic_info(db_id, file_id)
        content_info = await self.get_file_content(db_id, file_id)

        return {**basic_info, **content_info}

    async def export_data(self, db_id: str, format: str = "csv", **kwargs) -> str:
        """
        使用 LightRAG 原生功能导出知识库数据。
        """
        logger.info(f"Exporting data for db_id {db_id} in format {format} with options {kwargs}")

        rag = await self._get_lightrag_instance(db_id)
        if not rag:
            raise ValueError(f"Failed to get LightRAG instance for {db_id}")

        export_dir = os.path.join(self.work_dir, db_id, "exports")
        os.makedirs(export_dir, exist_ok=True)

        timestamp = shanghai_now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"export_{db_id}_{timestamp}.{format}"
        output_filepath = os.path.join(export_dir, output_filename)

        include_vectors = kwargs.get('include_vectors', False)

        # 直接调用 lightrag 的异步导出功能
        await rag.aexport_data(
            output_path=output_filepath,
            file_format=format,
            include_vector_data=include_vectors
        )

        logger.info(f"Successfully created export file: {output_filepath}")
        return output_filepath
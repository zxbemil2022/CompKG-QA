import asyncio
import traceback
from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from src.storage.db.models import User
from server.utils.auth_middleware import get_admin_user
from server.services.tasker import TaskContext, tasker
from src import graph_base, knowledge_base
from src.models.chat import select_model
from src.utils.logging_config import logger

graph = APIRouter(prefix="/graph", tags=["graph"])
GRAPH_QA_MEMORY: dict[str, list[str]] = {}
AUTO_GRAPH_REBUILD_LOCKS: dict[str, asyncio.Lock] = {}
AUTO_GRAPH_REBUILD_TASK_BY_DB: dict[str, str] = {}


def _get_rebuild_lock(db_id: str) -> asyncio.Lock:
    lock = AUTO_GRAPH_REBUILD_LOCKS.get(db_id)
    if lock is None:
        lock = asyncio.Lock()
        AUTO_GRAPH_REBUILD_LOCKS[db_id] = lock
    return lock


def _collect_graph_diagnostics(db_id: str, total_nodes: int) -> dict:
    """避免把异常误判为空图，返回更细粒度诊断。"""
    try:
        kb_instance = knowledge_base._get_kb_for_database(db_id)
    except Exception as exc:  # noqa: BLE001
        return {
            "state": "storage_error",
            "message": f"无法读取知识库实例: {exc}",
            "file_stats": {},
        }

    statuses = {"done": 0, "failed": 0, "error": 0, "processing": 0, "pending": 0, "other": 0}
    total_files = 0
    for _, file_meta in kb_instance.files_meta.items():
        if file_meta.get("database_id") != db_id:
            continue
        total_files += 1
        status = str(file_meta.get("status", "pending")).lower()
        if status in statuses:
            statuses[status] += 1
        else:
            statuses["other"] += 1

    if total_nodes > 0:
        state = "graph_ready"
        message = "图谱已构建完成"
    elif total_files == 0:
        state = "no_documents"
        message = "未上传文档"
    elif statuses["processing"] > 0:
        state = "pipeline_running"
        message = "入库/抽取 pipeline 正在执行"
    elif statuses["failed"] > 0 or statuses["error"] > 0:
        state = "pipeline_failed"
        message = "存在失败文档，建议查看任务中心或重建图谱"
    elif statuses["done"] > 0:
        state = "possible_schema_mismatch"
        message = "文档已完成但图为空，可能存在 schema mismatch/部分写入失败"
    else:
        state = "unknown"
        message = "图谱为空，原因待确认"

    return {"state": state, "message": message, "file_stats": {"total": total_files, **statuses}}


# =============================================================================
# === 子图查询分组 ===
# =============================================================================


@graph.get("/lightrag/subgraph")
async def get_lightrag_subgraph(
    db_id: str = Query(..., description="数据库ID"),
    node_label: str = Query(..., description="节点标签或实体名称"),
    max_depth: int = Query(2, description="最大深度", ge=1, le=5),
    max_nodes: int = Query(100, description="最大节点数", ge=1, le=1000),
    current_user: User = Depends(get_admin_user),
):
    """
    使用 LightRAG 原生方法获取知识图谱子图

    Args:
        db_id: LightRAG 数据库实例ID
        node_label: 节点标签，用于查找起始节点，使用 "*" 获取全图
        max_depth: 子图的最大深度
        max_nodes: 返回的最大节点数量

    Returns:
        包含节点和边的知识图谱数据
    """
    try:
        logger.info(
            f"获取子图数据 - db_id: {db_id}, node_label: {node_label}, max_depth: {max_depth}, max_nodes: {max_nodes}"
        )

        # 检查是否是 LightRAG 数据库
        if not knowledge_base.is_lightrag_database(db_id):
            raise HTTPException(
                status_code=400, detail=f"数据库 {db_id} 不是 LightRAG 类型，图谱功能仅支持 LightRAG 知识库"
            )

        # 获取 LightRAG 实例
        rag_instance = await knowledge_base._get_lightrag_instance(db_id)
        if not rag_instance:
            raise HTTPException(status_code=404, detail=f"LightRAG 数据库 {db_id} 不存在或无法访问")

        # 使用 LightRAG 的原生 get_knowledge_graph 方法
        knowledge_graph = await rag_instance.get_knowledge_graph(
            node_label=node_label, max_depth=max_depth, max_nodes=max_nodes
        )

        # 将 LightRAG 的 KnowledgeGraph 格式转换为前端需要的格式
        nodes = []
        for node in knowledge_graph.nodes:
            nodes.append(
                {
                    "id": node.id,
                    "labels": node.labels,
                    "entity_type": node.properties.get("entity_type", "unknown"),
                    "properties": node.properties,
                }
            )

        edges = []
        for edge in knowledge_graph.edges:
            edges.append(
                {
                    "id": edge.id,
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type,
                    "properties": edge.properties,
                }
            )

        result = {
            "success": True,
            "data": {
                "nodes": nodes,
                "edges": edges,
                "is_truncated": knowledge_graph.is_truncated,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
            },
        }
        result["data"]["diagnostics"] = _collect_graph_diagnostics(db_id, len(nodes))

        logger.info(f"成功获取子图 - 节点数: {len(nodes)}, 边数: {len(edges)}")
        return result

    except HTTPException:
        # 重新抛出 HTTP 异常
        raise
    except Exception as e:
        logger.error(f"获取子图数据失败: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取子图数据失败: {str(e)}")


@graph.get("/lightrag/databases")
async def get_lightrag_databases(current_user: User = Depends(get_admin_user)):
    """
    获取所有可用的 LightRAG 数据库

    Returns:
        可用的 LightRAG 数据库列表
    """
    try:
        lightrag_databases = knowledge_base.get_lightrag_databases()
        return {"success": True, "data": {"databases": lightrag_databases}}

    except Exception as e:
        logger.error(f"获取 LightRAG 数据库列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取 LightRAG 数据库列表失败: {str(e)}")


# =============================================================================
# === 节点管理分组 ===
# =============================================================================


@graph.get("/lightrag/labels")
async def get_lightrag_labels(
    db_id: str = Query(..., description="数据库ID"), current_user: User = Depends(get_admin_user)
):
    """
    获取知识图谱中的所有标签

    Args:
        db_id: LightRAG 数据库实例ID

    Returns:
        图谱中所有可用的标签列表
    """
    try:
        logger.info(f"获取图谱标签 - db_id: {db_id}")

        # 检查是否是 LightRAG 数据库
        if not knowledge_base.is_lightrag_database(db_id):
            raise HTTPException(
                status_code=400, detail=f"数据库 {db_id} 不是 LightRAG 类型，图谱功能仅支持 LightRAG 知识库"
            )

        # 获取 LightRAG 实例
        rag_instance = await knowledge_base._get_lightrag_instance(db_id)
        if not rag_instance:
            raise HTTPException(status_code=404, detail=f"LightRAG 数据库 {db_id} 不存在或无法访问")

        # 使用 LightRAG 的原生方法获取所有标签
        labels = await rag_instance.get_graph_labels()

        return {"success": True, "data": {"labels": labels}}

    except HTTPException:
        # 重新抛出 HTTP 异常
        raise
    except Exception as e:
        logger.error(f"获取图谱标签失败: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取图谱标签失败: {str(e)}")


@graph.get("/neo4j/nodes")
async def get_neo4j_nodes(
    kgdb_name: str = Query(..., description="知识图谱数据库名称"),
    num: int = Query(100, description="节点数量", ge=1, le=1000),
    subject: str | None = Query(None, description="408学科过滤"),
    current_user: User = Depends(get_admin_user),
):
    """
    获取图谱节点样本数据
    """
    try:
        logger.debug(f"Get graph nodes in {kgdb_name} with {num} nodes")

        if not graph_base.is_running():
            raise HTTPException(status_code=400, detail="图数据库未启动")

        result = graph_base.get_sample_nodes(kgdb_name, num, subject=subject)

        return {"success": True, "result": result, "message": "success"}

    except Exception as e:
        logger.error(f"获取图节点数据失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取图节点数据失败: {str(e)}")


@graph.get("/neo4j/node")
async def get_neo4j_node(
        entity_name: str = Query(..., description="实体名称"),
        subject: str | None = Query(None, description="408学科过滤，例如：数据结构/操作系统/计算机网络/计算机组成原理"),
        current_user: User = Depends(get_admin_user),
):
    """
    根据实体名称查询图节点
    """
    try:
        if not graph_base.is_running():
            raise HTTPException(status_code=400, detail="图数据库未启动")

        result = graph_base.query_node(keyword=entity_name, subject=subject)

        return {"success": True, "result": result, "message": "success"}

    except Exception as e:
        logger.error(f"查询图节点失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"查询图节点失败: {str(e)}")


# =============================================================================
# === 边管理分组 ===
# =============================================================================

# 可以在这里添加边相关的管理功能

# =============================================================================
# === 图谱分析分组 ===
# =============================================================================


@graph.get("/lightrag/stats")
async def get_lightrag_stats(
        db_id: str = Query(..., description="数据库ID"),
        current_user: User = Depends(get_admin_user),
):
    """
    获取知识图谱统计信息
    """
    try:
        logger.info(f"获取图谱统计信息 - db_id: {db_id}")

        # 检查是否是 LightRAG 数据库
        if not knowledge_base.is_lightrag_database(db_id):
            raise HTTPException(
                status_code=400, detail=f"数据库 {db_id} 不是 LightRAG 类型，图谱功能仅支持 LightRAG 知识库"
            )

        # 获取 LightRAG 实例
        rag_instance = await knowledge_base._get_lightrag_instance(db_id)
        if not rag_instance:
            raise HTTPException(status_code=404, detail=f"LightRAG 数据库 {db_id} 不存在或无法访问")

        # 通过获取全图来统计节点和边的数量
        knowledge_graph = await rag_instance.get_knowledge_graph(
            node_label="*",
            max_depth=1,
            max_nodes=10000,  # 设置较大值以获取完整统计
        )

        # 统计实体类型分布
        entity_types = {}
        for node in knowledge_graph.nodes:
            entity_type = node.properties.get("entity_type", "unknown")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        entity_types_list = [
            {"type": k, "count": v} for k, v in sorted(entity_types.items(), key=lambda x: x[1], reverse=True)
        ]

        result = {
            "success": True,
            "data": {
                "total_nodes": len(knowledge_graph.nodes),
                "total_edges": len(knowledge_graph.edges),
                "entity_types": entity_types_list,
                "is_truncated": knowledge_graph.is_truncated,
            },
        }
        result["data"]["diagnostics"] = _collect_graph_diagnostics(db_id, len(knowledge_graph.nodes))
        return result

    except HTTPException:
        # 重新抛出 HTTP 异常
        raise
    except Exception as e:
        logger.error(f"获取图谱统计信息失败: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取图谱统计信息失败: {str(e)}")


@graph.get("/neo4j/info")
async def get_neo4j_info(current_user: User = Depends(get_admin_user)):
    """获取Neo4j图数据库信息"""
    try:
        graph_info = graph_base.get_graph_info()
        if graph_info is None:
            raise HTTPException(status_code=400, detail="图数据库获取出错")
        return {"success": True, "data": graph_info}
    except Exception as e:
        logger.error(f"获取图数据库信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取图数据库信息失败: {str(e)}")


@graph.post("/lightrag/rebuild-if-empty")
@graph.post("/lightrag/rebuild-if-empty/")
@graph.post("/lightrag/rebuild_if_empty")
async def rebuild_lightrag_if_empty(
    db_id: str = Body(..., embed=True),
    force: bool = Body(False, embed=True),
    current_user: User = Depends(get_admin_user),
):
    """显式触发补建（POST 有副作用，避免 GET 写操作）。"""
    if not knowledge_base.is_lightrag_database(db_id):
        raise HTTPException(status_code=400, detail=f"数据库 {db_id} 不是 LightRAG 类型")

    lock = _get_rebuild_lock(db_id)
    async with lock:
        existing_task_id = AUTO_GRAPH_REBUILD_TASK_BY_DB.get(db_id)
        if existing_task_id:
            existing_task = await tasker.get_task(existing_task_id)
            if existing_task and existing_task.get("status") in {"pending", "running"}:
                return {
                    "success": True,
                    "status": "already_queued",
                    "task_id": existing_task_id,
                    "message": "已有补建任务在执行",
                }

        rag_instance = await knowledge_base._get_lightrag_instance(db_id)
        if not rag_instance:
            raise HTTPException(status_code=404, detail=f"LightRAG 数据库 {db_id} 不存在或无法访问")

        graph_data = await rag_instance.get_knowledge_graph(node_label="*", max_depth=1, max_nodes=200)
        if (not force) and len(graph_data.nodes) > 0:
            return {
                "success": True,
                "status": "skipped",
                "message": "图谱非空，跳过补建",
                "total_nodes": len(graph_data.nodes),
            }

        kb_instance = knowledge_base._get_kb_for_database(db_id)
        candidate_file_ids = []
        seen_paths = set()
        for file_id, file_meta in kb_instance.files_meta.items():
            if file_meta.get("database_id") != db_id:
                continue
            if file_meta.get("status") in {"done", "failed", "error"}:
                file_path = str(file_meta.get("path") or "")
                if file_path and file_path in seen_paths:
                    continue
                if file_path:
                    seen_paths.add(file_path)
                candidate_file_ids.append(file_id)

        if not candidate_file_ids:
            return {"success": True, "status": "skipped", "message": "没有可补建的文档"}

        async def run_auto_rebuild(context: TaskContext):
            await context.set_message("开始补建图谱")
            total = len(candidate_file_ids)
            results = []
            for idx, file_id in enumerate(candidate_file_ids, 1):
                await context.raise_if_cancelled()
                await context.set_progress((idx / total) * 100.0, f"补建 {idx}/{total}")
                try:
                    rebuild_result = await knowledge_base.rebuild_file_graph(db_id, file_id, params={})
                    results.append({"file_id": file_id, "status": "done", "result": rebuild_result})
                except Exception as exc:  # noqa: BLE001
                    results.append({"file_id": file_id, "status": "failed", "error": str(exc)})
            await context.set_result({"db_id": db_id, "items": results})
            return {"db_id": db_id, "items": results}

        task = await tasker.enqueue(
            name=f"显式补建图谱({db_id})",
            task_type="knowledge_rebuild_graph",
            payload={"db_id": db_id, "file_ids": candidate_file_ids, "source": "manual_rebuild_if_empty"},
            coroutine=run_auto_rebuild,
        )
        AUTO_GRAPH_REBUILD_TASK_BY_DB[db_id] = task.id
        return {
            "success": True,
            "status": "queued",
            "task_id": task.id,
            "message": "补建任务已提交",
            "candidate_files": len(candidate_file_ids),
        }


@graph.post("/neo4j/index-entities")
async def index_neo4j_entities(data: dict = Body(default={}), current_user: User = Depends(get_admin_user)):
    """为Neo4j图谱节点添加嵌入向量索引"""
    try:
        if not graph_base.is_running():
            raise HTTPException(status_code=400, detail="图数据库未启动")

        # 获取参数或使用默认值
        kgdb_name = data.get("kgdb_name", "neo4j")

        # 调用GraphDatabase的add_embedding_to_nodes方法
        count = graph_base.add_embedding_to_nodes(kgdb_name=kgdb_name)

        return {
            "success": True,
            "status": "success",
            "message": f"已成功为{count}个节点添加嵌入向量",
            "indexed_count": count,
        }
    except Exception as e:
        logger.error(f"索引节点失败: {e}")
        raise HTTPException(status_code=500, detail=f"索引节点失败: {str(e)}")


@graph.post("/neo4j/add-entities")
async def add_neo4j_entities(
    file_path: str = Body(...),
    kgdb_name: str | None = Body(None),
    skip_embedding: bool = Body(False),
    current_user: User = Depends(get_admin_user),
):
    """通过JSONL文件添加图谱实体到Neo4j"""
    try:
        if not file_path.endswith(".jsonl"):
            raise HTTPException(status_code=400, detail="文件格式错误，请上传jsonl文件")

        await graph_base.jsonl_file_add_entity(file_path, kgdb_name, with_embedding=not skip_embedding)
        return {"success": True, "message": "实体添加成功", "status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加实体失败: {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"添加实体失败: {e}")


@graph.post("/neo4j/auto-build-computer-kg")
async def auto_build_computer_kg(data: dict = Body(default={}), current_user: User = Depends(get_admin_user)):
    """自动构建计算机专业知识图谱（支持 JSON/文本输入）。"""
    try:
        if not graph_base.is_running():
            raise HTTPException(status_code=400, detail="图数据库未启动")

        kgdb_name = data.get("kgdb_name", "neo4j")
        clear_existing = bool(data.get("clear_existing", False))
        content = (data.get("content") or "").strip()
        source_name = data.get("source_name")
        file_path = data.get("file_path")

        if not content:
            if file_path:
                target = Path(file_path)
            else:
                target = Path("examples/cs408/cs408_auto_sample.json")
            if not target.exists():
                raise HTTPException(status_code=404, detail=f"示例文件不存在: {target}")
            content = target.read_text(encoding="utf-8")
            source_name = source_name or str(target)

        result = await graph_base.auto_build_computer_knowledge_graph(
            content=content,
            kgdb_name=kgdb_name,
            clear_existing=clear_existing,
            source_name=source_name,
       )

        return {
            "success": True,
            "status": "success",
            "message": "计算机专业知识图谱自动构建完成",
            "data": result,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"自动构建计算机知识图谱失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"自动构建计算机知识图谱失败: {str(e)}")


@graph.post("/neo4j/auto-build-408-subject-kgs")
async def auto_build_408_subject_kgs(data: dict = Body(default={}),current_user: User = Depends(get_admin_user)):
    """按 408 学科（数据结构/操作系统/计算机网络/组成原理）自动生成各自子图谱。"""
    try:
        if not graph_base.is_running():
            raise HTTPException(status_code=400, detail="图数据库未启动")

        kgdb_name = data.get("kgdb_name", "neo4j")
        clear_existing = bool(data.get("clear_existing", False))
        content = (data.get("content") or "").strip()
        file_path = data.get("file_path") or "examples/cs408/cs408_auto_sample.json"

        if not content:
            target = Path(file_path)
            if not target.exists():
                raise HTTPException(status_code=404, detail=f"文件不存在: {target}")
            content = target.read_text(encoding="utf-8")

        result = await graph_base.auto_build_cs408_subject_graphs(
            content=content,
            kgdb_name=kgdb_name,
            clear_existing=clear_existing,
        )
        return {
            "success": True,
            "status": "success",
            "message": "408学科知识图谱构建完成，可按学科进行专项问答",
            "data": result,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"构建408学科知识图谱失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"构建408学科知识图谱失败: {str(e)}")


@graph.get("/neo4j/subjects")
async def get_neo4j_subjects(
    kgdb_name: str = Query("neo4j", description="知识图谱数据库名称"),
    current_user: User = Depends(get_admin_user),
):
    """获取当前图谱中的学科标签统计。"""
    try:
        if not graph_base.is_running():
            raise HTTPException(status_code=400, detail="图数据库未启动")
        return {"success": True, "data": {"subjects": graph_base.list_subject_tags(kgdb_name=kgdb_name)}}
    except HTTPException:
            raise
    except Exception as e:
        logger.error(f"获取学科标签失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取学科标签失败: {str(e)}")


@graph.get("/neo4j/capabilities")
async def get_neo4j_capabilities(current_user: User = Depends(get_admin_user)):
    """返回图谱能力矩阵，便于前端做功能降级与专业化诊断。"""
    try:
        return {
            "success": True,
            "data": {
                "subjects_api": True,
                "auto_build_computer_kg": True,
                "auto_build_408_subject_kgs": True,
                "add_entities_skip_embedding": True,
                "builtin_expert_seed_supported": True,
            },
        }
    except Exception as e:
        logger.error(f"获取图谱能力失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取图谱能力失败: {str(e)}")


@graph.post("/neo4j/edits/submit")
async def submit_graph_edit(data: dict = Body(default={}), current_user: User = Depends(get_admin_user)):
    """提交图谱人工纠错/新增请求（进入审核流）。"""
    try:
        payload = data.get("payload") or {}
        result = graph_base.submit_graph_edit(edit_payload=payload,
                                                      submitter=str(getattr(current_user, "id", "admin")))
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"提交图谱编辑失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"提交图谱编辑失败: {str(e)}")

@graph.get("/neo4j/edits")
async def list_graph_edits(
    status: str | None = Query(None, description="pending/approved/rejected"),
    current_user: User = Depends(get_admin_user),
):
    """查看图谱编辑审核队列。"""
    try:
        return {"success": True, "data": {"items": graph_base.list_graph_edits(status=status)}}
    except Exception as e:
        logger.error(f"获取图谱编辑队列失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取图谱编辑队列失败: {str(e)}")


@graph.post("/neo4j/edits/review")
async def review_graph_edit(data: dict = Body(default={}), current_user: User = Depends(get_admin_user)):
    """审核图谱编辑（approve/reject），approve后自动回写种子库/本体。"""
    try:
        edit_id = data.get("edit_id")
        action = data.get("action")
        if not edit_id or action not in {"approve", "reject"}:
            raise HTTPException(status_code=400, detail="edit_id/action 参数错误")
        result = graph_base.review_graph_edit(
            edit_id=edit_id,
            action=action,
            reviewer=str(getattr(current_user, "id", "admin")),
        )
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"审核图谱编辑失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"审核图谱编辑失败: {str(e)}")


@graph.get("/neo4j/learning-path")
async def get_learning_path(
    concept: str = Query(..., description="目标知识点"),
    subject: str | None = Query(None, description="可选学科"),
    kgdb_name: str = Query("neo4j", description="图数据库名称"),
    current_user: User = Depends(get_admin_user),
):
    """学习路径推荐：按前置知识关系返回建议学习顺序。"""
    try:
        path = graph_base.recommend_learning_path(concept=concept, kgdb_name=kgdb_name, subject=subject)
        return {"success": True, "data": {"concept": concept, "path": path}}
    except Exception as e:
        logger.error(f"获取学习路径失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取学习路径失败: {str(e)}")


@graph.get("/neo4j/association")
async def get_knowledge_association(
    concept: str = Query(..., description="知识点"),
    subject: str | None = Query(None, description="可选学科"),
    kgdb_name: str = Query("neo4j", description="图数据库名称"),
    current_user: User = Depends(get_admin_user),
):
    """知识关联分析：返回关联节点与关系分布。"""
    try:
        result = graph_base.analyze_knowledge_association(concept=concept, kgdb_name=kgdb_name, subject=subject)
        return {"success": True, "data": {"concept": concept, **result}}
    except Exception as e:
        logger.error(f"获取知识关联分析失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取知识关联分析失败: {str(e)}")


@graph.get("/neo4j/ask")
async def ask_graph_question(
        question: str = Query(..., description="用户问题"),
        subject: str | None = Query(None, description="可选学科"),
        kgdb_name: str = Query("neo4j", description="图数据库名称"),
        session_id: str = Query("default", description="会话ID，用于多轮记忆"),
        current_user: User = Depends(get_admin_user),
):
    """
    图谱问答一体化接口：
    输入问题 -> GraphRAG检索 -> 返回答案 + 相关子图 + 高亮节点。
    """
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="question 不能为空")

        memory = GRAPH_QA_MEMORY.get(session_id, [])[-6:]
        graph_result = graph_base.query_node(
            keyword=question,
            kgdb_name=kgdb_name,
            subject=subject,
            return_format="graph",
            hops=2,
            max_entities=8,
        )
        triple_result = graph_base.query_node(
            keyword=question,
            kgdb_name=kgdb_name,
            subject=subject,
            return_format="triples",
            hops=2,
            max_entities=8,
        )
        triples = triple_result.get("triples", []) if isinstance(triple_result, dict) else []
        vector_evidence = []
        try:
            retrievers = knowledge_base.get_retrievers()
            for db_id, info in retrievers.items():
                retriever = info.get("retriever")
                if retriever is None:
                    continue
                if asyncio.iscoroutinefunction(retriever):
                    docs = await retriever(question, "", "")
                else:
                    docs = retriever(question, "", "")
                    if isinstance(docs, list):
                        for item in docs[:2]:
                            text = str(item).strip()
                            if text:
                               vector_evidence.append(
                                  {
                                    "source_db": db_id,
                                    "snippet": text[:220],
                                 }
                            )
        except Exception as e:
            logger.warning(f"图谱问答向量检索失败，已降级为图谱检索: {e}")

            highlight_nodes = []
            for t in triples[:8]:
                if len(t) >= 3:
                   highlight_nodes.extend([str(t[0]), str(t[2])])
            highlight_nodes = list(dict.fromkeys(highlight_nodes))

            reasoning_path = [f"Step {idx}: {h} --{r}--> {t}" for idx, (h, r, t) in enumerate(triples[:6], 1)]
            derivation_chain = " -> ".join(
                [str(triples[0][0])] + [str(item[2]) for item in triples[:3] if len(item) >= 3]
            ) if triples else ""

            if triples or vector_evidence:
                parts = []
                if triples:
                     parts.append("基于图谱证据可得：\n" + "\n".join(reasoning_path[:4]))
                if vector_evidence:
                    vec_lines = [f"- {ev['snippet']}" for ev in vector_evidence[:3]]
                    parts.append("结合知识库检索片段：\n" + "\n".join(vec_lines))
                answer = "\n\n".join(parts)
            else:
                # 图谱无命中时尝试模型增强
                answer = "图谱暂无直接证据。"
                try:
                    from src import config

                    provider, model_name = config.default_model.split("/", 1)
                    model = select_model(provider, model_name)
                    resp = model.call(
                         [
                            {"role": "system","content": "你是计算机专业助教，请给出简洁专业回答，并优先使用检索证据。"},
                            {"role": "user","content": f"学科：{subject or '综合'}\n历史问题：{memory or ['无']}\n当前问题：{question}\n图谱证据：{triples[:4]}\n向量证据：{vector_evidence[:3]}"},
                            ],
                            stream=False,
                    )
                    answer = f"{answer}\n模型增强回答：{getattr(resp, 'content', str(resp))}"
                except Exception as e:
                    logger.warning(f"图谱问答LLM增强失败: {e}")

            try:
                learning_path = graph_base.recommend_learning_path(
                    concept=question, kgdb_name=kgdb_name, subject=subject
                )
            except Exception:
                learning_path = []

            accuracy_analysis = {
                "level": "high" if len(triples) >= 6 else "medium" if len(triples) >= 3 else "low",
                "evidence_count": len(triples),
                "advice": "证据较少，建议结合教材与题库验证" if len(triples) < 3 else "证据充分，可用于学习路径规划",
            }

            memory.append(f"Q: {question} | A: {answer[:120]}")
            GRAPH_QA_MEMORY[session_id] = memory[-8:]

            return {
                "success": True,
                "data": {
                "answer": answer,
                "triples": triples[:10],
                "highlight_nodes": highlight_nodes,
                "graph": graph_result,
                "question": question,
                "subject": subject or "综合",
                "session_id": session_id,
                "memory_messages": GRAPH_QA_MEMORY.get(session_id, []),
                "reasoning_path": reasoning_path,
                "derivation_chain": derivation_chain or "暂无可推导链",
                "learning_path_recommendation": learning_path,
                "accuracy_analysis": accuracy_analysis,
                "vector_evidence": vector_evidence[:6],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"图谱问答失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"图谱问答失败: {str(e)}")
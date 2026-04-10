import asyncio
import os
import traceback
from urllib.parse import quote, unquote
from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.responses import FileResponse as StarletteFileResponse

from src.storage.db.models import User
from server.utils.auth_middleware import get_admin_user
from server.services.tasker import TaskContext, tasker
from src import config, knowledge_base
from src.knowledge.indexing import SUPPORTED_FILE_EXTENSIONS, is_supported_file_extension, process_file_to_markdown
from src.knowledge.config.domain_entity_config import get_domain_entity_relation_config
from src.knowledge.cs408_governance import audit_cs408_dataset, upgrade_cs408_dataset
from src.knowledge.utils import calculate_content_hash
from src.models.embed import test_embedding_model_status, test_all_embedding_models_status
from src.models.chat import select_model
from src.utils import hashstr, logger

knowledge = APIRouter(prefix="/knowledge", tags=["knowledge"])

# =============================================================================
# === 数据库管理分组 ===
# =============================================================================


@knowledge.get("/databases")
async def get_databases(current_user: User = Depends(get_admin_user)):
    """获取所有知识库"""
    try:
        database = knowledge_base.get_databases()
        return database
    except Exception as e:
        logger.error(f"获取数据库列表失败 {e}, {traceback.format_exc()}")
        return {"message": f"获取数据库列表失败 {e}", "databases": []}


@knowledge.post("/databases")
async def create_database(
    database_name: str = Body(...),
    description: str = Body(...),
    embed_model_name: str = Body(...),
    kb_type: str = Body("lightrag"),
    additional_params: dict = Body({}),
    llm_info: dict = Body(None),
    current_user: User = Depends(get_admin_user),
):
    """创建知识库"""
    logger.debug(
        f"Create database {database_name} with kb_type {kb_type}, "
        f"additional_params {additional_params}, llm_info {llm_info}"
    )
    try:
        embed_info = config.embed_model_names[embed_model_name]
        database_info = await knowledge_base.create_database(
            database_name, description, kb_type=kb_type, embed_info=embed_info, llm_info=llm_info, **additional_params
        )

        # 需要重新加载所有智能体，因为工具刷新了
        from src.agents import agent_manager

        await agent_manager.reload_all()

        return database_info
    except Exception as e:
        logger.error(f"创建数据库失败 {e}, {traceback.format_exc()}")
        return {"message": f"创建数据库失败 {e}", "status": "failed"}


@knowledge.get("/databases/{db_id}")
async def get_database_info(db_id: str, current_user: User = Depends(get_admin_user)):
    """获取知识库详细信息"""
    database = knowledge_base.get_database_info(db_id)
    if database is None:
        raise HTTPException(status_code=404, detail="Database not found")
    return database


@knowledge.get("/ontology/domains")
async def get_ontology_domains(current_user: User = Depends(get_admin_user)):
    """获取可选领域本体配置。"""
    return {"domains": knowledge_base.get_supported_ontology_domains()}


@knowledge.post("/cs408/audit")
async def audit_cs408(
    payload: dict = Body(default={"dataset_path": "examples/cs408/cs408_expert_seed.jsonl"}),
    current_user: User = Depends(get_admin_user),
):
    """离线审计：用于发布前数据质量 gate。"""
    try:
        dataset_path = payload.get("dataset_path") or "examples/cs408/cs408_expert_seed.jsonl"
        return {"status": "ok", "report": audit_cs408_dataset(dataset_path)}
    except Exception as e:
        logger.error(f"cs408 audit 失败: {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"cs408 audit 失败: {e}")


@knowledge.post("/cs408/upgrade")
async def upgrade_cs408(
    payload: dict = Body(default={"dataset_path": "examples/cs408/cs408_expert_seed.jsonl", "output_path": None}),
    current_user: User = Depends(get_admin_user),
):
    """离线升级：规范化 + 去重。"""
    try:
        dataset_path = payload.get("dataset_path") or "examples/cs408/cs408_expert_seed.jsonl"
        output_path = payload.get("output_path")
        result = upgrade_cs408_dataset(dataset_path, output_path=output_path)
        report = audit_cs408_dataset(result["output_path"])
        return {"status": "ok", "upgrade": result, "post_audit": report}
    except Exception as e:
        logger.error(f"cs408 upgrade 失败: {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"cs408 upgrade 失败: {e}")


@knowledge.get("/databases/{db_id}/ontology")
async def get_database_ontology(db_id: str, current_user: User = Depends(get_admin_user)):
    """获取数据库的领域本体设置（仅 lightrag）。"""
    try:
        return await knowledge_base.get_database_ontology(db_id)
    except Exception as e:
        logger.error(f"获取本体配置失败: {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"获取本体配置失败: {e}")


@knowledge.put("/databases/{db_id}/ontology")
async def update_database_ontology(
    db_id: str,
    domain: str = Body("computer"),
    entity_types: list[str] = Body(default=[]),
    relation_types: list[str] = Body(default=[]),
    kg_ner_plugin: str = Body("rule"),
    kg_re_plugin: str = Body("rule"),
    kg_ner_model_spec: str = Body(""),
    kg_re_model_spec: str = Body(""),
    kg_ner_llm_enabled: bool = Body(False),
    kg_re_llm_enabled: bool = Body(False),
    current_user: User = Depends(get_admin_user),
):
    """更新数据库的领域本体设置（仅 lightrag）。"""
    try:
        # 若只改 domain 且未提供实体/关系，则自动回填默认值
        resolved = get_domain_entity_relation_config(domain)
        updated = await knowledge_base.update_database_ontology(
            db_id=db_id,
            domain=domain,
            entity_types=entity_types or resolved["entity_types"],
            relation_types=relation_types or resolved["relation_types"],
            kg_ner_plugin=kg_ner_plugin,
            kg_re_plugin=kg_re_plugin,
            kg_ner_model_spec=kg_ner_model_spec,
            kg_re_model_spec=kg_re_model_spec,
            kg_ner_llm_enabled=kg_ner_llm_enabled,
            kg_re_llm_enabled=kg_re_llm_enabled,
        )
        return {"message": "本体配置更新成功", "ontology": updated}
    except Exception as e:
        logger.error(f"更新本体配置失败: {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"更新本体配置失败: {e}")


@knowledge.put("/databases/{db_id}")
async def update_database_info(
    db_id: str, name: str = Body(...), description: str = Body(...), current_user: User = Depends(get_admin_user)
):
    """更新知识库信息"""
    logger.debug(f"Update database {db_id} info: {name}, {description}")
    try:
        database = await knowledge_base.update_database(db_id, name, description)
        return {"message": "更新成功", "database": database}
    except Exception as e:
        logger.error(f"更新数据库失败 {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"更新数据库失败: {e}")


@knowledge.delete("/databases/{db_id}")
async def delete_database(db_id: str, current_user: User = Depends(get_admin_user)):
    """删除知识库"""
    logger.debug(f"Delete database {db_id}")
    try:
        await knowledge_base.delete_database(db_id)

        # 需要重新加载所有智能体，因为工具刷新了
        from src.agents import agent_manager

        await agent_manager.reload_all()

        return {"message": "删除成功"}
    except Exception as e:
        logger.error(f"删除数据库失败 {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"删除数据库失败: {e}")


@knowledge.get("/databases/{db_id}/export")
async def export_database(
    db_id: str,
    format: str = Query("csv", enum=["csv", "xlsx", "md", "txt"]),
    include_vectors: bool = Query(False, description="是否在导出中包含向量数据"),
    current_user: User = Depends(get_admin_user),
):
    """导出知识库数据"""
    logger.debug(f"Exporting database {db_id} with format {format}")
    try:
        file_path = await knowledge_base.export_data(db_id, format=format, include_vectors=include_vectors)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Exported file not found.")

        media_types = {
            "csv": "text/csv",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "md": "text/markdown",
            "txt": "text/plain",
        }
        media_type = media_types.get(format, "application/octet-stream")

        return FileResponse(path=file_path, filename=os.path.basename(file_path), media_type=media_type)
    except NotImplementedError as e:
        logger.warning(f"A disabled feature was accessed: {e}")
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"导出数据库失败 {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"导出数据库失败: {e}")


# =============================================================================
# === 文档管理分组 ===
# =============================================================================


@knowledge.post("/databases/{db_id}/documents")
async def add_documents(
    db_id: str, items: list[str] = Body(...), params: dict = Body(...), current_user: User = Depends(get_admin_user)
):
    """添加文档到知识库"""
    logger.debug(f"Add documents for db_id {db_id}: {items} {params=}")

    content_type = params.get("content_type", "file")
    auto_graph_ingest = bool(params.get("auto_graph_ingest", False))
    graph_db_id = params.get("graph_db_id")

    # 安全检查：验证文件路径
    if content_type == "file":
        from src.knowledge.utils.kb_utils import validate_file_path

        for item in items:
            try:
                validate_file_path(item, db_id)
            except ValueError as e:
                raise HTTPException(status_code=403, detail=str(e))

    async def run_ingest(context: TaskContext):
        await context.set_message("任务初始化")
        await context.set_progress(5.0, "准备处理文档")

        total = len(items)
        processed_items = []
        source_is_lightrag = knowledge_base.is_lightrag_database(db_id)
        target_is_valid_lightrag = bool(graph_db_id) and knowledge_base.is_lightrag_database(graph_db_id)

        try:
            # 逐个处理文档并更新进度
            for idx, item in enumerate(items, 1):
                await context.raise_if_cancelled()

                # 更新进度
                progress = 5.0 + (idx / total) * 90.0  # 5% ~ 95%
                await context.set_progress(progress, f"正在处理第 {idx}/{total} 个文档")

                file_path_obj = Path(item)
                file_ext = file_path_obj.suffix.lower()
                # 处理单个文档
                result = await knowledge_base.add_content(db_id, [item], params=params)

                for record in result:
                    graph_sync = {
                        "enabled": auto_graph_ingest,
                        "source_is_lightrag": source_is_lightrag,
                        "target_graph_db_id": graph_db_id,
                        "status": "skipped",
                        "message": "未启用图谱同步",
                    }

                    should_sync_to_graph = (
                            auto_graph_ingest and not source_is_lightrag and target_is_valid_lightrag and record.get("status") == "done"
                    )

                    if source_is_lightrag:
                        graph_sync["status"] = "done"
                        graph_sync["message"] = "当前知识库类型为 lightrag，文档已自动进行知识图谱抽取"
                    elif auto_graph_ingest and not target_is_valid_lightrag:
                        graph_sync["status"] = "failed"
                        graph_sync["message"] = "graph_db_id 未配置或不是 lightrag 类型知识库，无法执行图谱同步"
                    elif should_sync_to_graph:
                        sync_params = {
                            "content_type": content_type,
                        }
                        graph_result = await knowledge_base.add_content(graph_db_id, [item], params=sync_params)
                        synced_item = graph_result[0] if graph_result else {}
                        if synced_item.get("status") == "done":
                            graph_sync["status"] = "done"
                            graph_sync["message"] = f"已同步到图谱知识库 {graph_db_id}"
                        else:
                            graph_sync["status"] = "failed"
                            graph_sync["message"] = synced_item.get("error", "图谱同步失败")

                    record["graph_sync"] = graph_sync
                    processed_items.append(record)

        except asyncio.CancelledError:
            await context.set_progress(100.0, "任务已取消")
            raise

        item_type = "URL" if content_type == "url" else "文件"
        failed_count = len([_p for _p in processed_items if _p.get("status") == "failed"])
        graph_sync_failed_count = len([_p for _p in processed_items if _p.get("graph_sync", {}).get("status") == "failed"])
        summary = {
            "db_id": db_id,
            "item_type": item_type,
            "submitted": len(processed_items),
            "failed": failed_count,
            "graph_sync_failed": graph_sync_failed_count,
        }
        message = f"{item_type}处理完成，失败 {failed_count} 个" if failed_count else f"{item_type}处理完成"
        await context.set_result(summary | {"items": processed_items})
        await context.set_progress(100.0, message)
        return summary | {"items": processed_items}

    try:
        task = await tasker.enqueue(
            name=f"知识库文档处理({db_id})",
            task_type="knowledge_ingest",
            payload={
                "db_id": db_id,
                "items": items,
                "params": params,
                "content_type": content_type,
            },
            coroutine=run_ingest,
        )
        return {
            "message": "任务已提交，请在任务中心查看进度",
            "status": "queued",
            "task_id": task.id,
        }
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to enqueue {content_type}s: {e}, {traceback.format_exc()}")
        return {"message": f"Failed to enqueue task: {e}", "status": "failed"}


@knowledge.get("/databases/{db_id}/documents/{doc_id}")
async def get_document_info(db_id: str, doc_id: str, current_user: User = Depends(get_admin_user)):
    """获取文档详细信息（包含基本信息和内容信息）"""
    logger.debug(f"GET document {doc_id} info in {db_id}")

    try:
        info = await knowledge_base.get_file_info(db_id, doc_id)
        return info
    except Exception as e:
        logger.error(f"Failed to get file info, {e}, {db_id=}, {doc_id=}, {traceback.format_exc()}")
        return {"message": "Failed to get file info", "status": "failed"}


@knowledge.get("/databases/{db_id}/documents/{doc_id}/basic")
async def get_document_basic_info(db_id: str, doc_id: str, current_user: User = Depends(get_admin_user)):
    """获取文档基本信息（仅元数据）"""
    logger.debug(f"GET document {doc_id} basic info in {db_id}")

    try:
        info = await knowledge_base.get_file_basic_info(db_id, doc_id)
        return info
    except Exception as e:
        logger.error(f"Failed to get file basic info, {e}, {db_id=}, {doc_id=}, {traceback.format_exc()}")
        return {"message": "Failed to get file basic info", "status": "failed"}


@knowledge.get("/databases/{db_id}/documents/{doc_id}/content")
async def get_document_content(db_id: str, doc_id: str, current_user: User = Depends(get_admin_user)):
    """获取文档内容信息（chunks和lines）"""
    logger.debug(f"GET document {doc_id} content in {db_id}")

    try:
        info = await knowledge_base.get_file_content(db_id, doc_id)
        return info
    except Exception as e:
        logger.error(f"Failed to get file content, {e}, {db_id=}, {doc_id=}, {traceback.format_exc()}")
        return {"message": "Failed to get file content", "status": "failed"}


@knowledge.delete("/databases/{db_id}/documents/{doc_id}")
async def delete_document(db_id: str, doc_id: str, current_user: User = Depends(get_admin_user)):
    """删除文档"""
    logger.debug(f"DELETE document {doc_id} info in {db_id}")
    try:
        await knowledge_base.delete_file(db_id, doc_id)
        return {"message": "删除成功"}
    except Exception as e:
        logger.error(f"删除文档失败 {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"删除文档失败: {e}")


@knowledge.post("/databases/{db_id}/documents/{doc_id}/rebuild-graph")
async def rebuild_document_graph(
    db_id: str,
    doc_id: str,
    params: dict = Body(default={}),
    current_user: User = Depends(get_admin_user),
):
    """手动重建单个文档的知识图谱（不需重新上传文件）。"""
    logger.info(f"Rebuild graph for document {doc_id} in {db_id}, params={params}")

    async def run_rebuild(context: TaskContext):
        await context.set_message("准备重建图谱")
        await context.set_progress(10.0, "检查文件状态")
        await context.raise_if_cancelled()

        result = await knowledge_base.rebuild_file_graph(db_id, doc_id, params=params)
        await context.set_result(result)
        await context.set_progress(100.0, "重建完成")
        return result

    try:
        task = await tasker.enqueue(
            name=f"重建文档图谱({db_id}/{doc_id})",
            task_type="knowledge_rebuild_graph",
            payload={
                "db_id": db_id,
                "doc_id": doc_id,
                "params": params,
            },
            coroutine=run_rebuild,
        )
        return {
            "message": "图谱重建任务已提交，请在任务中心查看进度",
            "status": "queued",
            "task_id": task.id,
        }
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to enqueue rebuild graph task: {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"提交重建任务失败: {e}")

@knowledge.get("/databases/{db_id}/documents/{doc_id}/download")
async def download_document(db_id: str, doc_id: str, request: Request, current_user: User = Depends(get_admin_user)):
    """下载原始文件"""
    logger.debug(f"Download document {doc_id} from {db_id}")
    try:
        file_info = await knowledge_base.get_file_basic_info(db_id, doc_id)
        if not file_info:
            raise HTTPException(status_code=404, detail="File not found")

        file_path = file_info.get("meta", {}).get("path")
        if not file_path:
            raise HTTPException(status_code=404, detail="File path not found in metadata")

        # 安全检查：验证文件路径
        from src.knowledge.utils.kb_utils import validate_file_path

        try:
            normalized_path = validate_file_path(file_path, db_id)
        except ValueError as e:
            raise HTTPException(status_code=403, detail=str(e))

        if not os.path.exists(normalized_path):
            raise HTTPException(status_code=404, detail=f"File not found on disk: {file_info=}")

        # 获取文件扩展名和MIME类型，解码URL编码的文件名
        filename = file_info.get("meta", {}).get("filename", "file")
        logger.debug(f"Original filename from database: {filename}")

        # 解码URL编码的文件名（如果有的话）
        try:
            decoded_filename = unquote(filename, encoding="utf-8")
            logger.debug(f"Decoded filename: {decoded_filename}")
        except Exception as e:
            logger.debug(f"Failed to decode filename {filename}: {e}")
            decoded_filename = filename  # 如果解码失败，使用原文件名

        _, ext = os.path.splitext(decoded_filename)

        media_types = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".json": "application/json",
            ".csv": "text/csv",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".ppt": "application/vnd.ms-powerpoint",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".svg": "image/svg+xml",
            ".zip": "application/zip",
            ".rar": "application/x-rar-compressed",
            ".7z": "application/x-7z-compressed",
            ".tar": "application/x-tar",
            ".gz": "application/gzip",
            ".html": "text/html",
            ".htm": "text/html",
            ".xml": "text/xml",
            ".css": "text/css",
            ".js": "application/javascript",
            ".py": "text/x-python",
            ".java": "text/x-java-source",
            ".cpp": "text/x-c++src",
            ".c": "text/x-csrc",
            ".h": "text/x-chdr",
            ".hpp": "text/x-c++hdr",
        }
        media_type = media_types.get(ext.lower(), "application/octet-stream")

        # 创建自定义FileResponse，避免文件名编码问题
        response = StarletteFileResponse(path=normalized_path, media_type=media_type)

        # 正确处理中文文件名的HTTP头部设置
        # HTTP头部只能包含ASCII字符，所以需要对中文文件名进行编码
        try:
            # 尝试使用ASCII编码（适用于英文文件名）
            decoded_filename.encode("ascii")
            # 如果成功，直接使用简单格式
            response.headers["Content-Disposition"] = f'attachment; filename="{decoded_filename}"'
        except UnicodeEncodeError:
            # 如果包含非ASCII字符（如中文），使用RFC 2231格式
            encoded_filename = quote(decoded_filename.encode("utf-8"))
            response.headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{encoded_filename}"

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载文件失败: {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"下载失败: {e}")


# =============================================================================
# === 查询分组 ===
# =============================================================================


@knowledge.post("/databases/{db_id}/query")
async def query_knowledge_base(
    db_id: str, query: str = Body(...), meta: dict = Body(...), current_user: User = Depends(get_admin_user)
):
    """查询知识库"""
    logger.debug(f"Query knowledge base {db_id}: {query}")
    try:
        result = await knowledge_base.aquery(query, db_id=db_id, **meta)
        return {"result": result, "status": "success"}
    except Exception as e:
        logger.error(f"知识库查询失败 {e}, {traceback.format_exc()}")
        return {"message": f"知识库查询失败: {e}", "status": "failed"}


@knowledge.post("/databases/{db_id}/query-test")
async def query_test(
    db_id: str, query: str = Body(...), meta: dict = Body(...), current_user: User = Depends(get_admin_user)
):
    """测试查询知识库"""
    logger.debug(f"Query test in {db_id}: {query}")
    try:
        result = await knowledge_base.aquery(query, db_id=db_id, **meta)

        snippets = []
        citations = []
        if isinstance(result, list):
            for idx, item in enumerate(result[:6], 1):
                text = str(item).strip()
                if not text:
                    continue
                snippets.append(text[:260])
                source_hint = f"{db_id}#chunk{idx}"
                if isinstance(item, dict):
                    source_hint = (
                            item.get("source")
                            or item.get("file_name")
                            or item.get("filename")
                            or item.get("doc_id")
                            or source_hint
                    )
                citations.append(str(source_hint))
        elif isinstance(result, dict):
            text = str(result.get("content") or result.get("answer") or result).strip()
            if text:
                snippets.append(text[:260])
            source_hint = result.get("source") or result.get("file_name") or f"{db_id}#result"
            citations.append(str(source_hint))
        elif result:
            snippets.append(str(result)[:260])
            citations.append(f"{db_id}#result")

        answer = ""
        if snippets:
            try:
                provider, model_name = config.default_model.split("/", 1)
                model = select_model(provider, model_name)
                prompt = (
                        "请根据检索证据生成简洁自然语言回答，不要输出JSON。"
                        f"\n问题：{query}\n证据：\n" + "\n".join([f"- {s}" for s in snippets[:4]])
                )
                resp = model.call(
                    [
                        {"role": "system", "content": "你是知识库问答助手，回答简洁并保持可核验。"},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False,
                )
                answer = getattr(resp, "content", str(resp)).strip()
            except Exception as e:
                logger.warning(f"query-test LLM总结失败，回退为证据拼接: {e}")
                answer = "；".join(snippets[:3])

        return {
            "status": "success",
            "answer": answer or "未检索到可用内容。",
            "citations": list(dict.fromkeys([c for c in citations if c]))[:6],
            "raw_result": result,
        }
    except Exception as e:
        logger.error(f"测试查询失败 {e}, {traceback.format_exc()}")
        return {"message": f"测试查询失败: {e}", "status": "failed"}


@knowledge.get("/databases/{db_id}/query-params")
async def get_knowledge_base_query_params(db_id: str, current_user: User = Depends(get_admin_user)):
    """获取知识库类型特定的查询参数"""
    try:
        # 获取数据库信息
        db_info = knowledge_base.get_database_info(db_id)
        if not db_info:
            raise HTTPException(status_code=404, detail="Database not found")

        kb_type = db_info.get("kb_type", "lightrag")

        # 根据知识库类型返回不同的查询参数
        if kb_type == "lightrag":
            params = {
                "type": "lightrag",
                "options": [
                    {
                        "key": "mode",
                        "label": "检索模式",
                        "type": "select",
                        "default": "mix",
                        "options": [
                            {"value": "local", "label": "Local", "description": "上下文相关信息"},
                            {"value": "global", "label": "Global", "description": "全局知识"},
                            {"value": "hybrid", "label": "Hybrid", "description": "本地和全局混合"},
                            {"value": "naive", "label": "Naive", "description": "基本搜索"},
                            {"value": "mix", "label": "Mix", "description": "知识图谱和向量检索混合"},
                        ],
                    },
                    {
                        "key": "only_need_context",
                        "label": "只使用上下文",
                        "type": "boolean",
                        "default": True,
                        "description": "只返回上下文，不生成回答",
                    },
                    {
                        "key": "only_need_prompt",
                        "label": "只使用提示",
                        "type": "boolean",
                        "default": False,
                        "description": "只返回提示，不进行检索",
                    },
                    {
                        "key": "top_k",
                        "label": "TopK",
                        "type": "number",
                        "default": 10,
                        "min": 1,
                        "max": 100,
                        "description": "返回的最大结果数量",
                    },
                ],
            }
        elif kb_type == "chroma":
            params = {
                "type": "chroma",
                "options": [
                    {
                        "key": "top_k",
                        "label": "TopK",
                        "type": "number",
                        "default": 10,
                        "min": 1,
                        "max": 100,
                        "description": "返回的最大结果数量",
                    },
                    {
                        "key": "similarity_threshold",
                        "label": "相似度阈值",
                        "type": "number",
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "description": "过滤相似度低于此值的结果",
                    },
                    {
                        "key": "include_distances",
                        "label": "显示相似度",
                        "type": "boolean",
                        "default": True,
                        "description": "在结果中显示相似度分数",
                    },
                ],
            }
        elif kb_type == "milvus":
            params = {
                "type": "milvus",
                "options": [
                    {
                        "key": "top_k",
                        "label": "TopK",
                        "type": "number",
                        "default": 10,
                        "min": 1,
                        "max": 100,
                        "description": "返回的最大结果数量",
                    },
                    {
                        "key": "similarity_threshold",
                        "label": "相似度阈值",
                        "type": "number",
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "description": "过滤相似度低于此值的结果",
                    },
                    {
                        "key": "include_distances",
                        "label": "显示相似度",
                        "type": "boolean",
                        "default": True,
                        "description": "在结果中显示相似度分数",
                    },
                    {
                        "key": "metric_type",
                        "label": "距离度量类型",
                        "type": "select",
                        "default": "COSINE",
                        "options": [
                            {"value": "COSINE", "label": "余弦相似度", "description": "适合文本语义相似度"},
                            {"value": "L2", "label": "欧几里得距离", "description": "适合数值型数据"},
                            {"value": "IP", "label": "内积", "description": "适合标准化向量"},
                        ],
                        "description": "向量相似度计算方法",
                    },
                ],
            }
        else:
            # 未知类型，返回基本参数
            params = {
                "type": "unknown",
                "options": [
                    {
                        "key": "top_k",
                        "label": "TopK",
                        "type": "number",
                        "default": 10,
                        "min": 1,
                        "max": 100,
                        "description": "返回的最大结果数量",
                    }
                ],
            }

        return {"params": params, "message": "success"}

    except Exception as e:
        logger.error(f"获取知识库查询参数失败 {e}, {traceback.format_exc()}")
        return {"message": f"获取知识库查询参数失败 {e}", "params": {}}


# =============================================================================
# === 文件管理分组 ===
# =============================================================================


@knowledge.post("/files/upload")
async def upload_file(
    file: UploadFile = File(...),
    db_id: str | None = Query(None),
    allow_jsonl: bool = Query(False),
    auto_ingest: bool = Query(True, description="上传后是否自动触发知识入库任务（仅当 db_id 存在时生效）"),
    current_user: User = Depends(get_admin_user),
):
    """上传文件"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    logger.debug(f"Received upload file with filename: {file.filename}")

    ext = os.path.splitext(file.filename)[1].lower()

    if ext == ".jsonl":
        if allow_jsonl is not True or db_id is not None:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    elif not is_supported_file_extension(file.filename):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # 根据db_id获取上传路径，如果db_id为None则使用默认路径
    if db_id:
        upload_dir = knowledge_base.get_db_upload_path(db_id)
    else:
        upload_dir = os.path.join(config.save_dir, "database", "uploads")

    basename, ext = os.path.splitext(file.filename)
    filename = f"{basename}_{hashstr(basename, 4, with_salt=True)}{ext}".lower()
    file_path = os.path.join(upload_dir, filename)
    os.makedirs(upload_dir, exist_ok=True)

    file_bytes = await file.read()

    content_hash = calculate_content_hash(file_bytes)
    if knowledge_base.file_existed_in_db(db_id, content_hash):
        raise HTTPException(
            status_code=409,
            detail="数据库中已经存在了相同文件，File with the same content already exists in this database",
        )

    with open(file_path, "wb") as buffer:
        buffer.write(file_bytes)

    response = {
        "message": "File successfully uploaded",
        "file_path": file_path,
        "db_id": db_id,
        "content_hash": content_hash,
        "ingest_status": "not_requested",
    }

    should_auto_ingest = bool(db_id) and auto_ingest
    if not should_auto_ingest:
        return response

    async def run_ingest_after_upload(context: TaskContext):
        await context.set_message("任务初始化")
        await context.set_progress(10.0, "上传完成，准备自动入库")
        result = await knowledge_base.add_content(
            db_id,
            [file_path],
            params={"content_type": "file"},
        )
        done_count = len([item for item in result if item.get("status") == "done"])
        failed_count = len(result) - done_count
        await context.set_result(
            {
                "db_id": db_id,
                "file_path": file_path,
                "submitted": len(result),
                "done": done_count,
                "failed": failed_count,
                "items": result,
            }
        )
        await context.set_progress(100.0, "自动入库完成")
        return result

    try:
        task = await tasker.enqueue(
            name=f"上传后自动入库({db_id})",
            task_type="knowledge_ingest",
            payload={
                "db_id": db_id,
                "items": [file_path],
                "params": {"content_type": "file"},
                "source": "upload_auto_ingest",
            },
            coroutine=run_ingest_after_upload,
        )
        response["ingest_status"] = "queued"
        response["ingest_task_id"] = task.id
        response["message"] = "File successfully uploaded, ingest task queued"
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to enqueue auto-ingest task for {db_id=}, {file_path=}: {exc}, {traceback.format_exc()}")
        response["ingest_status"] = "failed"
        response["ingest_error"] = str(exc)
        response["message"] = "File uploaded, but auto ingest failed to enqueue"

    return response

@knowledge.get("/files/supported-types")
async def get_supported_file_types(current_user: User = Depends(get_admin_user)):
    """获取当前支持的文件类型"""
    return {"message": "success", "file_types": sorted(SUPPORTED_FILE_EXTENSIONS)}


@knowledge.post("/files/markdown")
async def mark_it_down(file: UploadFile = File(...), current_user: User = Depends(get_admin_user)):
    """调用 src.knowledge.indexing 下面的 process_file_to_markdown 解析为 markdown，参数是文件，需要管理员权限"""
    try:
        content = await file.read()
        markdown_content = await process_file_to_markdown(content)
        return {"markdown_content": markdown_content, "message": "success"}
    except Exception as e:
        logger.error(f"文件解析失败 {e}, {traceback.format_exc()}")
        return {"message": f"文件解析失败 {e}", "markdown_content": ""}


# =============================================================================
# === 知识库类型分组 ===
# =============================================================================


@knowledge.get("/types")
async def get_knowledge_base_types(current_user: User = Depends(get_admin_user)):
    """获取支持的知识库类型"""
    try:
        kb_types = knowledge_base.get_supported_kb_types()
        return {"kb_types": kb_types, "message": "success"}
    except Exception as e:
        logger.error(f"获取知识库类型失败 {e}, {traceback.format_exc()}")
        return {"message": f"获取知识库类型失败 {e}", "kb_types": {}}


@knowledge.get("/stats")
async def get_knowledge_base_statistics(current_user: User = Depends(get_admin_user)):
    """获取知识库统计信息"""
    try:
        stats = knowledge_base.get_statistics()
        return {"stats": stats, "message": "success"}
    except Exception as e:
        logger.error(f"获取知识库统计失败 {e}, {traceback.format_exc()}")
        return {"message": f"获取知识库统计失败 {e}", "stats": {}}


# =============================================================================
# === Embedding模型状态检查分组 ===
# =============================================================================


@knowledge.get("/embedding-models/{model_id}/status")
async def get_embedding_model_status(model_id: str, current_user: User = Depends(get_admin_user)):
    """获取指定embedding模型的状态"""
    logger.debug(f"Checking embedding model status: {model_id}")
    try:
        status = await test_embedding_model_status(model_id)
        return {"status": status, "message": "success"}
    except Exception as e:
        logger.error(f"获取embedding模型状态失败 {model_id}: {e}, {traceback.format_exc()}")
        return {
            "message": f"获取embedding模型状态失败: {e}",
            "status": {"model_id": model_id, "status": "error", "message": str(e)},
        }


@knowledge.get("/embedding-models/status")
async def get_all_embedding_models_status(current_user: User = Depends(get_admin_user)):
    """获取所有embedding模型的状态"""
    logger.debug("Checking all embedding models status")
    try:
        status = await test_all_embedding_models_status()
        return {"status": status, "message": "success"}
    except Exception as e:
        logger.error(f"获取所有embedding模型状态失败: {e}, {traceback.format_exc()}")
        return {"message": f"获取所有embedding模型状态失败: {e}", "status": {"models": {}, "total": 0, "available": 0}}

#!/usr/bin/env bash
set -euo pipefail

# =========================
# 0) 基础配置（按需修改）
# =========================
BASE_URL="${BASE_URL:-http://127.0.0.1:5050}"
USER_ID="${USER_ID:-admin}"
PASSWORD="${PASSWORD:-admin123}"
EMBED_MODEL_NAME="${EMBED_MODEL_NAME:-siliconflow/BAAI/bge-m3}"

AUTO_DB_NAME="${AUTO_DB_NAME:-cs408_auto}"
AUTO_DB_DESC="${AUTO_DB_DESC:-408 自动解析知识图谱}"
AUTO_FILE="${AUTO_FILE:-./examples/cs408/cs408_auto_sample.json}"

TRIPLE_FILE="${TRIPLE_FILE:-./examples/cs408/cs408_seed_template.jsonl}"

# =========================
# 1) 登录（首次可先调用 initialize）
# =========================
# 首次初始化（系统首次启动时可用，已初始化会失败，可忽略）
# curl -sS -X POST "$BASE_URL/api/auth/initialize" \
#   -H "Content-Type: application/json" \
#   -d '{"user_id":"'"$USER_ID"'","password":"'"$PASSWORD"'"}'

TOKEN_JSON=$(curl -sS -X POST "$BASE_URL/api/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "username=$USER_ID" \
  --data-urlencode "password=$PASSWORD")

ACCESS_TOKEN=$(echo "$TOKEN_JSON" | jq -r '.access_token')
if [[ -z "$ACCESS_TOKEN" || "$ACCESS_TOKEN" == "null" ]]; then
  echo "[ERROR] 登录失败：$TOKEN_JSON" >&2
  exit 1
fi

echo "[OK] 登录成功"

# =========================
# 2) 方案A：自动解析（创建库→上传→入库→查询）
# =========================
CREATE_DB_JSON=$(curl -sS -X POST "$BASE_URL/api/knowledge/databases" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "database_name": "'"$AUTO_DB_NAME"'",
    "description": "'"$AUTO_DB_DESC"'",
    "embed_model_name": "'"$EMBED_MODEL_NAME"'",
    "kb_type": "lightrag",
    "additional_params": {"addon_params": {"domain": "computer"}}
  }')

DB_ID=$(echo "$CREATE_DB_JSON" | jq -r '.db_id')
if [[ -z "$DB_ID" || "$DB_ID" == "null" ]]; then
  echo "[ERROR] 创建知识库失败：$CREATE_DB_JSON" >&2
  exit 1
fi

echo "[OK] 已创建知识库 DB_ID=$DB_ID"

UPLOAD_AUTO_JSON=$(curl -sS -X POST "$BASE_URL/api/knowledge/files/upload?db_id=$DB_ID" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -F "file=@$AUTO_FILE")

AUTO_FILE_PATH=$(echo "$UPLOAD_AUTO_JSON" | jq -r '.file_path')
if [[ -z "$AUTO_FILE_PATH" || "$AUTO_FILE_PATH" == "null" ]]; then
  echo "[ERROR] 自动解析文件上传失败：$UPLOAD_AUTO_JSON" >&2
  exit 1
fi

echo "[OK] 自动解析文件已上传：$AUTO_FILE_PATH"

curl -sS -X POST "$BASE_URL/api/knowledge/databases/$DB_ID/documents" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "items": ["'"$AUTO_FILE_PATH"'"],
    "params": {
      "content_type": "file",
      "chunk_size": 700,
      "chunk_overlap": 100
    }
  }' | jq

echo "[OK] 已提交自动解析任务（请在任务中心观察完成状态）"

# 直接知识库查询（可在任务完成后执行）
curl -sS -X POST "$BASE_URL/api/knowledge/databases/$DB_ID/query" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "请说明 TCP 三次握手与滑动窗口的关系",
    "meta": {"mode": "mix", "top_k": 10}
  }' | jq

# =========================
# 3) 方案B：手工三元组（上传jsonl→入库→图查询）
# =========================
UPLOAD_TRIPLE_JSON=$(curl -sS -X POST "$BASE_URL/api/knowledge/files/upload?allow_jsonl=true" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -F "file=@$TRIPLE_FILE")

TRIPLE_FILE_PATH=$(echo "$UPLOAD_TRIPLE_JSON" | jq -r '.file_path')
if [[ -z "$TRIPLE_FILE_PATH" || "$TRIPLE_FILE_PATH" == "null" ]]; then
  echo "[ERROR] JSONL 上传失败：$UPLOAD_TRIPLE_JSON" >&2
  exit 1
fi

echo "[OK] JSONL 已上传：$TRIPLE_FILE_PATH"

curl -sS -X POST "$BASE_URL/api/graph/neo4j/add-entities" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"file_path":"'"$TRIPLE_FILE_PATH"'","kgdb_name":"neo4j"}' | jq

echo "[OK] 已导入 Neo4j 三元组"

# 按实体查询图谱
curl -sS "$BASE_URL/api/graph/neo4j/node?entity_name=TCP" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | jq

echo "[DONE] 408 两套流程示例执行完毕"

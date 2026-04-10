<!-- <h1 align="center">基于多智能体和工作流的大模型的调研报告生成系统</h1> -->
<h1 align="center">CompKG-QA: 计算机知识图谱智能知识问答与多模态检索系统</h1>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
# CompKG-QA

计算机知识图谱智能问答系统，基于 LangGraph 和 LangChain 构建的智能助手，专注于计算机学科知识领域的知识管理和智能问答。

## 项目简介

CompKG-QA 是一个功能完整的计算机知识图谱问答系统，集成了向量检索和知识图谱技术，支持多种文件格式的知识库构建，并提供基于智能体的对话交互能力。

## 📸 项目预览

<summary>点击放大截图查看</summary>

| 截图1 | 截图2 | 截图3 |
|[img_1.png]|-------|-------|


### 核心特性

- **多模态知识库管理**
  - 支持 ChromaDB 向量数据库（语义检索）
  - 支持 LightRAG 知识图谱（复杂推理）
  - 支持多种文件格式：PDF、图片、文本、CSV、JSON 等
  - 图片嵌入支持（基于 CN-CLIP 模型）

- **智能体系统**
  - ChatbotAgent：基础对话机器人
  - 支持动态工具调用
  - 支持 MCP（Model Context Protocol）工具扩展
  - 基于 LangGraph 构建的状态机工作流

- **知识图谱**
  - 基于 Neo4j 的图存储
  - 自定义实体和关系类型（博物馆文物领域）
  - 多种检索模式：mix、local、global、hybrid、naive

- **安全与认证**
  - JWT 身份认证
  - 登录频率限制
  - API 权限管理

## 技术栈

### 后端

- **框架**：FastAPI
- **AI/ML**：LangGraph、LangChain、OpenAI
- **向量数据库**：ChromaDB、FAISS
- **图数据库**：Neo4j
- **知识图谱**：LightRAG
- **图像处理**：CN-CLIP、PyMuPDF、RapidOCR
- **数据库**：SQLite
- **包管理**：Poetry

### 前端

- **框架**：Vue 3
- **构建工具**：Vite
- **UI 组件**：Ant Design Vue
- **状态管理**：Pinia
- **路由**：Vue Router
- **数据可视化**：ECharts、G6、Sigma
- **Markdown 编辑器**：md-editor-v3
- **包管理**：pnpm

## 项目结构
CompKG-QA 是一个功能完整的计算机知识图谱问答系统，集成了
```
compKG-QA/
├── server/                 # FastAPI 后端服务
│   ├── routers/           # API 路由
│   ├── services/          # 业务服务
│   └── utils/             # 工具函数
├── src/                   # 核心业务逻辑
│   ├── agents/           # 智能体系统
│   ├── knowledge/        # 知识库管理
│   ├── models/           # 模型封装
│   ├── storage/          # 存储层
│   └── utils/            # 工具函数
├── web/                   # Vue 3 前端
│   ├── src/
│   │   ├── apis/        # API 调用
│   │   └── App.vue      # 应用入口
│   └── public/          # 静态资源
├── saves/                # 数据存储目录
│   ├── agents/          # 智能体数据
│   ├── knowledge_base_data/  # 知识库数据
│   └── database/        # 数据库文件
└── test/                 # 测试文件
```

## 快速开始

### 环境要求

- Python >= 3.10, < 3.13
- Node.js >= 18
- pnpm >= 10
- Poetry

### 1. 克隆项目

```bash
git clone <repository-url>
cd CompKG-QA
```

### 2. 配置环境变量

复制 `.env.template` 为 `.env` 并配置必要的环境变量：

```bash
cp .env.template .env
```

主要配置项：

```env
# 模型提供商 API 密钥
SILICONFLOW_API_KEY=your_api_key
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://api.openai.com/v1

# Neo4j 配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

```

### 3. 安装后端依赖

```bash
poetry install
```

### 4. 安装前端依赖

```bash
cd web
pnpm install
cd ..
```

### 5. 启动服务

**启动后端服务：**


```bash
.\neo4j console


```bash

uvicorn server.main:app --reload --port 5050
或者
poetry run python -m server.main
或
python -m server.main
```

后端服务将在 `http://0.0.0.0:5050` 启动。

**启动前端服务：**

```bash
cd web
npm install
npm run dev
```

前端服务将在 `http://localhost:5173` 启动。

## 核心功能说明

### 知识库管理系统

支持创建、删除、更新知识库，提供两种类型的知识库实现：

1. **ChromaKB（向量知识库）**
   - 使用 OpenAIEmbeddingFunction 进行文本嵌入
   - 使用 CN-CLIP 模型生成图片嵌入向量
   - 支持普通分块和 QA 分块两种模式
   - 适用于语义检索场景

2. **LightRagKB（知识图谱）**
   - 基于 LightRAG 框架
   - 支持实体和关系抽取
   - 多存储后端：FaissVectorDBStorage、JsonKVStorage、Neo4JStorage
   - 支持多种检索模式：mix、local、global、hybrid、naive
   - 适用于复杂推理场景
   - 上传文档后自动执行知识图谱抽取（`/knowledge/databases/{db_id}/documents`）

### 文档上传自动转知识图谱说明

- 当知识库类型为 `lightrag` 时，文档上传后会自动执行实体/关系抽取并写入 Neo4j 图谱。
- 当知识库类型为 `chroma` 时，默认只进行向量化；如果希望“上传即自动入图谱”，可在上传参数中增加：
  - `auto_graph_ingest: true`
  - `graph_db_id: <目标 lightrag 知识库 ID>`
- 系统会在每个文档处理结果里返回 `graph_sync` 字段，明确图谱同步状态（`done/failed/skipped`）。
### OCR 模型配置（图像问答推荐）

- **是否必须下载 RapidOCR 模型？**
  - 不是强制。系统在 OCR 不可用时会回退到 VL/基础图像分析流程，服务可继续运行。
  - 但如果你希望图像中的文字（截图、公式、流程图中的标签）被稳定识别，建议配置 RapidOCR 模型。
- **模型目录约定：**
  - 设置 `MODEL_DIR`（Docker 下可用 `MODEL_DIR_IN_DOCKER`）后，程序会在以下路径查找：
    - `SWHL/RapidOCR/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx`
    - `SWHL/RapidOCR/PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx`
- **Windows 示例：**
  ```env
  MODEL_DIR=G:\\AI\\models
  ```
  则模型文件应位于：
  - `G:\AI\models\SWHL\RapidOCR\PP-OCRv4\ch_PP-OCRv4_det_infer.onnx`
  - `G:\AI\models\SWHL\RapidOCR\PP-OCRv4\ch_PP-OCRv4_rec_infer.onnx`
- 若未配置 `MODEL_DIR`，系统会尝试 `./models` 与 `./saves/models` 作为默认目录。
- **若 RapidOCR 无法下载/部署，可切换 PaddleX 作为图像 OCR 回退：**
  - 启动 PaddleX 服务（默认 `http://localhost:8080`）。
  - 配置环境变量：
    ```env
    OCR_IMAGE_FALLBACK_PROVIDER=paddlex
    PADDLEX_URI=http://localhost:8080
    ```
  - 含义：当 RapidOCR 本地模型缺失时，`process_image` 自动回退到 PaddleX OCR，避免图像问答丢失 OCR 文本信号。


### 智能体系统

基于 LangGraph 构建的智能体框架，支持：

- **ChatbotAgent**：基础对话机器人，可配置工具
- 动态工具调用
- MCP（Model Context Protocol）工具扩展
- 状态机工作流管理
- 对话历史持久化

### 知识图谱

针对计算机学科领域预定义的实体和关系类型：

**实体类型：**
Technology（技术概念）
- Algorithm（算法）
- DataStructure（数据结构）
- ProgrammingLanguage（编程语言）
- Framework / Library（框架/库）
- Protocol（协议）
- Database（数据库）
- Tool / Platform（工具/平台）
- Organization / Person（组织/人物）

**关系类型：**
is_a / part_of（层级与组成）
- depends_on / uses / implements（依赖与实现）
- compatible_with / alternative_to（兼容与替代）
- runs_on / stores_in / communicates_via（运行与交互）
- proposed_by / developed_by / introduced_in（来源与演化）
- optimized_for / measured_by（优化目标与评估指标）

### 非结构化文档自动建图建议（可借鉴 GraphRAG-Example）

建议采用以下流水线（当前项目已落地规则版骨架，可逐步替换为模型版 NER/RE）：

1. 文档预处理（清洗、分段、去噪）
2. 实体识别（NER）
3. 关系抽取（RE）
4. 生成三元组（S-P-O）
5. 图数据库写入与可视化

对应代码入口：`src/knowledge/pipeline/unstructured_to_kg.py`（`UnstructuredToKGPipeline`）。
已提供 NER/RE 插件接口（默认 `rule` 实现），可通过环境变量切换：

- `LIGHTRAG_KG_NER_PLUGIN=rule`
- `LIGHTRAG_KG_RE_PLUGIN=rule`

后续可直接注册模型版插件（如 GLiNER/LLM-RE）而不改主流程。

示例（已内置占位插件）：

- RE 插件名：`llm_stub`
- 配置：
  - `LIGHTRAG_KG_NER_PLUGIN=llm`
  - `LIGHTRAG_KG_RE_PLUGIN=llm`
  - `KG_NER_PLUGIN_LLM_ENABLED=true`
  - `KG_NER_PLUGIN_MODEL_SPEC=<provider/model>`
  - `LIGHTRAG_KG_RE_PLUGIN=llm_stub`
  - `KG_RE_PLUGIN_LLM_ENABLED=true`（占位开关）
  - `KG_RE_PLUGIN_MODEL=stub-model`（占位模型名）
  - `KG_RE_PLUGIN_MODEL_SPEC=<provider/model>`

实现位置：`src/knowledge/pipeline/plugins/llm_ner.py`、`src/knowledge/pipeline/plugins/llm_re.py`。

**实体链接（Entity Linking）**：新增 `OntologyEntityLinker`，支持实体 mention 到专业术语 canonical 映射，并返回 `ontology_id/source/confidence`。
  - 代码：`src/knowledge/entity_linking.py`
- **图谱补全 + 多源融合（KGC + Fusion）**：新增 `KGCompleterAndFusion`，支持对称/逆关系/有限传递补全与来源追踪。
  - 代码：`src/knowledge/kg_enhancement.py`
- **可视化增强**：新增 `visualize_reasoning_paths`，输出 `nodes/edges/reasoning_paths` 用于前端路径推理可视化。

- **智能体扩展（ChatbotAgent）**：
  - 多任务分层规划（任务分解）
  - 自动工具路由（按 query 关键词筛选工具）
  - 短期/长期记忆分离（上下文压缩）
  - 自我反思（回答后质量自检）
  - 多模态图像理解工具（OCR + 视觉描述证据包）

可直接调用新增工具：
- `link_entities_to_ontology`
- `kg_complete_and_fuse`
- `graph_reasoning_visualization`
- `multimodal_image_understand`


并且支持数据库级配置（`PUT /knowledge/databases/{db_id}/ontology`）：

- `kg_ner_plugin` / `kg_re_plugin`（`rule` 或 `llm`）
- `kg_ner_model_spec` / `kg_re_model_spec`
- `kg_ner_llm_enabled` / `kg_re_llm_enabled`
## 开发指南

### 添加新的智能体

1. 在 `src/agents/` 目录下创建新的智能体类
2. 继承 `BaseAgent` 基类
3. 实现 `get_graph()` 方法定义工作流
4. 在 `AgentManager` 中注册智能体

### 添加新的知识库类型

1. 在 `src/knowledge/implementations/` 目录下创建新的知识库实现
2. 继承 `KnowledgeBase` 基类
3. 实现核心方法：`create_database()`, `add_content()`, `aquery()` 等
4. 在 `KnowledgeBaseFactory` 中注册新类型

## 常见问题

### 后端启动失败

- 检查 Python 版本是否符合要求（>= 3.10, < 3.13）
- 确认所有依赖已正确安装：`poetry install`
- 检查环境变量配置是否正确

### 前端启动失败

- 确认 Node.js 版本 >= 18
- 检查 pnpm 版本 >= 10
- 删除 `node_modules` 和 `pnpm-lock.yaml` 后重新安装

### 知识库查询无结果

- 确认知识库已正确创建并添加内容
- 检查 LLM API 密钥是否配置正确
- 查看日志文件排查具体错误

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue

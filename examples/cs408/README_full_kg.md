# CS408 Full Knowledge Graph Dataset（专业级论文标准）

该数据集用于 CompKG-QA 的 408 课程知识图谱构建，遵循“实体分类严格 + 关系语义清晰 + 可评估可迭代”的设计原则。

## 1) 专业级实体分类（严格）
- `Course`：课程（数据结构、操作系统、计算机网络、计算机组成原理）
- `Chapter`：章节（线性表、传输层、流水线等）
- `Concept`：核心概念
- `Method`：方法/算法/操作
- `Property`：性质/指标
- `Formula`：公式
- `Example`：例题或应用场景

## 2) 关系类型（严格语义）
- `BELONGS_TO`
- `HAS_SUB`
- `HAS_METHOD`
- `HAS_PROPERTY`
- `HAS_COMPLEXITY`
- `PREREQUISITE`
- `CAUSE`
- `USED_IN`
- `EQUIVALENT`

> 关系和实体标准定义见：`cs408_relation_ontology.yaml`、`cs408_paper_schema.yaml`。

## 3) 文件说明
- `cs408_full_kg_nodes.json`：节点数据
- `cs408_full_kg_edges.csv`：边关系数据
- `cs408_full_kg_triples.json`：三元组 JSON
- `cs408_full_kg_triples.jsonl`：三元组 JSONL（导入推荐）
- `cs408_full_kg_meta.json`：规模与统计元信息
- `cs408_expert_seed.jsonl`：专家种子数据（含 `h_type` / `t_type`）

## 4) 字段定义（关键）
### Triples JSONL
```json
{"h":"快速排序","h_type":"Method","r":"HAS_COMPLEXITY","t":"O(nlogn)","t_type":"Property","subject":"数据结构"}
```

### Edges CSV
- `source`：源节点 ID
- `target`：目标节点 ID
- `relation`：关系类型（仅允许标准词表）
- `subject`：学科标签（跨学科边为 `A|B`）
- `weight`：边权重

## 5) 导入建议
优先导入：`cs408_full_kg_triples.jsonl`。

可使用接口：
- `POST /api/graph/neo4j/add-entities`

## 6) 项目升级能力（配套）
- 学习路径推荐优化：基于 `PREREQUISITE` 自动生成学习顺序
- 多轮对话记忆：图谱问答接口支持 `session_id` 记忆上下文
- 实验评估（准确率）：`examples/cs408/eval/score_eval.py`
- Agent 推理链：输出 `reasoning_path` + `derivation_chain`
- 可选：错题系统、自动生成题库（可基于 `USED_IN` / `HAS_METHOD` 扩展）

## 7) 数据生成与校验
- 生成：`python examples/cs408/generate_full_kg_dataset.py`
- 校验：`python examples/cs408/validate_expert_seed.py`
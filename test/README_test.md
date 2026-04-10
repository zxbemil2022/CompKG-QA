# ChromaDB 计算机知识库测试

## 测试说明

这个测试程序用于验证 ChromaDB 知识库在计算机知识图谱场景下的功能。

## 测试数据

使用 `examples/cs408/cs408_auto_sample.json` 作为样例数据：
- 包含课程、概念、算法、协议等计算机知识条目；
- 每条记录至少包含 `name`、`description`，可选 `source_url`、`image_url`；
- 用于验证 JSON 解析、向量化与检索效果。

## 覆盖功能

1. 创建知识库
2. 添加文档与图片嵌入
3. 语义检索与召回结果检查
4. 元数据完整性检查
5. 异常处理与失败场景


## 运行方式

### 方法1：直接运行
```bash
python test/test_simpleRetrieval.py
或
```
### 方法2：使用 Poetry
```bash
poetry run python test/test_simpleRetrieval.py
```

### 发布前建议
```bash
python examples/cs408/eval/run_qa_regression_gate.py \
  --dataset examples/cs408/eval/qa_regression_predictions.dataset.jsonl \
  --predictions examples/cs408/eval/qa_regression_predictions.template.jsonl \
  --min-overall 0.65 \
  --min-citation 0.60
```

## 测试查询示例

测试程序会执行以下查询来验证搜索功能：

- 所有测试应该通过
- 查询应该返回相关的信息
- 相似度分数应该合理
- 元数据应该完整

## 注意事项

- 确保 ` ` 文件在项目根目录
- 测试会创建临时目录，测试完成后自动清理
- 如果测试失败，检查日志输出获取详细错误信息

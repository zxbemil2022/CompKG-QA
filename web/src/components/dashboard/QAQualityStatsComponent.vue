<template>
  <a-card title="问答质量与知识库调用" :loading="loading" class="dashboard-card">
    <template #extra>
      <a-tag color="geekblue">Accuracy Proxy</a-tag>
    </template>

    <a-row :gutter="12">
      <a-col :span="6">
        <a-statistic title="反馈总数" :value="qaStats?.total_feedbacks || 0" />
      </a-col>
      <a-col :span="6">
        <a-statistic title="满意率" :value="qaStats?.satisfaction_rate || 0" suffix="%" />
      </a-col>
      <a-col :span="6">
        <a-statistic title="KB工具成功率" :value="qaStats?.kb_tool_success_rate || 0" suffix="%" />
      </a-col>
      <a-col :span="6">
        <a-statistic title="质量覆盖率" :value="qaStats?.quality_report_coverage_rate || 0" suffix="%" />
      </a-col>
    </a-row>

    <a-divider style="margin: 12px 0" />

    <a-row :gutter="12" style="margin-bottom: 10px;">
      <a-col :span="8">
        <a-statistic title="低置信度占比" :value="qaStats?.low_confidence_rate || 0" suffix="%" />
      </a-col>
      <a-col :span="8">
        <a-statistic title="契约失败率" :value="qaStats?.contract_fail_rate || 0" suffix="%" />
      </a-col>
      <a-col :span="8">
        <a-statistic title="平均来源引用数" :value="qaStats?.avg_source_ref_count || 0" />
      </a-col>
    </a-row>

    <a-row :gutter="12">
      <a-col :span="12">
        <div class="sub-title">知识库工具调用 Top</div>
        <a-list
          size="small"
          :data-source="qaStats?.top_kb_tools || []"
          :locale="{ emptyText: '暂无调用数据' }"
        >
          <template #renderItem="{ item, index }">
            <a-list-item>
              <span>{{ index + 1 }}. {{ item.tool_name }}</span>
              <a-tag color="blue">{{ item.count }}</a-tag>
            </a-list-item>
          </template>
        </a-list>
      </a-col>

      <a-col :span="12">
        <div class="sub-title">点踩原因 Top</div>
        <a-list
          size="small"
          :data-source="qaStats?.dislike_reasons_top || []"
          :locale="{ emptyText: '暂无负反馈原因' }"
        >
          <template #renderItem="{ item, index }">
            <a-list-item>
              <span class="reason-text">{{ index + 1 }}. {{ item.reason }}</span>
              <a-tag color="volcano">{{ item.count }}</a-tag>
            </a-list-item>
          </template>
        </a-list>
      </a-col>
    </a-row>

    <a-divider style="margin: 12px 0" />

    <div class="sub-title">按 Agent 质量回归（Top 10）</div>
    <a-table
      size="small"
      :pagination="false"
      :data-source="qaStats?.quality_by_agent || []"
      :columns="agentColumns"
      row-key="agent_id"
      :locale="{ emptyText: '暂无质量分层数据' }"
    />
  </a-card>
</template>

<script setup>
const agentColumns = [
  { title: 'Agent', dataIndex: 'agent_id', key: 'agent_id', width: 180 },
  { title: '样本数', dataIndex: 'sample_count', key: 'sample_count', width: 90 },
  { title: '质量覆盖率', dataIndex: 'quality_coverage_rate', key: 'quality_coverage_rate', customRender: ({ text }) => `${text}%` },
  { title: '低置信度率', dataIndex: 'low_confidence_rate', key: 'low_confidence_rate', customRender: ({ text }) => `${text}%` },
  { title: '契约失败率', dataIndex: 'contract_fail_rate', key: 'contract_fail_rate', customRender: ({ text }) => `${text}%` },
]

defineProps({
  qaStats: {
    type: Object,
    default: () => ({})
  },
  loading: {
    type: Boolean,
    default: false
  }
})
</script>

<style scoped>
.sub-title {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 8px;
  opacity: 0.88;
}

.reason-text {
  max-width: 90%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>

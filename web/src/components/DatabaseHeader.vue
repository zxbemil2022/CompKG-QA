<template>
  <HeaderComponent
    :title="database.name || '数据库信息加载中'"
    :loading="loading"
    class="database-info-header"
  >
    <template #left>
      <a-button @click="backToDatabase" shape="circle" :icon="h(LeftOutlined)" type="text"></a-button>
    </template>
    <template #behind-title>
      <a-button type="link" @click="showEditModal" :style="{ padding: '0px', color: 'inherit' }">
        <EditOutlined />
      </a-button>
    </template>
    <template #actions>
      <div class="header-info">
        <span class="db-id">ID:
          <span style="user-select: all;">{{ database.db_id || 'N/A' }}</span>
        </span>
        <span class="file-count">{{ database.files ? Object.keys(database.files).length : 0 }} 文件</span>
        <a-tag color="blue">{{ database.embed_info?.name }}</a-tag>
        <a-tag
          :color="getKbTypeColor(database.kb_type || 'lightrag')"
          class="kb-type-tag"
          size="small"
        >
          <component :is="getKbTypeIcon(database.kb_type || 'lightrag')" class="type-icon" />
          {{ getKbTypeLabel(database.kb_type || 'lightrag') }}
        </a-tag>
      </div>
    </template>
  </HeaderComponent>

  <!-- 添加编辑对话框 -->
  <a-modal v-model:open="editModalVisible" title="编辑知识库信息">
    <template #footer>
      <a-button danger @click="deleteDatabase" style="margin-right: auto; margin-left: 0;">
        <DeleteOutlined /> 删除数据库
      </a-button>
      <a-button key="back" @click="editModalVisible = false">取消</a-button>
      <a-button key="submit" type="primary" :loading="loading" @click="handleEditSubmit">确定</a-button>
    </template>
    <a-form :model="editForm" :rules="rules" ref="editFormRef" layout="vertical">
      <a-form-item label="知识库名称" name="name" required>
        <a-input v-model:value="editForm.name" placeholder="请输入知识库名称" />
      </a-form-item>
      <a-form-item label="知识库描述" name="description">
        <a-textarea v-model:value="editForm.description" placeholder="请输入知识库描述" :rows="4" />
      </a-form-item>

      <template v-if="isLightrag">
        <a-divider orientation="left">知识图谱本体配置</a-divider>
        <a-form-item label="领域模板">
          <a-select
            v-model:value="ontologyForm.domain"
            :options="domainOptions"
            placeholder="请选择领域模板"
            @change="handleDomainChange"
          />
        </a-form-item>
        <a-form-item label="实体类型（可多选）">
          <a-select
            v-model:value="ontologyForm.entity_types"
            mode="tags"
            :options="entityTypeOptions"
            :token-separators="[',']"
            placeholder="输入或选择实体类型，回车确认"
          />
        </a-form-item>
        <a-form-item label="关系类型（可多选）">
          <a-select
            v-model:value="ontologyForm.relation_types"
            mode="tags"
            :options="relationTypeOptions"
            :token-separators="[',']"
            placeholder="输入或选择关系类型，回车确认"
          />
        </a-form-item>
        <a-divider orientation="left">NER/RE 插件配置</a-divider>
        <a-form-item label="NER 插件">
          <a-select
            v-model:value="ontologyForm.kg_ner_plugin"
            :options="pluginOptions"
          />
        </a-form-item>
        <a-form-item label="RE 插件">
          <a-select
            v-model:value="ontologyForm.kg_re_plugin"
            :options="pluginOptions"
          />
        </a-form-item>
        <a-form-item label="启用 NER 模型抽取">
          <a-switch v-model:checked="ontologyForm.kg_ner_llm_enabled" />
        </a-form-item>
        <a-form-item label="启用 RE 模型抽取">
          <a-switch v-model:checked="ontologyForm.kg_re_llm_enabled" />
        </a-form-item>
        <a-form-item label="NER 模型规格（provider/model）">
          <a-input v-model:value="ontologyForm.kg_ner_model_spec" placeholder="例如 openai/gpt-4o-mini" />
        </a-form-item>
        <a-form-item label="RE 模型规格（provider/model）">
          <a-input v-model:value="ontologyForm.kg_re_model_spec" placeholder="例如 openai/gpt-4o-mini" />
        </a-form-item>
      </template>
    </a-form>
  </a-modal>
</template>

<script setup>
import { ref, reactive, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useDatabaseStore } from '@/stores/database';
import { getKbTypeLabel, getKbTypeIcon, getKbTypeColor } from '@/utils/kb_utils';
import {
  LeftOutlined,
  EditOutlined,
  DeleteOutlined,
} from '@ant-design/icons-vue';
import HeaderComponent from '@/components/HeaderComponent.vue';
import { h } from 'vue';

const router = useRouter();
const store = useDatabaseStore();

const database = computed(() => store.database);
const loading = computed(() => store.state.databaseLoading);
const isLightrag = computed(() => String(database.value.kb_type || '').toLowerCase() === 'lightrag');
const ontologyDomains = computed(() => store.ontologyDomains || {});
const ontology = computed(() => store.ontology || { domain: 'computer', entity_types: [], relation_types: [] });

const editModalVisible = ref(false);
const editFormRef = ref(null);
const editForm = reactive({
  name: '',
  description: ''
});
const ontologyForm = reactive({
  domain: 'computer',
  entity_types: [],
  relation_types: [],
  kg_ner_plugin: 'rule',
  kg_re_plugin: 'rule',
  kg_ner_model_spec: '',
  kg_re_model_spec: '',
  kg_ner_llm_enabled: false,
  kg_re_llm_enabled: false,
});

const rules = {
  name: [{ required: true, message: '请输入知识库名称' }]
};

const domainOptions = computed(() =>
  Object.entries(ontologyDomains.value).map(([value, config]) => ({
    value,
    label: `${config.label || value} (${value})`,
  }))
);

const entityTypeOptions = computed(() => {
  const domainConfig = ontologyDomains.value[ontologyForm.domain] || {};
  return (domainConfig.entity_types || []).map(item => ({ label: item, value: item }));
});

const relationTypeOptions = computed(() => {
  const domainConfig = ontologyDomains.value[ontologyForm.domain] || {};
  return (domainConfig.relation_types || []).map(item => ({ label: item, value: item }));
});
const pluginOptions = [
  { label: '规则插件 (rule)', value: 'rule' },
  { label: '模型插件 (llm)', value: 'llm' },
];

const backToDatabase = () => {
  router.push('/database');
};

const showEditModal = () => {
  editForm.name = database.value.name || '';
  editForm.description = database.value.description || '';
  if (isLightrag.value) {
    ontologyForm.domain = ontology.value.domain || 'computer';
    ontologyForm.entity_types = [...(ontology.value.entity_types || [])];
    ontologyForm.relation_types = [...(ontology.value.relation_types || [])];
    ontologyForm.kg_ner_plugin = ontology.value.kg_ner_plugin || 'rule';
    ontologyForm.kg_re_plugin = ontology.value.kg_re_plugin || 'rule';
    ontologyForm.kg_ner_model_spec = ontology.value.kg_ner_model_spec || '';
    ontologyForm.kg_re_model_spec = ontology.value.kg_re_model_spec || '';
    ontologyForm.kg_ner_llm_enabled = !!ontology.value.kg_ner_llm_enabled;
    ontologyForm.kg_re_llm_enabled = !!ontology.value.kg_re_llm_enabled;
  }
  editModalVisible.value = true;
};

const handleDomainChange = (domain) => {
  const domainConfig = ontologyDomains.value[domain];
  if (!domainConfig) return;
  ontologyForm.entity_types = [...(domainConfig.entity_types || [])];
  ontologyForm.relation_types = [...(domainConfig.relation_types || [])];
};

const handleEditSubmit = () => {
  editFormRef.value.validate().then(async () => {
    await store.updateDatabaseInfo({
      name: editForm.name,
      description: editForm.description
    });
    if (isLightrag.value) {
      await store.updateDatabaseOntology({
        domain: ontologyForm.domain,
        entity_types: ontologyForm.entity_types,
        relation_types: ontologyForm.relation_types,
        kg_ner_plugin: ontologyForm.kg_ner_plugin,
        kg_re_plugin: ontologyForm.kg_re_plugin,
        kg_ner_model_spec: ontologyForm.kg_ner_model_spec,
        kg_re_model_spec: ontologyForm.kg_re_model_spec,
        kg_ner_llm_enabled: ontologyForm.kg_ner_llm_enabled,
        kg_re_llm_enabled: ontologyForm.kg_re_llm_enabled,
      });
    }
    editModalVisible.value = false;
  }).catch(err => {
    console.error('表单验证失败:', err);
  });
};

const deleteDatabase = () => {
  store.deleteDatabase();
};
</script>

<style scoped>
.database-info-header {
  padding: 8px;
  height: 50px;
}

.header-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.db-id {
  font-size: 12px;
  color: #666;
}

.file-count {
  font-size: 12px;
  color: #666;
}

.kb-type-tag {
  display: flex;
  align-items: center;
  gap: 4px;
}

.type-icon {
  width: 14px;
  height: 14px;
}
</style>
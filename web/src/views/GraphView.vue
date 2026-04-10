<template>
  <div class="database-empty" v-if="!state.showPage">
    <a-empty>
      <template #description>
        <span>
          前往 <router-link to="/setting" style="color: var(--main-color); font-weight: bold;">设置</router-link> 页面启用知识图谱。
        </span>
      </template>
    </a-empty>
  </div>
  <div class="graph-container layout-container" v-else>
    <HeaderComponent
      title="图数据库"
    >
      <template #actions>
        <div class="status-wrapper">
          <div class="status-indicator" :class="graphStatusClass"></div>
          <span class="status-text">{{ graphStatusText }}</span>
        </div>
        <a-button type="default" @click="openLink('http://localhost:7474/')" :icon="h(GlobalOutlined)">
          Neo4j 浏览器
        </a-button>
        <a-button type="default" @click="autoBuildComputerKG" :loading="state.autoBuilding" :icon="h(HighlightOutlined)">
          自动构建计算机知识图谱
        </a-button>
        <a-button type="default" @click="autoBuild408SubjectKGs" :loading="state.autoBuildingSubjects">
          构建408学科子图谱
        </a-button>
        <a-button type="default" @click="loadBuiltinFullKG" :loading="state.loadingBuiltinKG">
          加载内置408完整版图谱
        </a-button>
        <a-button type="primary" @click="state.showModal = true" ><UploadOutlined/> 上传文件</a-button>
        <a-button v-if="unindexedCount > 0" type="primary" @click="indexNodes" :loading="state.indexing">
          <SyncOutlined v-if="!state.indexing"/> 为{{ unindexedCount }}个节点添加索引
        </a-button>
      </template>
    </HeaderComponent>

    <div class="container-outter">
    <div class="graph-workspace graph-layout">
        <aside class="workspace-left qa-side-panel">
          <div class="workspace-block-title">筛选区</div>
          <div class="workspace-filter-box">
            <a-select
              v-model:value="state.selectedSubject"
              style="width: 100%; margin-bottom: 8px"
              :options="subjectOptions"
              :loading="state.loadingSubjects"
              @change="loadSampleNodes"
            />
            <a-input
              v-model:value="state.searchInput"
              :placeholder="state.selectedSubject ? `在${state.selectedSubject}中查询实体` : '输入要查询的实体'"
              @keydown.enter="onSearch"
              allow-clear
            >
              <template #suffix>
                <component :is="state.searchLoading ? LoadingOutlined : SearchOutlined" @click="onSearch" />
              </template>
            </a-input>
          </div>
          <div class="qa-panel-header">💬 （图谱 + 知识库）</div>
          <div class="qa-messages">
            <div
              v-for="(msg, idx) in state.qaMessages"
              :key="`${msg.role}-${idx}`"
              class="qa-message"
              :class="msg.role"
            >
              <div class="qa-role">{{ msg.role === 'user' ? '我' : '助手' }}</div>
              <div class="qa-content">{{ msg.content }}</div>
              <div v-if="msg.citations?.length" class="qa-citations">
                <div>【参考来源】</div>
                <ol>
                  <li v-for="(c, i) in msg.citations" :key="`${c}-${i}`">{{ c }}</li>
                </ol>
              </div>
            </div>
          </div>
          <div class="qa-input-row">
            <a-input
              v-model:value="state.questionInput"
              placeholder="输入问题并回车"
              @keydown.enter="askGraphQuestion"
              allow-clear
            />
            <a-button type="primary" :loading="state.qaLoading" @click="askGraphQuestion">发送</a-button>
          </div>
        </aside>
        <section class="workspace-main">
        <div class="graph-workspace-container">
        <GraphCanvas
          ref="graphRef"
          :graph-data="graphData"
          :highlight-keywords="state.highlightKeywords"
          :layout-options="layoutOptions"
          :node-style-options="nodeStyleOptions"
        >
        <template #top>
          <div class="actions">
            <div class="actions-left">
              <a-tag color="blue">Graph Workspace</a-tag>
              <a-tag color="purple">容器化视图</a-tag>
            </div>
            <div class="actions-right">
             <a-select
                v-model:value="state.layoutMode"
                style="width: 140px"
                :options="layoutModeOptions"
                @change="() => graphRef?.refreshGraph?.()"
              />
              <a-select
                v-model:value="state.nodeShape"
                style="width: 130px"
                :options="nodeShapeOptions"
                @change="() => graphRef?.refreshGraph?.()"
              />
              <a-button type="default" @click="state.showInfoModal = true" :icon="h(InfoCircleOutlined)">
                说明
              </a-button>
              <a-input
                v-model:value="sampleNodeCount"
                placeholder="查询三元组数量"
                style="width: 100px"
                @keydown.enter="loadSampleNodes"
                :loading="state.fetching"
              >
                <template #suffix>
                  <component :is="state.fetching ? LoadingOutlined : ReloadOutlined" @click="loadSampleNodes" />
                </template>
              </a-input>
            </div>
          </div>
        </template>
        <template #content>
          <a-empty v-show="graphData.nodes.length === 0" style="padding: 4rem 0;">
            <template #description>
              当前图谱无可展示内容，可点击“加载内置408完整版图谱”快速初始化。
            </template>
            <a-button type="primary" @click="loadBuiltinFullKG" :loading="state.loadingBuiltinKG">
              一键加载内置图谱
            </a-button>
          </a-empty>
        </template>
        <template #bottom>
          <div class="footer"></div>
        </template>
      </GraphCanvas>
      </div>
      <div class="workspace-statusbar">
         <a-tag :bordered="false" v-for="tag in graphTags" :key="tag.key" :color="tag.type">{{ tag.text }}</a-tag>
        <div v-if="state.qaAnswer && state.qaMessages.length === 0" class="qa-answer">{{ state.qaAnswer }}</div>
      </div>
      </section>
    </div>
    </div>

    <a-modal
      :open="state.showModal" title="上传文件"
      @ok="addDocumentByFile"
      @cancel="() => state.showModal = false"
      ok-text="添加到图数据库" cancel-text="取消"
      :confirm-loading="state.processing">
      <div class="upload">
        <div class="note">
          <p>上传的文件内容参考 test/data/A_Dream_of_Red_Mansions_tiny.jsonl 中的格式：</p>
        </div>
        <a-upload-dragger
          class="upload-dragger"
          v-model:fileList="fileList"
          name="file"
          :fileList="fileList"
          :max-count="1"
          action="/api/knowledge/files/upload"
          :headers="getAuthHeaders()"
          @change="handleFileUpload"
          @drop="handleDrop"
        >
          <p class="ant-upload-text">点击或者把文件拖拽到这里上传</p>
          <p class="ant-upload-hint">
            目前仅支持上传 jsonl 文件。
          </p>
        </a-upload-dragger>
      </div>
    </a-modal>

    <!-- 说明弹窗 -->
    <a-modal
      :open="state.showInfoModal"
      title="图数据库说明"
      @cancel="() => state.showInfoModal = false"
      :footer="null"
      width="600px"
    >
      <div class="info-content">
        <p>本页面展示的是 Neo4j 图数据库中的知识图谱信息。</p>
        <p>具体展示内容包括：</p>
        <ul>
          <li>带有 <code>Entity</code> 标签的节点</li>
          <li>带有 <code>RELATION</code> 类型的关系边</li>
        </ul>
        <p>注意：</p>
        <ul>
          <li>这里仅展示用户上传的实体和关系，不包含知识库中自动创建的图谱。</li>
          <li>查询逻辑基于 <code>graphbase.py</code> 中的 <code>get_sample_nodes</code> 方法实现：</li>
        </ul>
        <pre><code>MATCH (n:Entity)-[r]-&gt;(m:Entity)
RETURN
    {id: elementId(n), name: n.name} AS h,
    {type: r.type, source_id: elementId(n), target_id: elementId(m)} AS r,
    {id: elementId(m), name: m.name} AS t
LIMIT $num</code></pre>
        <p>如需查看完整的 Neo4j 数据库内容，请使用 "Neo4j 浏览器" 按钮访问原生界面。</p>
      </div>
    </a-modal>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, h, defineAsyncComponent } from 'vue';
import { message } from 'ant-design-vue';
import { useConfigStore } from '@/stores/config';
import { UploadOutlined, SyncOutlined, GlobalOutlined, InfoCircleOutlined, SearchOutlined, ReloadOutlined, LoadingOutlined, HighlightOutlined } from '@ant-design/icons-vue';
import HeaderComponent from '@/components/HeaderComponent.vue';
import { neo4jApi } from '@/apis/graph_api';
import { useUserStore } from '@/stores/user';
const GraphCanvas = defineAsyncComponent(() => import('@/components/GraphCanvas.vue'));

const configStore = useConfigStore();
const cur_embed_model = computed(() => configStore.config?.embed_model);
const modelMatched = computed(() => !graphInfo?.value?.embed_model_name || graphInfo.value.embed_model_name === cur_embed_model.value)

const graphRef = ref(null)
const graphInfo = ref(null)
const fileList = ref([]);
const sampleNodeCount = ref(100);
const graphData = reactive({
  nodes: [],
  edges: [],
});

const state = reactive({
  fetching: false,
  loadingGraphInfo: false,
  searchInput: '',
  searchLoading: false,
  questionInput: '',
  qaLoading: false,
  qaAnswer: '',
  qaSessionId: `graph-${Date.now()}`,
  qaMessages: [],
  highlightKeywords: [],
  showModal: false,
  showInfoModal: false,
  processing: false,
  indexing: false,
  autoBuilding: false,
  autoBuildingSubjects: false,
  loadingSubjects: false,
  loadingBuiltinKG: false,
  subjectsApiAvailable: true,
  capabilities: {
    subjects_api: false,
    auto_build_computer_kg: false,
    auto_build_408_subject_kgs: false,
    add_entities_skip_embedding: false,
    builtin_expert_seed_supported: false,
  },
  layoutMode: 'network',
  nodeShape: 'circle',
  selectedSubject: '',
  subjects: [],
  showPage: true,
})

const subjectOptions = computed(() => {
  const options = [{ value: '', label: '全部学科（总图谱）' }];
  for (const item of state.subjects) {
    options.push({ value: item.subject, label: `${item.subject} (${item.count})` });
  }
  return options;
});

const layoutModeOptions = [
  { value: 'network', label: '网状' },
  { value: 'tree', label: '树状' },
  { value: 'circular', label: '圆形' },
  { value: 'star', label: '星盘' },
  { value: 'polygon', label: '多边形' },
]

const nodeShapeOptions = [
  { value: 'circle', label: '圆形节点' },
  { value: 'rect', label: '方形节点' },
  { value: 'diamond', label: '菱形节点' },
  { value: 'triangle', label: '三角节点' },
]

const layoutOptions = computed(() => {
  const mode = state.layoutMode;
  if (mode === 'tree') return { type: 'dagre', rankdir: 'LR', nodesep: 25, ranksep: 70 };
  if (mode === 'circular') return { type: 'circular', preventOverlap: true };
  if (mode === 'star') return { type: 'concentric', preventOverlap: true };
  if (mode === 'polygon') return { type: 'grid', cols: 12, sortBy: 'degree' };
  return { type: 'd3-force', preventOverlap: true };
});

const nodeStyleOptions = computed(() => ({
  type: state.nodeShape || 'circle',
}));

// 计算未索引节点数量
const unindexedCount = computed(() => {
  return graphInfo.value?.unindexed_node_count || 0;
});

const loadGraphInfo = () => {
  state.loadingGraphInfo = true
  neo4jApi.getInfo()
    .then(data => {
      console.log(data)
      graphInfo.value = data.data
      state.loadingGraphInfo = false
    })
    .catch(error => {
      console.error(error)
      message.error(error.message || '加载图数据库信息失败')
      state.loadingGraphInfo = false
    })
}

const addDocumentByFile = () => {
  state.processing = true
  const files = fileList.value.filter(file => file.status === 'done').map(file => file.response.file_path)
  neo4jApi.addEntities(files[0])
    .then((data) => {
      if (data.status === 'success') {
        message.success(data.message);
        state.showModal = false;
      } else {
        throw new Error(data.message);
      }
    })
    .catch((error) => {
      console.error(error)
      message.error(error.message || '添加文件失败');
    })
    .finally(() => state.processing = false)
};

const loadSampleNodes = () => {
  state.fetching = true
  neo4jApi.getSampleNodes('neo4j', sampleNodeCount.value, state.selectedSubject)
    .then((data) => {
      graphData.nodes = data.result.nodes
      graphData.edges = data.result.edges
      console.log(graphData)
      if (graphData.nodes.length === 0) {
        message.warning('当前图数据库为空，可点击“加载内置408完整版图谱”初始化数据');
      }
      // 初次加载后兜底刷新一次，避免容器初次可见尺寸未稳定
      setTimeout(() => graphRef.value?.refreshGraph?.(), 500)
    })
    .catch((error) => {
      console.error(error)
      message.error(error.message || '加载节点失败');
    })
    .finally(() => state.fetching = false)
}

const onSearch = () => {
  if (state.searchLoading) {
    message.error('请稍后再试')
    return
  }

  if (graphInfo?.value?.embed_model_name !== cur_embed_model.value) {
    // 可选：提示模型不一致
  }

  if (!state.searchInput) {
    message.error('请输入要查询的实体')
    return
  }

  state.searchLoading = true
  neo4jApi.queryNode(state.searchInput, state.selectedSubject)
    .then((data) => {
      if (!data.result || !data.result.nodes || !data.result.edges) {
        throw new Error('返回数据格式不正确');
      }
      graphData.nodes = data.result.nodes
      graphData.edges = data.result.edges
      state.highlightKeywords = [state.searchInput]
      if (graphData.nodes.length === 0) {
        message.info('未找到相关实体')
      }
      console.log(data)
      console.log(graphData)
      graphRef.value?.refreshGraph?.()
    })
    .catch((error) => {
      console.error('查询错误:', error);
      message.error(`查询出错：${error.message || '未知错误'}`);
    })
    .finally(() => state.searchLoading = false)
};

const askGraphQuestion = () => {
  const question = state.questionInput?.trim();
  if (!question) {
    message.warning('请输入问题');
    return;
  }
  if (state.qaLoading) return;

  state.qaLoading = true;
  state.qaAnswer = '';
  // 发送后立即清空输入框，避免“输入框状态残留”
  state.questionInput = '';
  neo4jApi.askQuestion({
    question,
    subject: state.selectedSubject,
    kgdb_name: 'neo4j',
    session_id: state.qaSessionId,
  })
    .then((data) => {
      const payload = data?.data || {};
      state.qaAnswer = payload.answer || '未返回答案';
      state.qaMessages.push({ role: 'user', content: question, citations: [] });
      state.qaMessages.push({
        role: 'assistant',
        content: state.qaAnswer,
        citations: (payload.vector_evidence || []).map(item => item.source_db).filter(Boolean),
      });
      if (payload?.graph?.nodes && payload?.graph?.edges) {
        graphData.nodes = payload.graph.nodes;
        graphData.edges = payload.graph.edges;
      } else {
        graphData.nodes = [];
        graphData.edges = [];
      }
      state.highlightKeywords = payload.highlight_nodes || [];
      graphRef.value?.refreshGraph?.();
    })
    .catch((error) => {
      console.error(error);
      message.error(error.message || '图谱问答失败');
      // 失败时恢复原问题，便于用户一键重试
      state.questionInput = question;
    })
    .finally(() => {
      state.qaLoading = false;
    });
};

onMounted(() => {
  loadCapabilities();
  loadGraphInfo();
  loadSampleNodes();
  loadSubjects();
});

const handleFileUpload = (event) => {
  console.log(event)
  console.log(fileList.value)
}

const handleDrop = (event) => {
  console.log(event)
  console.log(fileList.value)
}

const graphStatusClass = computed(() => {
  if (state.loadingGraphInfo) return 'loading';
  return graphInfo.value?.status === 'open' ? 'open' : 'closed';
});

const graphStatusText = computed(() => {
  if (state.loadingGraphInfo) return '加载中';
  return graphInfo.value?.status === 'open' ? '已连接' : '已关闭';
});

// 新增：将图谱信息拆分为多条标签用于展示
const graphTags = computed(() => {
  const tags = [];
  const dbName = graphInfo.value?.graph_name;
  const entityCount = graphInfo.value?.entity_count;
  const relationCount = graphInfo.value?.relationship_count;

  if (dbName) tags.push({ key: 'name', text: `图谱 ${dbName}`, type: 'blue' });
  if (typeof entityCount === 'number') tags.push({ key: 'entities', text: `实体 ${graphData.nodes.length} of ${entityCount}`, type: 'success' });
  if (typeof relationCount === 'number') tags.push({ key: 'relations', text: `关系 ${graphData.edges.length} of ${relationCount}`, type: 'purple' });
  if (unindexedCount.value > 0) tags.push({ key: 'unindexed', text: `未索引 ${unindexedCount.value}`, type: 'warning' });

  return tags;
});

// 为未索引节点添加索引
const indexNodes = () => {
  // 判断 embed_model_name 是否相同
  if (!modelMatched.value) {
    message.error(`向量模型不匹配，无法添加索引，当前向量模型为 ${cur_embed_model.value}，图数据库向量模型为 ${graphInfo.value?.embed_model_name}`)
    return
  }

  if (state.processing) {
    message.error('后台正在处理，请稍后再试')
    return
  }

  state.indexing = true;
  neo4jApi.indexEntities('neo4j')
    .then(data => {
      message.success(data.message || '索引添加成功');
      // 刷新图谱信息
      loadGraphInfo();
    })
    .catch(error => {
      console.error(error);
      message.error(error.message || '添加索引失败');
    })
    .finally(() => {
      state.indexing = false;
    });
};

const autoBuildComputerKG = () => {
  if (state.autoBuilding || state.processing || state.indexing) {
    message.warning('当前仍有任务在处理中，请稍后再试');
    return;
  }

  state.autoBuilding = true;
  neo4jApi.autoBuildComputerKG({
    file_path: 'examples/cs408/cs408_auto_sample.json',
    clear_existing: false,
  })
    .then((data) => {
      message.success(data.message || '自动构建完成');
      loadGraphInfo();
      loadSampleNodes();
      loadSubjects();
    })
    .catch((error) => {
      console.error(error);
      if ((error?.message || '').includes('404')) {
        message.warning('后端未提供自动构建接口，已回退为加载内置图谱');
        loadBuiltinFullKG();
        return;
      }
      message.error(error.message || '自动构建失败');
    })
    .finally(() => {
      state.autoBuilding = false;
    });
};

const autoBuild408SubjectKGs = () => {
  if (state.autoBuildingSubjects || state.autoBuilding || state.processing || state.indexing) {
    message.warning('当前仍有任务在处理中，请稍后再试');
    return;
  }

  state.autoBuildingSubjects = true;
  neo4jApi.autoBuild408SubjectKGs({
    file_path: 'examples/cs408/cs408_auto_sample.json',
    clear_existing: false,
  })
    .then((data) => {
      message.success(data.message || '408学科子图谱构建完成');
      loadGraphInfo();
      loadSampleNodes();
      loadSubjects();
    })
    .catch((error) => {
      console.error(error);
      if ((error?.message || '').includes('404')) {
        message.warning('后端未提供408子图谱接口，已回退为加载内置图谱');
        loadBuiltinFullKG();
        return;
      }
      message.error(error.message || '408学科子图谱构建失败');
    })
    .finally(() => {
      state.autoBuildingSubjects = false;
    });
};

const loadBuiltinFullKG = () => {
  if (state.loadingBuiltinKG || state.autoBuilding || state.autoBuildingSubjects || state.processing || state.indexing) {
    message.warning('当前仍有任务在处理中，请稍后再试');
    return;
  }

  state.loadingBuiltinKG = true;
  neo4jApi.addEntities('examples/cs408/cs408_expert_seed.jsonl', 'neo4j', true)
    .then((data) => {
      if (data.status !== 'success') {
        throw new Error(data.message || '内置专家图谱导入失败');
      }
      message.success('内置专家版408图谱已导入');
      loadGraphInfo();
      loadSampleNodes();
      loadSubjects();
    })
    .catch((error) => {
      console.error(error);
      message.error(error.message || '加载内置图谱失败');
    })
    .finally(() => {
      state.loadingBuiltinKG = false;
    });
};

const loadSubjects = () => {
  if (!state.subjectsApiAvailable || !state.capabilities.subjects_api) {
    state.subjects = [];
    return;
  }

  state.loadingSubjects = true;
  neo4jApi.getSubjects('neo4j')
    .then((data) => {
      state.subjects = data?.data?.subjects || [];
    })
    .catch((error) => {
      console.error(error);
      if ((error?.message || '').includes('404')) {
        state.subjectsApiAvailable = false;
      }
      state.subjects = [];
    })
    .finally(() => {
      state.loadingSubjects = false;
    });
}

const loadCapabilities = () => {
  neo4jApi.getCapabilities()
    .then((data) => {
      state.capabilities = {
        ...state.capabilities,
        ...(data?.data || {})
      }
      state.subjectsApiAvailable = !!state.capabilities.subjects_api
    })
    .catch((error) => {
      console.error(error)
      state.capabilities = {
        ...state.capabilities,
        auto_build_computer_kg: false,
        auto_build_408_subject_kgs: false,
        subjects_api: false,
      }
      state.subjectsApiAvailable = false
      message.warning('后端能力探测失败，部分高级功能已降级为基础模式')
    })
}

const getAuthHeaders = () => {
  const userStore = useUserStore();
  return userStore.getAuthHeaders();
};

const openLink = (url) => {
  window.open(url, '_blank')
}

</script>

<style lang="less" scoped>
@graph-header-height: 55px;

.graph-container {
  padding: 0;

  .header-container {
    height: @graph-header-height;
  }
}

.status-wrapper {
  display: flex;
  align-items: center;
  margin-right: 16px;
  font-size: 14px;
  color: rgba(0, 0, 0, 0.65);
}

.status-text {
  margin-left: 8px;
}

.status-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  display: inline-block;

  &.loading {
    background-color: #faad14;
    animation: pulse 1.5s infinite ease-in-out;
  }

  &.open {
    background-color: #52c41a;
  }

  &.closed {
    background-color: #f5222d;
  }
}

@keyframes pulse {
  0% {
    transform: scale(0.8);
    opacity: 0.5;
  }
  50% {
    transform: scale(1.2);
    opacity: 1;
  }
  100% {
    transform: scale(0.8);
    opacity: 0.5;
  }
}


.upload {
  margin-bottom: 20px;

  .upload-dragger {
    margin: 0px;
  }
}

.container-outter {
  width: 100%;
  height: calc(100vh - @graph-header-height);
  overflow: hidden;
  background: var(--gray-10);

.graph-workspace {
    height: 100%;
    display: grid;
    grid-template-columns: 360px 1fr;
    gap: 12px;
    padding: 10px;
  }

  .workspace-main {
    display: flex;
    flex-direction: column;
    min-width: 0;
    gap: 8px;
  }

  .graph-workspace-container {
    flex: 1;
    min-height: 0;
    border: 1px solid #dbe3ee;
    border-radius: 12px;
    background: #fff;
    box-shadow: inset 0 0 0 1px #f4f6fa;
    overflow: hidden;
  }

  .workspace-statusbar {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 8px 12px;
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }

  .qa-side-panel {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .workspace-block-title {
    font-size: 12px;
    color: #64748b;
    font-weight: 600;
    padding: 8px 10px 4px;
  }

  .workspace-filter-box {
    padding: 0 10px 10px;
    border-bottom: 1px solid #eef2f7;
  }

  .qa-panel-header {
    font-weight: 600;
    padding: 10px 12px;
    border-bottom: 1px solid #eef2f7;
  }

  .qa-messages {
    flex: 1;
    overflow-y: auto;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .qa-message {
    background: #f8fafc;
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 13px;
    &.user {
      background: #eff6ff;
    }
  }

  .qa-role {
    font-weight: 600;
    margin-bottom: 4px;
    color: #475569;
  }

  .qa-content {
    white-space: pre-line;
    line-height: 1.6;
  }

  .qa-citations {
    margin-top: 6px;
    font-size: 12px;
    color: #64748b;
    ol {
      margin: 4px 0 0;
      padding-left: 18px;
    }
  }

  .qa-input-row {
    display: flex;
    gap: 8px;
    padding: 10px;
    border-top: 1px solid #eef2f7;
  }

  .actions,
  .footer {
    display: flex;
    justify-content: space-between;
    margin: 12px 0;
    padding: 0 12px;
    width: 100%;
  }
}

.actions {
  top: 0;

  .actions-left, .actions-right {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  :deep(.ant-input) {
    padding: 2px 10px;
  }

  button {
    height: 37px;
    box-shadow: none;
  }
}

.qa-answer {
  max-width: 45%;
  font-size: 13px;
  color: #334155;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 8px 10px;
  line-height: 1.5;
  white-space: pre-line;
}
</style>
<template>
  <div class="routing-card">
    <div class="routing-header">
      <el-icon class="routing-icon"><Connection /></el-icon>
      <span class="routing-title">Routing</span>
      <el-tag :type="confidenceTagType" size="small" class="confidence-tag">
        {{ Math.round(confidence * 100) }}% confidence
      </el-tag>
    </div>
    <div class="routing-body">
      <div class="routing-agent">
        <span class="label">Intent</span>
        <el-tag :type="agentTagType" size="small">
          {{ agentIcon[agentType] ?? '' }} {{ agentDisplay }}
        </el-tag>
      </div>
      <div class="routing-reason">
        <span class="label">Reason</span>
        <span class="reason-text">{{ reason }}</span>
      </div>
      <div class="routing-mode">
        <span class="label">Mode</span>
        <span class="mode-text">{{ modeDisplay }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Connection } from '@element-plus/icons-vue'

const props = defineProps<{
  agentType: string
  agentDisplay: string
  confidence: number
  reason: string
  executionMode: string
}>()

const agentIcon: Record<string, string> = {
  qa: '🤖', exam: '📝', resume: '📄', interview: '🎤', multi_agent: '🧭', clarify: '❓',
}

const agentTagType = computed(() => {
  const map: Record<string, string> = {
    qa: 'success', exam: 'warning', resume: 'info', interview: 'danger',
  }
  return map[props.agentType] ?? 'info'
})

const confidenceTagType = computed(() => {
  if (props.confidence >= 0.85) return 'success'
  if (props.confidence >= 0.65) return 'warning'
  return 'danger'
})

const modeDisplay = computed(() => {
  const map: Record<string, string> = {
    single:   'Single agent',
    pipeline: 'Several agents in order',
    clarify:  'Needs clarification',
  }
  return map[props.executionMode] ?? props.executionMode
})
</script>

<style scoped>
.routing-card {
  background: #f0f7ff;
  border: 1px solid #bae0ff;
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 4px;
  font-size: 13px;
}
.routing-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
  font-weight: 500;
  color: #1677ff;
}
.routing-icon {
  font-size: 14px;
}
.routing-title {
  flex: 1;
}
.confidence-tag {
  font-size: 11px;
}
.routing-body {
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.routing-agent,
.routing-reason,
.routing-mode {
  display: flex;
  align-items: flex-start;
  gap: 8px;
}
.label {
  color: #8c8c8c;
  min-width: 52px;
  flex-shrink: 0;
}
.reason-text,
.mode-text {
  color: #262626;
  line-height: 1.5;
}
</style>

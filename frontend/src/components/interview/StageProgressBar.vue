<template>
  <div class="stage-bar">
    <div
      v-for="(stage, i) in stages"
      :key="stage.key"
      class="stage-item"
      :class="stateOf(stage.key)"
    >
      <div class="stage-dot">
        <el-icon v-if="stateOf(stage.key) === 'done'"><Check /></el-icon>
        <span v-else-if="stateOf(stage.key) === 'skipped'">–</span>
        <span v-else>{{ i + 1 }}</span>
      </div>
      <div class="stage-label">{{ stage.label }}<template v-if="stateOf(stage.key) === 'skipped'"> (skipped)</template></div>
      <div v-if="i < stages.length - 1" class="stage-line" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Check } from '@element-plus/icons-vue'

const stages = [
  { key: 'warmup', label: 'Warm-up' },
  { key: 'tech_base', label: 'Technical' },
  { key: 'project', label: 'Projects' },
  { key: 'closing', label: 'Your questions' },
  { key: 'finished', label: 'Finished' },
]

const stageOrder = stages.map(s => s.key)

// reached: stages the interview went through (from the backend). Without it every stage
// before the current one counts as done.
const props = defineProps<{ current: string; reached?: string[] }>()

const currentIndex = computed(() => stageOrder.indexOf(props.current))

function stateOf(key: string): 'done' | 'skipped' | 'active' | 'pending' {
  const index = stageOrder.indexOf(key)
  if (key === props.current) return 'active'
  if (index > currentIndex.value) return 'pending'
  return !props.reached?.length || props.reached.includes(key) ? 'done' : 'skipped'
}
</script>

<style scoped>
.stage-bar {
  display: flex;
  align-items: center;
  padding: 12px 0;
  position: relative;
}
.stage-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  flex: 1;
}
.stage-dot {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 500;
  border: 2px solid #d9d9d9;
  background: #fff;
  color: #8c8c8c;
  z-index: 1;
}
.stage-item.done .stage-dot {
  background: #52c41a;
  border-color: #52c41a;
  color: #fff;
}
.stage-item.active .stage-dot {
  background: #1677ff;
  border-color: #1677ff;
  color: #fff;
}
.stage-label {
  font-size: 12px;
  margin-top: 4px;
  color: #8c8c8c;
}
.stage-item.active .stage-label { color: #1677ff; font-weight: 500; }
.stage-item.done .stage-label { color: #52c41a; }
.stage-line {
  position: absolute;
  top: 14px;
  left: 50%;
  width: 100%;
  height: 2px;
  background: #d9d9d9;
  z-index: 0;
}
.stage-item.done .stage-line { background: #52c41a; }
.stage-item.skipped .stage-dot { border-style: dashed; color: #bfbfbf; }
.stage-item.skipped .stage-label { color: #bfbfbf; }
</style>

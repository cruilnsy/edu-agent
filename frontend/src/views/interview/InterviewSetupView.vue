<template>
  <div class="interview-setup">
    <el-card style="max-width: 560px">
      <template #header><span>🎤 Start a mock interview</span></template>

      <el-form :model="form" label-width="150px">
        <el-form-item label="Target position">
          <el-input
            v-model="form.targetPosition"
            placeholder="e.g. Backend Engineer, Frontend Engineer, AI Engineer"
            maxlength="100"
          />
        </el-form-item>
        <el-form-item label="Resume (optional)">
          <el-select
            v-model="form.resumeReviewId"
            placeholder="Use a reviewed resume: the interviewer asks about its projects"
            clearable
            style="width: 100%"
          >
            <el-option
              v-for="r in resumeHistory"
              :key="r.review_id"
              :label="`${formatTime(r.created_at)} · score ${r.weighted_score?.toFixed(0) ?? '-'}`"
              :value="r.review_id"
            />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button
            type="primary"
            :loading="loading"
            :disabled="!form.targetPosition.trim()"
            @click="startInterview"
          >
            Start interview
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- Earlier interviews -->
    <el-card style="margin-top: 16px; max-width: 800px">
      <template #header>
        <span>My interviews</span>
        <el-button style="float: right" :icon="Refresh" circle size="small" @click="fetchHistory" />
      </template>
      <el-empty v-if="!history.length" description="No interviews yet" />
      <el-table v-else :data="history" size="small">
        <el-table-column prop="target_position" label="Position" />
        <el-table-column label="Score" width="100">
          <template #default="{ row }">
            <span v-if="row.overall_score !== null && row.overall_score !== undefined">{{ row.overall_score }}</span>
            <span v-else>—</span>
          </template>
        </el-table-column>
        <el-table-column label="Status" width="110">
          <template #default="{ row }">
            <el-tag :type="row.status === 'finished' ? 'success' : 'warning'" size="small">
              {{ row.status === 'finished' ? 'Finished' : 'In progress' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="Started" width="180">
          <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="Action" width="110">
          <template #default="{ row }">
            <el-button
              text type="primary" size="small"
              @click="router.push(`/interview/${row.session_id}`)"
            >
              {{ row.status === 'finished' ? 'View report' : 'Continue' }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { formatTime } from '@/utils/format'
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { Refresh } from '@element-plus/icons-vue'
import { interviewApi, type SessionListItem } from '@/api/interview'
import { resumeApi, type ReviewListItem } from '@/api/resume'

const router = useRouter()
const loading = ref(false)
const form = reactive({ targetPosition: '', resumeReviewId: '' })
const history = ref<SessionListItem[]>([])
const resumeHistory = ref<ReviewListItem[]>([])

async function startInterview() {
  if (!form.targetPosition.trim()) return
  loading.value = true
  try {
    const { data } = await interviewApi.startSession({
      target_position: form.targetPosition,
      resume_review_id: form.resumeReviewId || null,
    })
    router.push({ path: `/interview/${data.session_id}`, state: { openingMessage: data.message } })
  } finally {
    loading.value = false
  }
}

async function fetchHistory() {
  try {
    const { data } = await interviewApi.listSessions()
    history.value = data.items
  } catch { /* ignore */ }
}

onMounted(async () => {
  fetchHistory()
  try {
    const { data } = await resumeApi.listReviews()
    resumeHistory.value = data.items.filter(r => r.status === 'done')
  } catch { /* ignore */ }
})
</script>

<style scoped>
.interview-setup { max-width: 800px; }
</style>

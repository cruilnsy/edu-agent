<template>
  <div class="resume-report">
    <el-page-header @back="router.push('/resume')" title="Back" content="Resume review" />

    <div v-if="loading" style="margin-top: 24px">
      <el-skeleton :rows="10" animated />
    </div>

    <template v-else-if="review?.status === 'done' && review.dimension_scores">
      <!-- Overall score -->
      <el-card style="margin: 16px 0; text-align: center">
        <div class="total-score">{{ review.weighted_score?.toFixed(1) }}</div>
        <div class="total-label">Overall score (out of 100)</div>
        <div class="total-comment" v-if="review.summary?.overall_comment">
          {{ review.summary.overall_comment }}
        </div>
      </el-card>

      <!-- Six dimensions -->
      <el-row :gutter="16" style="margin-bottom: 16px">
        <el-col
          :span="8"
          v-for="dim in review.dimension_scores"
          :key="dim.key"
          style="margin-bottom: 16px"
        >
          <DimensionScoreCard
            :dimension="dim.dimension"
            :score="dim.score"
            :weight="dim.weight"
            :issues="dim.issues"
            :suggestions="dim.suggestions"
          />
        </el-col>
      </el-row>

      <!-- Issues -->
      <el-card v-if="review.issues?.length" style="margin-bottom: 16px">
        <template #header>Issues (by priority)</template>
        <el-table :data="review.issues" size="small">
          <el-table-column label="Priority" width="80">
            <template #default="{ row }">
              <el-tag :type="priorityType(row.priority)" size="small">{{ row.priority }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="dimension" label="Dimension" width="100" />
          <el-table-column prop="description" label="Issue" />
          <el-table-column prop="location" label="Where" width="160" />
          <el-table-column prop="suggestion" label="Suggestion" />
        </el-table>
      </el-card>

      <!-- Summary -->
      <el-card v-if="review.summary">
        <template #header>Summary</template>
        <el-row :gutter="16">
          <el-col :span="12">
            <div class="summary-section">
              <div class="summary-title">✅ Strengths</div>
              <ul>
                <li v-for="(h, i) in review.summary.highlights" :key="i">{{ h }}</li>
              </ul>
            </div>
          </el-col>
          <el-col :span="12">
            <div class="summary-section">
              <div class="summary-title">🔧 Most important improvements</div>
              <ul>
                <li v-for="(c, i) in review.summary.core_improvements" :key="i">{{ c }}</li>
              </ul>
            </div>
          </el-col>
        </el-row>
        <div class="fit-assessment" v-if="review.summary.fit_assessment">
          <b>Fit for the role: </b>{{ review.summary.fit_assessment }}
        </div>
      </el-card>
    </template>

    <el-card v-else-if="review?.status === 'processing'" style="margin-top: 16px; text-align: center">
      <el-icon class="is-loading" style="font-size: 32px; color: #1677ff"><Loading /></el-icon>
      <p>The AI is reviewing your resume...</p>
      <p style="color: #8c8c8c; font-size: 13px">This usually takes 30-60 seconds</p>
      <p v-if="transientError" style="color: #d46b08; font-size: 13px">{{ transientError }}</p>
    </el-card>

    <el-card v-else-if="review?.status === 'failed'" style="margin-top: 16px; text-align: center">
      <el-result icon="error" title="Review failed" :sub-title="review?.error_msg || 'Please upload your resume again'">
        <template #extra>
          <el-button type="primary" @click="router.push('/resume')">Upload again</el-button>
        </template>
      </el-result>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Loading } from '@element-plus/icons-vue'
import { resumeApi, type ReviewDetail } from '@/api/resume'
import DimensionScoreCard from '@/components/resume/DimensionScoreCard.vue'

const route = useRoute()
const router = useRouter()
const loading = ref(true)
const review = ref<ReviewDetail | null>(null)
const transientError = ref('')
let pollTimer: ReturnType<typeof setTimeout> | null = null
let mounted = false

function priorityType(p: string) {
  if (p === 'high') return 'danger'
  if (p === 'medium') return 'warning'
  return 'info'
}

async function fetchReview() {
  if (!mounted) return
  try {
    const { data } = await resumeApi.getReview(route.params.reviewId as string)
    if (!mounted) return  // navigated away while request was in-flight
    transientError.value = ''
    review.value = data
    if (data.status !== 'processing') stopPoll()
  } catch {
    if (mounted) transientError.value = 'The service is briefly unavailable; retrying...'
  } finally {
    if (mounted) loading.value = false
  }
}

function stopPoll() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null }
}

onMounted(() => {
  mounted = true
  fetchReview()
  // A chained setTimeout instead of setInterval, so requests cannot pile up
  const schedulePoll = () => {
    if (!mounted) return
    pollTimer = setTimeout(async () => {
      await fetchReview()
      // Keep polling until a final state (also after a brief request failure)
      if (mounted && (!review.value || review.value.status === 'processing')) {
        schedulePoll()
      }
    }, 5_000)
  }
  schedulePoll()
})
onUnmounted(() => {
  mounted = false
  stopPoll()
})
</script>

<style scoped>
.resume-report { max-width: 1100px; }
.total-score { font-size: 56px; font-weight: 700; color: #1677ff; line-height: 1; }
.total-label { font-size: 14px; color: #8c8c8c; margin: 4px 0 8px; }
.total-comment { font-size: 14px; color: #595959; max-width: 600px; margin: 0 auto; }
.summary-section { margin-bottom: 12px; }
.summary-title { font-weight: 500; margin-bottom: 6px; }
.summary-section ul { padding-left: 20px; margin: 0; font-size: 14px; line-height: 1.8; }
.fit-assessment { margin-top: 12px; font-size: 14px; color: #595959; }
</style>

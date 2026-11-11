<template>
  <div class="exam-result">
    <el-page-header @back="router.push('/exam')" title="Back" content="Grading result" />

    <div v-if="loading" class="loading-state">
      <el-skeleton :rows="8" animated />
    </div>

    <template v-else-if="review">
      <!-- Status -->
      <el-alert
        :title="statusTitle"
        :type="statusAlertType"
        :closable="false"
        style="margin: 16px 0"
      />

      <!-- Total score (the student sees the result once the teacher publishes it) -->
      <el-row v-if="review.pre_review_summary" :gutter="16" style="margin-bottom: 16px">
        <el-col :span="6">
          <el-card class="score-card">
            <div class="score-num">
              {{ review.pre_review_summary?.total_score ?? '--' }}
              <span class="score-full">/ {{ review.pre_review_summary?.full_score ?? '--' }}</span>
            </div>
            <div class="score-label">{{ review.status === 'published' ? 'Final score' : 'AI pre-grade' }}</div>
          </el-card>
        </el-col>
        <el-col :span="18" v-if="review.weak_points?.length">
          <el-card>
            <div class="weak-title">Topics to review</div>
            <div class="weak-tags">
              <el-tag
                v-for="wp in review.weak_points"
                :key="wp.tag"
                type="danger"
                size="small"
                style="margin: 4px"
              >
                {{ wp.tag }} ({{ wp.wrong_count }} of {{ wp.total_count ?? wp.wrong_count }} wrong)
              </el-tag>
            </div>
            <div v-if="review.weak_points_summary" class="weak-summary">
              {{ review.weak_points_summary }}
            </div>
          </el-card>
        </el-col>
      </el-row>

      <!-- Per question -->
      <el-card v-if="review.pre_review_summary">
        <template #header>Question by question</template>
        <el-collapse>
          <el-collapse-item
            v-for="q in review.pre_review_summary?.by_question"
            :key="q.question_id"
            :name="q.question_id"
          >
            <template #title>
              <div class="q-title">
                <span>Question {{ q.question_no }} ({{ QUESTION_TYPE_LABELS[q.question_type] ?? q.question_type }})</span>
                <el-tag
                  :type="scoreTagType(q.final_score ?? q.score, q.full_score)"
                  size="small"
                  style="margin-left: 8px"
                >
                  {{ q.final_score ?? q.score }} / {{ q.full_score }} points
                </el-tag>
              </div>
            </template>

            <div class="q-detail">
              <div class="q-row"><b>Your answer: </b>{{ q.student_answer }}</div>
              <div v-if="q.correct_answer" class="q-row"><b>Reference answer: </b>{{ q.correct_answer }}</div>
              <div class="q-row"><b>AI feedback: </b>{{ q.ai_feedback }}</div>
              <div v-if="q.teacher_comment" class="q-row teacher-comment">
                <b>Teacher's comment: </b>{{ q.teacher_comment }}
              </div>
              <div v-if="q.final_score !== undefined" class="q-row">
                <b>Final score: </b>{{ q.final_score }} points
              </div>
            </div>
          </el-collapse-item>
        </el-collapse>
      </el-card>
    </template>

    <el-empty v-else description="No result yet; check back soon" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { examApi, QUESTION_TYPE_LABELS, type ReviewDetail } from '@/api/exam'

const route = useRoute()
const router = useRouter()
const loading = ref(true)
const review = ref<ReviewDetail | null>(null)
let pollTimer: ReturnType<typeof setTimeout> | null = null

const statusTitle = computed(() => {
  if (!review.value) return ''
  if (review.value.status === 'published') return '✅ Your teacher has confirmed and published the result'
  if (review.value.status === 'pending_review') return '⏳ The AI has graded it; waiting for your teacher to confirm'
  return '🔄 The AI is grading your answers...'
})

const statusAlertType = computed(() => {
  if (review.value?.status === 'published') return 'success'
  if (review.value?.status === 'pending_review') return 'warning'
  return 'info'
})

// Green for full marks, orange for partial, red for zero
function scoreTagType(score: number | null | undefined, full: number) {
  if (!score) return 'danger'
  return score >= full ? 'success' : 'warning'
}

async function fetchReview() {
  try {
    const { data } = await examApi.getSubmissionReview(route.params.submissionId as string)
    review.value = data
    // Stop polling once there is a result
    if (data.status === 'published' || data.pre_review_summary) {
      stopPoll()
    }
  } catch {
    stopPoll()
  } finally {
    loading.value = false
  }
}

function stopPoll() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null }
}

onMounted(() => {
  fetchReview()
  // A chained setTimeout instead of setInterval, so requests cannot pile up
  const schedulePoll = () => {
    pollTimer = setTimeout(async () => {
      await fetchReview()
      // Keep polling only while it is still in progress
      if (review.value?.status !== 'published' && !review.value?.pre_review_summary) {
        schedulePoll()
      }
    }, 8_000)
  }
  schedulePoll()
})

onUnmounted(stopPoll)
</script>

<style scoped>
.exam-result { max-width: 900px; }
.loading-state { margin-top: 24px; }
.score-card { text-align: center; padding: 8px 0; }
.score-num { font-size: 36px; font-weight: 700; color: #1677ff; }
.score-full { font-size: 18px; color: #8c8c8c; }
.score-label { font-size: 13px; color: #8c8c8c; margin-top: 4px; }
.weak-title { font-weight: 500; margin-bottom: 8px; }
.weak-tags { margin-bottom: 8px; }
.weak-summary { font-size: 13px; color: #595959; }
.q-title { display: flex; align-items: center; }
.q-detail { padding: 8px 0; }
.q-row { margin-bottom: 8px; font-size: 14px; line-height: 1.6; }
.teacher-comment { color: #d46b08; background: #fff7e6; padding: 6px 10px; border-radius: 4px; }
</style>

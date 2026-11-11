<template>
  <div class="exam-review">
    <el-card>
      <template #header>
        <span>Submissions waiting for review</span>
        <el-button style="float: right" :icon="Refresh" circle size="small" @click="fetchList" />
      </template>

      <el-table :data="list" v-loading="loading" size="default">
        <el-table-column prop="student_name" label="Student" width="120" />
        <el-table-column prop="exam_title" label="Exam" />
        <el-table-column label="Submitted" width="180">
          <template #default="{ row }">{{ formatTime(row.submitted_at) }}</template>
        </el-table-column>
        <el-table-column label="AI pre-grade" width="120">
          <template #default="{ row }">
            {{ row.pre_review?.total_score }} / {{ row.pre_review?.full_score }}
          </template>
        </el-table-column>
        <el-table-column label="To check" width="100">
          <template #default="{ row }">
            <el-tag type="warning" size="small">{{ row.pre_review?.needs_review_count }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="Action" width="100">
          <template #default="{ row }">
            <el-button type="primary" text size="small" @click="openReview(row.submission_id)">
              Review
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- Review drawer -->
    <el-drawer v-model="drawerVisible" title="Review grading" size="65%" direction="rtl" destroy-on-close>
      <div v-if="currentReview" class="review-drawer">

        <!-- Summary -->
        <el-descriptions :column="3" border size="small" style="margin-bottom: 16px">
          <el-descriptions-item label="AI pre-grade">
            <b>{{ currentReview.pre_review_summary?.total_score }}</b>
            / {{ currentReview.pre_review_summary?.full_score }} points
          </el-descriptions-item>
          <el-descriptions-item label="To check">
            <el-tag type="warning" size="small">
              {{ currentReview.pre_review_summary?.by_question?.filter(q => q.needs_review).length }} questions
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="Weak topics">
            <span v-if="!currentReview.weak_points?.length" style="color:#999">—</span>
            <el-tag
              v-for="wp in currentReview.weak_points?.slice(0, 3)"
              :key="wp.tag"
              type="danger"
              size="small"
              style="margin-right: 4px"
            >
              {{ wp.tag }}
            </el-tag>
          </el-descriptions-item>
        </el-descriptions>

        <!-- Questions -->
        <el-collapse v-model="expandedKeys">
          <el-collapse-item
            v-for="q in currentReview.pre_review_summary?.by_question"
            :key="q.question_id"
            :name="q.question_id"
          >
            <template #title>
              <div class="q-title-row">
                <span class="q-no">Question {{ q.question_no }}</span>
                <el-tag :type="typeTagType(q.question_type)" size="small" style="margin-left: 6px">
                  {{ typeLabel(q.question_type) }}
                </el-tag>
                <el-tag
                  :type="scoreChanged(q) ? 'danger' : (q.needs_review ? 'warning' : 'success')"
                  size="small"
                  style="margin-left: 6px"
                >
                  {{ scoreChanged(q) ? 'Changed to ' + modifications[q.question_id]?.new_score : q.score }}
                  / {{ q.full_score }} points
                  <span v-if="q.needs_review && !scoreChanged(q)">· check</span>
                </el-tag>
                <el-tag v-if="q.grader_instructions_found" type="danger" size="small" effect="plain" style="margin-left: 6px">
                  Asks the grader for marks
                </el-tag>
                <el-tag v-if="q.fallback" type="danger" size="small" effect="plain" style="margin-left: 6px">
                  AI grading failed: grade by hand
                </el-tag>
              </div>
            </template>

            <div class="q-detail">
              <!-- Question -->
              <div v-if="q.content" class="q-section">
                <div class="q-label">Question</div>
                <div class="q-content">{{ q.content }}</div>
              </div>

              <!-- Student answer and reference -->
              <div class="q-row">
                <div class="q-col">
                  <div class="q-label">Student answer</div>
                  <div class="q-answer">{{ q.student_answer || '(no answer)' }}</div>
                </div>
                <div v-if="q.correct_answer" class="q-col">
                  <div class="q-label">Reference answer</div>
                  <div class="q-answer correct">{{ q.correct_answer }}</div>
                </div>
              </div>

              <!-- AI feedback (for code questions it repeats the code review list below) -->
              <div v-if="!q.quality_feedback?.length" class="q-section">
                <div class="q-label">AI feedback</div>
                <div class="q-feedback">{{ q.ai_feedback }}</div>
              </div>

              <!-- Scoring points (short answer) -->
              <div v-if="q.point_results?.length" class="q-section">
                <div class="q-label">Scoring points</div>
                <div v-for="(pt, i) in q.point_results" :key="i" class="point-row">
                  <el-icon :color="pt.earned ? '#67c23a' : '#f56c6c'">
                    <component :is="pt.earned ? 'CircleCheck' : 'CircleClose'" />
                  </el-icon>
                  <span style="margin-left: 4px">{{ pt.point_desc }} ({{ pt.earned_score ?? (pt.earned ? pt.point_score : 0) }}/{{ pt.point_score }})</span>
                  <span v-if="!pt.earned && pt.missing" style="color:#f56c6c; margin-left: 4px">— {{ pt.missing }}</span>
                </div>
              </div>

              <!-- Code feedback -->
              <div v-if="q.quality_feedback?.length" class="q-section">
                <div class="q-label">Code review</div>
                <ul class="q-feedback-list">
                  <li v-for="(item, i) in q.quality_feedback" :key="i">{{ item }}</li>
                </ul>
              </div>

              <!-- Teacher's score -->
              <div class="q-modify-row">
                <span class="q-label" style="min-width: 64px">Your score</span>
                <el-input-number
                  v-model="modifications[q.question_id].new_score"
                  :min="0"
                  :max="q.full_score"
                  size="small"
                  style="width: 110px"
                  @change="onScoreChange(q)"
                />
                <span style="font-size:12px; color:#909399; margin: 0 4px">/ {{ q.full_score }}</span>
                <el-input
                  v-model="modifications[q.question_id].comment"
                  placeholder="Comment (optional)"
                  size="small"
                  style="flex: 1; min-width: 180px; max-width: 320px"
                />
              </div>
            </div>
          </el-collapse-item>
        </el-collapse>

        <!-- Actions -->
        <div class="action-bar">
          <div style="font-size:13px; color:#606266">
            <span v-if="changedCount === 0">No changes to the AI grades; confirm to publish them.</span>
            <span v-else>You changed <b>{{ changedCount }}</b> scores; your scores are published.</span>
          </div>
          <div style="display:flex; gap:10px; margin-top:10px">
            <el-button @click="drawerVisible = false" :disabled="confirming">Cancel</el-button>
            <el-button type="primary" :loading="confirming" @click="confirmReview">
              {{ changedCount > 0 ? 'Publish with changes' : 'Confirm and publish' }}
            </el-button>
          </div>
        </div>
      </div>
    </el-drawer>
  </div>
</template>

<script setup lang="ts">
import { formatTime } from '@/utils/format'
import { ref, reactive, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh, CircleCheck, CircleClose } from '@element-plus/icons-vue'
import { examApi, type PendingReviewItem, type ReviewDetail } from '@/api/exam'

const loading = ref(false)
const list = ref<PendingReviewItem[]>([])
const drawerVisible = ref(false)
const currentReview = ref<ReviewDetail | null>(null)
const confirming = ref(false)
const expandedKeys = ref<string[]>([])

// Per question: { new_score, comment }, starting from the AI score
const modifications = reactive<Record<string, { new_score: number; comment: string }>>({})

// Questions whose score differs from the AI score
const changedCount = computed(() => {
  if (!currentReview.value) return 0
  return (currentReview.value.pre_review_summary?.by_question ?? []).filter(q => scoreChanged(q)).length
})

function scoreChanged(q: { question_id: string; score: number }) {
  const mod = modifications[q.question_id]
  return mod !== undefined && mod.new_score !== q.score
}

function onScoreChange(_q: { question_id: string; score: number }) {
  // changedCount is computed; nothing else to do yet
}

function typeLabel(type: string) {
  const map: Record<string, string> = {
    single_choice: 'Single choice',
    multi_choice:  'Multiple choice',
    judge:         'True/false',
    short_answer:  'Short answer',
    code:          'Code',
  }
  return map[type] ?? type
}

function typeTagType(type: string) {
  const map: Record<string, string> = {
    single_choice: '',
    multi_choice:  '',
    judge:         '',
    short_answer:  'warning',
    code:          'danger',
  }
  return (map[type] ?? '') as '' | 'success' | 'warning' | 'danger' | 'info'
}

async function fetchList() {
  loading.value = true
  try {
    const { data } = await examApi.getPendingReviews()
    list.value = data.items
  } finally {
    loading.value = false
  }
}

async function openReview(submissionId: string) {
  const { data } = await examApi.getSubmissionReviewTeacher(submissionId)
  currentReview.value = data

  // Start the score table from the AI scores
  for (const q of data.pre_review_summary?.by_question ?? []) {
    modifications[q.question_id] = { new_score: q.score, comment: '' }
  }

  // Open the questions that need checking
  expandedKeys.value = (data.pre_review_summary?.by_question ?? [])
    .filter(q => q.needs_review)
    .map(q => q.question_id)

  drawerVisible.value = true
}

async function confirmReview() {
  if (!currentReview.value) return
  confirming.value = true
  try {
    const questions = currentReview.value.pre_review_summary?.by_question ?? []

    // Send only the changed questions
    const changedMods = questions
      .filter(q => scoreChanged(q))
      .map(q => ({
        question_id: q.question_id,
        new_score:   modifications[q.question_id].new_score,
        comment:     modifications[q.question_id].comment || undefined,
      }))

    const action = changedMods.length > 0 ? 'modify' : 'approve'

    await examApi.confirmReview(currentReview.value.submission_id, {
      action,
      modifications: changedMods,
    })

    ElMessage.success('Published')
    drawerVisible.value = false
    fetchList()
  } catch {
    // The client interceptor shows the error (e.g. 409 if it was already published)
  } finally {
    confirming.value = false
  }
}

onMounted(fetchList)
</script>

<style scoped>
.exam-review { max-width: 1100px; }

.q-title-row {
  display: flex;
  align-items: center;
  gap: 0;
}
.q-no { font-weight: 600; }

.q-detail {
  padding: 4px 0 8px 0;
  font-size: 14px;
  line-height: 1.8;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.q-section { display: flex; flex-direction: column; gap: 2px; }

.q-row {
  display: flex;
  gap: 24px;
}
.q-col { flex: 1; display: flex; flex-direction: column; gap: 2px; }

.q-label {
  font-size: 12px;
  color: #909399;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.q-content {
  background: #f5f7fa;
  border-radius: 4px;
  padding: 6px 10px;
  white-space: pre-wrap;
  word-break: break-all;
}

.q-answer {
  background: #f5f7fa;
  border-radius: 4px;
  padding: 6px 10px;
  white-space: pre-wrap;
  word-break: break-all;
  min-height: 32px;
}
.q-answer.correct {
  background: #f0f9eb;
  color: #529b2e;
}

.q-feedback {
  color: #606266;
  padding: 4px 0;
  white-space: pre-wrap;
}
.q-feedback-list {
  margin: 0;
  padding-left: 18px;
  color: #606266;
}

.point-row {
  display: flex;
  align-items: center;
  font-size: 13px;
  padding: 2px 0;
}

.q-modify-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  padding: 8px 10px;
  background: #fafafa;
  border-radius: 4px;
  border: 1px solid #e4e7ed;
}

.action-bar {
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid #e4e7ed;
}
</style>

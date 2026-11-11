<template>
  <div class="exam-submit">
    <el-card>
      <template #header>
        <span>📝 Submit an exam</span>
      </template>

      <el-form :model="form" label-width="100px" style="max-width: 560px">
        <el-form-item label="Exam">
          <el-select v-model="form.examId" placeholder="Choose an exam" style="width: 100%">
            <el-option
              v-for="exam in exams"
              :key="exam.exam_id"
              :label="`${exam.title} (${exam.question_count} questions, ${exam.full_score} points)`"
              :value="exam.exam_id"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="Answer sheet">
          <el-upload
            ref="uploadRef"
            :auto-upload="false"
            :limit="1"
            accept=".docx"
            :on-change="handleFileChange"
            :on-remove="() => (form.file = null)"
          >
            <el-button :icon="Upload">Choose file</el-button>
            <template #tip>
              <div class="upload-tip">Word (.docx) only, up to 5 MB. Write "Question N" before each answer.</div>
            </template>
          </el-upload>
        </el-form-item>
        <el-form-item>
          <el-button
            type="primary"
            :loading="loading"
            :disabled="!form.examId || !form.file"
            @click="handleSubmit"
          >
            Submit for grading
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- Earlier submissions -->
    <el-card style="margin-top: 16px">
      <template #header>
        <span>My submissions</span>
      </template>
      <el-empty v-if="!recentSubmissions.length" description="No submissions yet" />
      <el-table v-else :data="recentSubmissions" size="small">
        <el-table-column prop="exam_title" label="Exam" />
        <el-table-column label="Submitted" width="180">
          <template #default="{ row }">{{ formatTime(row.submitted_at) }}</template>
        </el-table-column>
        <el-table-column label="Status" width="140">
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)" size="small">{{ statusLabel(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="Action" width="100">
          <template #default="{ row }">
            <el-button
              text
              type="primary"
              size="small"
              @click="router.push(`/exam/${row.submission_id}`)"
            >
              View result
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
import { ElMessage, type UploadFile } from 'element-plus'
import { Upload } from '@element-plus/icons-vue'
import { examApi, type ExamItem, type MySubmissionItem } from '@/api/exam'

const router = useRouter()
const loading = ref(false)
const form = reactive({ examId: '', file: null as File | null })
const recentSubmissions = ref<MySubmissionItem[]>([])
const exams = ref<ExamItem[]>([])

async function fetchExams() {
  try {
    const { data } = await examApi.listExams()
    exams.value = data.items
    if (data.items.length === 1) form.examId = data.items[0].exam_id
  } catch {
    // The client interceptor shows the error
  }
}

async function fetchSubmissions() {
  try {
    const { data } = await examApi.listMySubmissions()
    recentSubmissions.value = data.items
  } catch {
    // The client interceptor shows the error
  }
}

function handleFileChange(file: UploadFile) {
  form.file = file.raw ?? null
}

function statusType(status: string) {
  if (status === 'published') return 'success'
  if (status === 'pending_review') return 'warning'
  if (status === 'submitted') return 'danger'
  return 'info'
}

function statusLabel(status: string) {
  if (status === 'published') return 'Published'
  if (status === 'pending_review') return 'Waiting for the teacher'
  if (status === 'ai_processing') return 'AI grading'
  if (status === 'submitted') return 'Failed; please submit again'
  return status
}

async function handleSubmit() {
  if (!form.examId || !form.file) return
  loading.value = true
  try {
    await examApi.submit(form.examId, form.file)
    ElMessage.success('Submitted. The AI is grading it...')
    form.file = null
    await fetchSubmissions()
  } catch {
    // The client interceptor shows the error
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  fetchExams()
  fetchSubmissions()
})
</script>

<style scoped>
.exam-submit { max-width: 800px; }
.upload-tip { font-size: 12px; color: #8c8c8c; margin-top: 4px; }
</style>

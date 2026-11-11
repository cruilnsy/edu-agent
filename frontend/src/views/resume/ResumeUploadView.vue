<template>
  <div class="resume-upload">
    <el-card>
      <template #header><span>📄 Upload a resume</span></template>

      <div class="file-picker">
        <el-button type="default" :icon="UploadFilled" @click="triggerFilePicker">
          Click to choose a PDF
        </el-button>
        <input
          ref="fileInputRef"
          type="file"
          accept=".pdf"
          style="display: none"
          @change="handleFileChange"
        />
        <span v-if="selectedFile" class="file-name">{{ selectedFile.name }}</span>
        <el-button v-if="selectedFile" text type="danger" size="small" @click="clearFile">
          Remove
        </el-button>
      </div>
      <div class="upload-tip">PDF only, up to 10 MB</div>

      <el-button
        type="primary"
        :loading="loading"
        :disabled="!selectedFile"
        style="margin-top: 16px"
        @click="handleUpload"
      >
        Start review
      </el-button>
    </el-card>

    <!-- Earlier reviews -->
    <el-card style="margin-top: 16px">
      <template #header>
        <span>My reviews</span>
        <el-button style="float: right" :icon="Refresh" circle size="small" @click="fetchHistory" />
      </template>
      <el-empty v-if="!history.length" description="No reviews yet" />
      <el-table v-else :data="history" size="small">
        <el-table-column prop="review_id" label="Review ID" width="220" />
        <el-table-column label="Created" width="180">
          <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="Score" width="100">
          <template #default="{ row }">
            <span v-if="row.weighted_score !== undefined">
              {{ row.weighted_score.toFixed(1) }}
            </span>
            <el-tag v-else type="info" size="small">Processing</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="Action" width="160">
          <template #default="{ row }">
            <el-button text type="primary" size="small" @click="router.push(`/resume/${row.review_id}`)">
              View report
            </el-button>
            <el-button text type="danger" size="small" @click="handleDelete(row.review_id)">
              Delete
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { formatTime } from '@/utils/format'
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { UploadFilled, Refresh } from '@element-plus/icons-vue'
import { resumeApi, type ReviewListItem } from '@/api/resume'

const router = useRouter()
const loading = ref(false)
const selectedFile = ref<File | null>(null)
const fileInputRef = ref<HTMLInputElement | null>(null)
const history = ref<ReviewListItem[]>([])

function triggerFilePicker() {
  fileInputRef.value?.click()
}

function handleFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0] ?? null
  if (file && file.size > 10 * 1024 * 1024) {
    ElMessage.error('The file is larger than 10 MB')
    input.value = ''
    return
  }
  selectedFile.value = file
}

function clearFile() {
  selectedFile.value = null
  if (fileInputRef.value) fileInputRef.value.value = ''
}

async function handleUpload() {
  if (!selectedFile.value) return
  loading.value = true
  try {
    const { data } = await resumeApi.upload(selectedFile.value)
    ElMessage.success('Uploaded. The AI is reviewing it...')
    router.push(`/resume/${data.review_id}`)
  } finally {
    loading.value = false
  }
}

async function handleDelete(reviewId: string) {
  try {
    await resumeApi.deleteReview(reviewId)
    history.value = history.value.filter(r => r.review_id !== reviewId)
    ElMessage.success('Deleted')
  } catch { /* error handled by client interceptor */ }
}

async function fetchHistory() {
  try {
    const { data } = await resumeApi.listReviews()
    history.value = data.items
  } catch { /* ignore */ }
}

onMounted(fetchHistory)
</script>

<style scoped>
.resume-upload { max-width: 800px; }
.file-picker { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.file-name { font-size: 13px; color: #595959; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.upload-tip { font-size: 12px; color: #8c8c8c; margin-top: 8px; }
</style>

<template>
  <div class="interview-chat">
    <!-- Stage progress -->
    <el-card style="margin-bottom: 16px; padding: 0">
      <StageProgressBar :current="currentStage" :reached="stagesReached" />
    </el-card>

    <!-- Report (after the interview) -->
    <template v-if="isFinished && report">
      <el-card class="report-card">
        <template #header>
          <span>🎉 Interview report — {{ report.target_position }}</span>
        </template>

        <el-row :gutter="16" style="margin-bottom: 16px">
          <el-col :span="6" style="text-align: center">
            <div class="overall-score">{{ report.overall_score ?? '—' }}</div>
            <div style="color: #8c8c8c; font-size: 13px">Overall score (out of 100)</div>
          </el-col>
          <el-col :span="18">
            <div class="report-comment">{{ report.overall_comment }}</div>
          </el-col>
        </el-row>

        <el-row :gutter="16" style="margin-bottom: 16px">
          <el-col :span="12">
            <div class="report-section-title">✅ Strengths</div>
            <ul class="report-list">
              <li v-for="(s, i) in report.strengths" :key="i">{{ s }}</li>
            </ul>
          </el-col>
          <el-col :span="12">
            <div class="report-section-title">🔧 To improve</div>
            <ul class="report-list">
              <li v-for="(s, i) in report.improvements" :key="i">{{ s }}</li>
            </ul>
          </el-col>
        </el-row>

        <el-row :gutter="16">
          <el-col :span="12">
            <div class="report-section-title">📚 Topics to review</div>
            <el-tag
              v-for="(t, i) in report.recommended_topics"
              :key="i"
              style="margin: 4px"
            >{{ t }}</el-tag>
          </el-col>
          <el-col :span="12">
            <div class="report-section-title">🚀 Next step</div>
            <div style="font-size: 14px; color: #595959">{{ report.next_step_advice }}</div>
          </el-col>
        </el-row>

        <el-divider />
        <div class="report-section-title" style="margin-bottom: 12px">Scores by dimension</div>
        <el-row :gutter="12">
          <el-col :span="8" v-for="dim in report.dimensions" :key="dim.dimension" style="margin-bottom: 12px">
            <el-card shadow="never" style="border: 1px solid #f0f0f0">
              <div style="display: flex; justify-content: space-between; align-items: center">
                <span style="font-size: 13px">{{ dim.dimension }}</span>
                <span style="font-size: 18px; font-weight: 700; color: #1677ff">{{ dim.score }}</span>
              </div>
              <el-progress
                :percentage="dim.score"
                :show-text="false"
                :stroke-width="6"
                style="margin-top: 6px"
              />
              <div style="font-size: 12px; color: #8c8c8c; margin-top: 4px">{{ dim.comment }}</div>
            </el-card>
          </el-col>
        </el-row>

        <div style="margin-top: 16px">
          <el-button @click="router.push('/interview')">Back to interviews</el-button>
        </div>
      </el-card>
    </template>

    <!-- Chat (during the interview) -->
    <template v-else>
      <el-card class="chat-card">
        <div class="chat-messages" ref="messagesEl">
          <ChatBubble
            v-for="(msg, i) in messages"
            :key="i"
            :role="msg.role"
          >
            <MarkdownRenderer :content="msg.content" />
          </ChatBubble>
          <!-- "Thinking" until the first token arrives -->
          <ChatBubble v-if="loading && !streamingReply" role="assistant">
            <span style="color: #8c8c8c">Thinking...</span>
          </ChatBubble>
        </div>

        <div class="chat-input-area">
          <el-input
            v-model="inputText"
            type="textarea"
            :rows="3"
            placeholder="Type your answer. Ctrl+Enter to send"
            resize="none"
            :disabled="loading || isFinished"
            @keydown.ctrl.enter="sendMessage"
          />

          <div class="input-actions">
            <!-- Left: turn -->
            <div style="display: flex; align-items: center; gap: 12px">
              <span style="font-size: 12px; color: #8c8c8c">Turn {{ totalTurns }}</span>
            </div>

            <!-- Right: end and send -->
            <div style="display: flex; gap: 8px; align-items: center">
              <el-button
                type="danger"
                plain
                size="small"
                :disabled="loading || isFinished"
                @click="endInterview"
              >
                End interview
              </el-button>
              <el-button
                type="primary"
                :loading="loading"
                :disabled="!inputText.trim() || isFinished"
                @click="sendMessage"
              >
                Send
              </el-button>
            </div>
          </div>
        </div>
      </el-card>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import ChatBubble from '@/components/chat/ChatBubble.vue'
import MarkdownRenderer from '@/components/chat/MarkdownRenderer.vue'
import StageProgressBar from '@/components/interview/StageProgressBar.vue'
import { interviewApi, type InterviewReport } from '@/api/interview'
import client from '@/api/client'

interface Message { role: 'user' | 'assistant'; content: string }

const route = useRoute()
const router = useRouter()
const sessionId = route.params.sessionId as string

// ── Chat state ──
const messages = ref<Message[]>([])
const inputText = ref('')
const loading = ref(false)
const streamingReply = ref(false)  // Tokens are arriving (as opposed to waiting for the first)
const currentStage = ref('warmup')
const stagesReached = ref<string[]>([])
const totalTurns = ref(0)
const isFinished = ref(false)
const report = ref<InterviewReport | null>(null)
const messagesEl = ref<HTMLElement>()


// ── Send (streamed) ──
async function sendMessage() {
  const text = inputText.value.trim()
  if (!text || loading.value) return

  messages.value.push({ role: 'user', content: text })
  inputText.value = ''
  loading.value = true
  await scrollToBottom()

  // The reply bubble is added on the first token (or the done event), so it never shows
  // next to the "Thinking..." bubble
  let streamBubbleIndex = -1  // -1: no reply bubble yet

  return new Promise<void>((resolve) => {
    interviewApi.chatStream(sessionId, text, {
      onToken: async (chunk) => {
        // First token: add the bubble
        if (streamBubbleIndex === -1) {
          messages.value.push({ role: 'assistant', content: '' })
          streamBubbleIndex = messages.value.length - 1
          streamingReply.value = true
        }
        messages.value[streamBubbleIndex].content += chunk
        await scrollToBottom()
      },
      onDone: async (meta) => {
        streamingReply.value = false
        currentStage.value = meta.current_stage
        stagesReached.value = meta.stages_reached ?? []
        totalTurns.value = meta.total_turns

        if (streamBubbleIndex === -1) {
          // No tokens arrived (e.g. a fixed reply): the done event carries the whole reply
          if (meta.reply) {
            messages.value.push({ role: 'assistant', content: meta.reply })
            streamBubbleIndex = messages.value.length - 1
          }
          // No reply at all: no bubble
        } else if (!messages.value[streamBubbleIndex]?.content && meta.reply) {
          // A bubble but no text: use the reply
          messages.value[streamBubbleIndex].content = meta.reply
        }


        if (meta.is_finished) {
          isFinished.value = true
          await fetchReport()
        }

        loading.value = false
        await scrollToBottom()
        resolve()
      },
      onError: async (err) => {
        streamingReply.value = false
        // Remove an empty bubble
        if (streamBubbleIndex !== -1 && !messages.value[streamBubbleIndex]?.content) {
          messages.value.splice(streamBubbleIndex, 1)
        }
        ElMessage.error(err.message || 'Sending failed; please try again')
        loading.value = false
        await scrollToBottom()
        resolve()
      },
    })
  })
}

async function endInterview() {
  inputText.value = "I'd like to end the interview here."
  await sendMessage()
}

async function fetchReport() {
  try {
    const res = await client.get(`/interview/sessions/${sessionId}/report`, {
      validateStatus: (s: number) => s < 500,
    })
    if (res.status === 200) report.value = res.data
  } catch { /* ignore */ }
}

async function scrollToBottom() {
  await nextTick()
  if (messagesEl.value) {
    messagesEl.value.scrollTop = messagesEl.value.scrollHeight
  }
}

// ── Lifecycle ──
onMounted(async () => {
  const openingMessage = (history.state as any)?.openingMessage
  if (openingMessage) {
    messages.value.push({ role: 'assistant', content: openingMessage })
  }

  try {
    const res = await client.get(`/interview/sessions/${sessionId}/report`, {
      validateStatus: (s: number) => s < 500,
    })
    if (res.status === 200) {       // The report exists only for a finished interview
      isFinished.value = true
      report.value = res.data
      currentStage.value = 'finished'
      stagesReached.value = res.data.stages_reached ?? []
    }
  } catch { /* ignore */ }
})

</script>

<style scoped>
.interview-chat { max-width: 900px; }
.chat-card { display: flex; flex-direction: column; }
.chat-messages {
  height: calc(100vh - 56px - 48px - 120px - 200px);
  overflow-y: auto;
  padding: 8px 0;
}
.chat-input-area { border-top: 1px solid #f0f0f0; padding-top: 12px; margin-top: 12px; }
.input-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 8px;
}
.overall-score { font-size: 52px; font-weight: 700; color: #1677ff; line-height: 1; }
.report-comment { font-size: 14px; color: #595959; line-height: 1.8; }
.report-section-title { font-weight: 500; margin-bottom: 8px; }
.report-list { padding-left: 20px; margin: 0; font-size: 14px; line-height: 1.8; }
.report-card { margin-bottom: 16px; }


</style>

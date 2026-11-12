<template>
  <div class="chat-page">
    <div class="chat-panel">
      <!-- Hint -->
      <div class="chat-header-hint">
        <el-icon><MagicStick /></el-icon>
        The assistant works out what you need and sends you to the right agent
      </div>

      <!-- Messages -->
      <div class="chat-messages" ref="messagesEl">
        <div v-if="messages.length === 0 && !isStreaming" class="empty-hint">
          <p>✨ Hi! I'm the EduAgent assistant</p>
          <p>Tell me what you need, for example:</p>
          <div class="example-queries">
            <el-tag
              v-for="q in exampleQueries"
              :key="q"
              class="example-tag"
              @click="sendExample(q)"
            >{{ q }}</el-tag>
          </div>
        </div>

        <template v-for="(msg, i) in messages" :key="i">
          <!-- Routing card (assistant messages only) -->
          <RoutingDecisionCard
            v-if="msg.routingDecision"
            :agent-type="msg.routingDecision.agent_type"
            :agent-display="msg.routingDecision.agent_display"
            :confidence="msg.routingDecision.confidence"
            :reason="msg.routingDecision.reason"
            :execution-mode="msg.routingDecision.execution_mode"
          />

          <!-- Job preparation plan card -->
          <PipelinePlanCard
            v-if="msg.pipelinePlan"
            :title="msg.pipelinePlan.title"
            :intro="msg.pipelinePlan.intro"
            :steps="msg.pipelinePlan.steps"
          />

          <!-- Normal bubble (a plan card needs none) -->
          <ChatBubble v-if="!msg.pipelinePlan" :role="msg.role" :sources="msg.sources">
            <!-- Guidance with a link button -->
            <template v-if="msg.guidance">
              <p style="margin: 0 0 10px">{{ msg.guidance.message }}</p>
              <el-button
                v-if="msg.guidance.action_label"
                type="primary"
                size="small"
                @click="router.push(msg.guidance.action_url)"
              >
                {{ msg.guidance.action_label }} →
              </el-button>
            </template>
            <MarkdownRenderer v-else :content="msg.content" />
          </ChatBubble>
        </template>

        <!-- Streaming bubble -->
        <template v-if="isStreaming">
          <RoutingDecisionCard
            v-if="streamingRouting"
            :agent-type="streamingRouting.agent_type"
            :agent-display="streamingRouting.agent_display"
            :confidence="streamingRouting.confidence"
            :reason="streamingRouting.reason"
            :execution-mode="streamingRouting.execution_mode"
          />
          <ChatBubble role="assistant">
            <template v-if="streamingText">
              <MarkdownRenderer :content="streamingText" />
            </template>
            <span v-else-if="progressStage" class="progress-hint">
              <span class="dot" /><span class="dot" /><span class="dot" />
              {{ progressStage }}
            </span>
            <span v-else class="thinking">
              <span class="dot" /><span class="dot" /><span class="dot" />
            </span>
          </ChatBubble>
        </template>
      </div>

      <!-- Input -->
      <div class="chat-input-area">
        <el-input
          ref="inputRef"
          v-model="inputText"
          type="textarea"
          :rows="3"
          placeholder="Describe what you need. Enter to send, Shift+Enter for a new line"
          resize="none"
          :disabled="isStreaming"
          @keydown="handleKeydown"
        />
        <div class="input-actions">
          <el-tag v-if="lastAnswerMode" size="small" :type="answerModeTagType">
            {{ answerModeLabel }}
          </el-tag>
          <el-button
            type="primary"
            :loading="isStreaming"
            :disabled="!inputText.trim() || isStreaming"
            @click="sendMessage"
          >
            Send
          </el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { MagicStick } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import ChatBubble from '@/components/chat/ChatBubble.vue'
import MarkdownRenderer from '@/components/chat/MarkdownRenderer.vue'
import RoutingDecisionCard from '@/components/chat/RoutingDecisionCard.vue'
import PipelinePlanCard from '@/components/chat/PipelinePlanCard.vue'
import { useAuthStore } from '@/stores/auth'
import { API_BASE, errorText } from '@/api/client'
import { DEMO, DEMO_NOTICE } from '@/demo'

const router = useRouter()
const auth = useAuthStore()

interface RoutingDecision {
  agent_type: string
  agent_display: string
  confidence: number
  reason: string
  execution_mode: string
}

interface GuidanceInfo {
  message: string
  action_label: string
  action_url: string
}

interface PipelineStep {
  step: number
  agent_type: string
  label: string
  desc: string
  action_label: string
  action_url: string
  tip: string
}

interface PipelinePlan {
  title: string
  intro: string
  steps: PipelineStep[]
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
  routingDecision?: RoutingDecision
  guidance?: GuidanceInfo
  pipelinePlan?: PipelinePlan
}

const messages = ref<Message[]>([])
const inputText = ref('')
const messagesEl = ref<HTMLElement>()
// One fixed session per user, so the checkpointer does not collect a new thread every visit
const sessionId = ref(`unified_${auth.user?.userId ?? 'guest'}`)

const isStreaming = ref(false)
const streamingText = ref('')
const streamingRouting = ref<RoutingDecision | null>(null)
const progressStage = ref('')
const lastAnswerMode = ref('')
const lastConfidence = ref(0)
const lastSources = ref<string[]>([])

const exampleQueries = [
  'What does a CORS preflight request check?',
  'I want to submit my exam for grading',
  'Can you review my resume?',
  'I want to do a mock interview',
  'Help me get ready for job hunting',
]

const answerModeTagType = computed(() => {
  if (lastAnswerMode.value === 'rag') return 'success'
  if (lastAnswerMode.value === 'general') return 'warning'
  return 'info'
})

const answerModeLabel = computed(() => {
  const map: Record<string, string> = {
    rag: `RAG ${lastConfidence.value}%`,
    web_augmented: '🌐 Web-augmented',
    general: 'General answer',
    llm_direct: 'LLM answer',
  }
  return map[lastAnswerMode.value] ?? lastAnswerMode.value
})

function sendExample(q: string) {
  inputText.value = q
  sendMessage()
}

function handleKeydown(e: KeyboardEvent) {
  if (e.isComposing) return
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
}

async function sendMessage() {
  const text = inputText.value.trim()
  if (!text || isStreaming.value) return
  if (DEMO) { ElMessage.info(DEMO_NOTICE); return }      // Answers need the backend

  inputText.value = ''
  messages.value.push({ role: 'user', content: text })
  await scrollToBottom()

  isStreaming.value = true
  streamingText.value = ''
  streamingRouting.value = null
  progressStage.value = ''
  lastAnswerMode.value = ''
  lastSources.value = []

  let pendingRouting: RoutingDecision | null = null
  let pendingGuidance: GuidanceInfo | null = null
  let pendingPipelinePlan: PipelinePlan | null = null

  try {
    const resp = await fetch(`${API_BASE}/api/v1/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${auth.token}`,
      },
      body: JSON.stringify({
        session_id: sessionId.value,
        message: text,
      }),
    })

    if (resp.status === 401) {
      auth.logout()              // Clears the store too, so the router guard allows /login
      router.push('/login')
      return
    }
    if (!resp.ok || !resp.body) {
      const body = await resp.json().catch(() => null)   // e.g. 429 with a wait message
      throw new Error(errorText(body?.detail) ?? `HTTP ${resp.status}`)
    }

    const reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    let streamError = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buf += decoder.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data:')) continue
        const raw = line.slice(5).trim()
        if (!raw) continue

        try {
          const evt = JSON.parse(raw)

          if (evt.type === 'routing_decision') {
            pendingRouting = {
              agent_type:     evt.agent_type,
              agent_display:  evt.agent_display,
              confidence:     evt.confidence,
              reason:         evt.reason,
              execution_mode: evt.execution_mode,
            }
            // Show the routing card at once, above the streaming bubble
            streamingRouting.value = pendingRouting
            await scrollToBottom()

          } else if (evt.type === 'progress') {
            progressStage.value = evt.stage

          } else if (evt.type === 'token') {
            progressStage.value = ''
            streamingText.value += evt.content
            await scrollToBottom()

          } else if (evt.type === 'guidance') {
            pendingGuidance = {
              message:      evt.message,
              action_label: evt.action_label ?? '',
              action_url:   evt.action_url ?? '',
            }

          } else if (evt.type === 'pipeline_plan') {
            pendingPipelinePlan = {
              title: evt.title,
              intro: evt.intro,
              steps: evt.steps ?? [],
            }
            await scrollToBottom()

          } else if (evt.type === 'meta') {
            lastAnswerMode.value = evt.answer_mode ?? ''
            lastConfidence.value = Math.round((evt.confidence ?? 0) * 100)
            lastSources.value    = evt.sources ?? []

          } else if (evt.type === 'error') {
            streamError = evt.message ?? 'Streaming failed'
          }
        } catch {
          // Ignore non-JSON lines
        }
      }
    }

    if (streamError) throw new Error(streamError)

    // Stream finished: keep the result as a message
    if (pendingPipelinePlan) {
      messages.value.push({
        role:            'assistant',
        content:         '',
        routingDecision: pendingRouting ?? undefined,
        pipelinePlan:    pendingPipelinePlan,
      })
    } else if (pendingGuidance) {
      messages.value.push({
        role:            'assistant',
        content:         pendingGuidance.message,
        routingDecision: pendingRouting ?? undefined,
        guidance:        pendingGuidance,
      })
    } else if (streamingText.value) {
      messages.value.push({
        role:            'assistant',
        content:         streamingText.value,
        sources:         lastSources.value,
        routingDecision: pendingRouting ?? undefined,
      })
    }

  } catch (err) {
    ElMessage.error(err instanceof Error && err.message ? err.message : 'The request failed; please try again')
    console.error('[UnifiedChat SSE]', err)
  } finally {
    isStreaming.value = false
    streamingText.value = ''
    streamingRouting.value = null
    progressStage.value = ''
    await scrollToBottom()
  }
}

async function scrollToBottom() {
  await nextTick()
  if (messagesEl.value) {
    messagesEl.value.scrollTop = messagesEl.value.scrollHeight
  }
}
</script>

<style scoped>
.chat-page {
  display: flex;
  height: calc(100vh - 56px - 48px);
}
.chat-panel {
  flex: 1;
  background: #fff;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.chat-header-hint {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 20px;
  font-size: 13px;
  color: #8c8c8c;
  border-bottom: 1px solid #f0f0f0;
  background: #fafafa;
}
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}
.empty-hint {
  text-align: center;
  color: #8c8c8c;
  margin-top: 60px;
  font-size: 15px;
  line-height: 2;
}
.example-queries {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
  margin-top: 12px;
}
.example-tag {
  cursor: pointer;
  transition: opacity 0.15s;
}
.example-tag:hover { opacity: 0.75; }
.chat-input-area {
  border-top: 1px solid #f0f0f0;
  padding: 12px 16px;
}
.input-actions {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
}
.thinking, .progress-hint {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 0;
  font-size: 13px;
  color: #8c8c8c;
}
.dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: #bfbfbf;
  animation: bounce 1.2s infinite ease-in-out;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
  0%, 80%, 100% { transform: scale(0.7); opacity: 0.4; }
  40%            { transform: scale(1.1); opacity: 1; }
}
</style>

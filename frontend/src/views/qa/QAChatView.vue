<template>
  <div class="qa-page">
    <!-- Left: sessions -->
    <div class="session-panel">
      <div class="session-header">
        <span>Sessions</span>
        <el-button :icon="Plus" circle size="small" @click="newSession" />
      </div>
      <div class="session-list">
        <div
          v-for="s in sessions"
          :key="s.id"
          class="session-item"
          :class="{ active: s.id === currentSessionId }"
          @click="switchSession(s.id)"
        >
          <el-icon><ChatDotRound /></el-icon>
          <span class="session-name">{{ s.name }}</span>
          <el-icon class="delete-btn" @click.stop="deleteSession(s.id)"><Close /></el-icon>
        </div>
      </div>
    </div>

    <!-- Right: chat -->
    <div class="chat-panel">
      <div class="chat-messages" ref="messagesEl">
        <div v-if="messages.length === 0 && !isStreaming" class="empty-hint">
          <p>🤖 Hi! I'm the EduAgent assistant</p>
          <p>Ask me anything about software or AI engineering</p>
        </div>

        <ChatBubble
          v-for="(msg, i) in messages"
          :key="i"
          :role="msg.role"
          :sources="msg.sources"
        >
          <MarkdownRenderer :content="msg.content" />
        </ChatBubble>

        <!-- Streaming bubble: shown while streaming; progress or dots until the first token -->
        <ChatBubble v-if="isStreaming" role="assistant">
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
      </div>

      <div class="chat-input-area">
        <el-input
          ref="inputRef"
          v-model="inputText"
          type="textarea"
          :rows="3"
          placeholder="Ask a question. Enter to send, Shift+Enter for a new line"
          resize="none"
          :disabled="isStreaming"
          @keydown="handleKeydown"
        />
        <div class="input-actions">
          <el-tooltip
            :content="webSearchEnabled ? 'Web search on: when the knowledge base is not confident, search the web first' : 'Web search off: when the knowledge base is not confident, the LLM answers directly'"
            placement="top"
          >
            <el-switch
              v-model="webSearchEnabled"
              :disabled="isStreaming"
              active-text="🌐"
              inactive-text=""
              size="small"
            />
          </el-tooltip>
          <el-tag
            v-if="lastAnswerMode"
            size="small"
            :type="answerModeTagType"
          >
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
import { ref, computed, nextTick, onMounted } from 'vue'

import { useRouter } from 'vue-router'
import { Plus, ChatDotRound, Close } from '@element-plus/icons-vue'
import { v4 as uuidv4 } from 'uuid'
import { ElMessage } from 'element-plus'
import ChatBubble from '@/components/chat/ChatBubble.vue'
import MarkdownRenderer from '@/components/chat/MarkdownRenderer.vue'
import { useAuthStore } from '@/stores/auth'
import { API_BASE, errorText } from '@/api/client'
import { DEMO, DEMO_NOTICE } from '@/demo'

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
}

interface Session {
  id: string
  name: string
  messages: Message[]
}

const router = useRouter()
const auth = useAuthStore()

const sessions = ref<Session[]>([])
const currentSessionId = ref('')
const messages = ref<Message[]>([])
const inputText = ref('')
const messagesEl = ref<HTMLElement>()

// Streaming state (fetch is called directly to avoid the Vite proxy buffering SSE)
const isStreaming = ref(false)
const streamingText = ref('')
const progressStage = ref('')   // Progress text
const lastAnswerMode = ref('')
const lastConfidence = ref(0)
const lastSources = ref<string[]>([])

// Web search switch (per session, on by default)
const webSearchEnabled = ref(true)

// ── Answer mode tag ──
const answerModeTagType = computed(() => {
  switch (lastAnswerMode.value) {
    case 'rag':           return 'success'
    case 'web_augmented': return 'primary'
    case 'general':       return 'warning'
    default:              return 'info'      // llm_direct
  }
})

const answerModeLabel = computed(() => {
  switch (lastAnswerMode.value) {
    case 'rag':           return `RAG ${lastConfidence.value}%`
    case 'web_augmented': return `🌐 Web-augmented`
    case 'general':       return 'General answer'
    default:              return 'LLM answer'
  }
})

// ── Sessions ──

function newSession() {
  const id = `student_session_${uuidv4()}`
  const session: Session = { id, name: `Session ${sessions.value.length + 1}`, messages: [] }
  sessions.value.unshift(session)
  switchSession(id)
}

function switchSession(id: string) {
  if (isStreaming.value) return
  currentSessionId.value = id
  const s = sessions.value.find(s => s.id === id)
  // messages.value is session.messages itself, so only messages.value is updated
  messages.value = s ? s.messages : []
  scrollToBottom()
}

function deleteSession(id: string) {
  const idx = sessions.value.findIndex(s => s.id === id)
  if (idx === -1) return
  sessions.value.splice(idx, 1)
  if (currentSessionId.value === id) {
    sessions.value.length > 0 ? switchSession(sessions.value[0].id) : newSession()
  }
}

// ── Keys: Enter sends, Shift+Enter adds a line ──

function handleKeydown(e: KeyboardEvent) {
  // Do not send while an IME is composing
  if (e.isComposing) return
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
}

// ── Send (fetch + ReadableStream, bypassing the Vite proxy's buffering) ──

async function sendMessage() {
  const text = inputText.value.trim()
  if (!text || isStreaming.value) return
  if (DEMO) { ElMessage.info(DEMO_NOTICE); return }      // Answers need the backend

  if (!currentSessionId.value) newSession()

  // Clear the input at once so the message cannot be sent twice
  inputText.value = ''

  const session = sessions.value.find(s => s.id === currentSessionId.value)

  // Add the user's message (messages.value is session.messages; push once)
  const userMsg: Message = { role: 'user', content: text }
  messages.value.push(userMsg)
  if (session && session.messages.filter(m => m.role === 'user').length === 1) {
    session.name = text.slice(0, 20) + (text.length > 20 ? '…' : '')
  }

  await scrollToBottom()

  // Start streaming
  isStreaming.value = true
  streamingText.value = ''
  progressStage.value = ''
  lastAnswerMode.value = ''
  lastSources.value = []

  try {
    // Call the backend directly, not through the Vite proxy (it buffers SSE frames)
    const resp = await fetch(`${API_BASE}/api/v1/qa/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${auth.token}`,
      },
      body: JSON.stringify({
        session_id:        currentSessionId.value,
        message:           text,
        enable_web_search: webSearchEnabled.value,
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
          if (evt.type === 'progress' && evt.stage) {
            progressStage.value = evt.stage
          } else if (evt.type === 'token' && evt.content) {
            progressStage.value = ''   // First token: hide the progress text
            streamingText.value += evt.content
            await scrollToBottom()
          } else if (evt.type === 'meta') {
            lastAnswerMode.value = evt.answer_mode ?? ''
            lastConfidence.value = Math.round((evt.confidence ?? 0) * 100)
            lastSources.value = evt.sources ?? []
          } else if (evt.type === 'error') {
            streamError = evt.message ?? 'Streaming failed'
          }
        } catch {
          // Ignore non-JSON lines
        }
      }
    }

    if (streamError) throw new Error(streamError)

    // Stream finished: keep the streamed text as a message (push once)
    if (streamingText.value) {
      const assistantMsg: Message = {
        role: 'assistant',
        content: streamingText.value,
        sources: lastSources.value,
      }
      messages.value.push(assistantMsg)
    }

  } catch (err) {
    ElMessage.error(err instanceof Error && err.message ? err.message : 'The request failed; please try again')
    console.error('[QA SSE]', err)
  } finally {
    isStreaming.value = false
    streamingText.value = ''
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

onMounted(() => {
  newSession()
})
</script>

<style scoped>
.qa-page {
  display: flex;
  height: calc(100vh - 56px - 48px);
  gap: 16px;
}

/* ── Left: sessions ── */
.session-panel {
  width: 200px;
  background: #fff;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.session-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid #f0f0f0;
  font-size: 14px;
  font-weight: 500;
}
.session-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}
.session-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  color: #595959;
  transition: background 0.15s;
}
.session-item:hover { background: #f5f7fa; }
.session-item.active { background: #e6f4ff; color: #1677ff; }
.session-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.delete-btn {
  opacity: 0;
  flex-shrink: 0;
  color: #bfbfbf;
  transition: opacity 0.15s, color 0.15s;
}
.session-item:hover .delete-btn { opacity: 1; }
.delete-btn:hover { color: #ff4d4f !important; }

/* ── Right: chat ── */
.chat-panel {
  flex: 1;
  background: #fff;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}
.empty-hint {
  text-align: center;
  color: #8c8c8c;
  margin-top: 80px;
  font-size: 15px;
  line-height: 2;
}
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

/* ── Thinking dots ── */
.thinking {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 0;
}
.progress-hint {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 0;
  font-size: 13px;
  color: #8c8c8c;
}
.dot {
  width: 7px;
  height: 7px;
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

import client, { API_BASE, errorText } from './client'
import { DEMO, DEMO_NOTICE } from '@/demo'

export interface StartSessionRequest {
  target_position: string
  resume_review_id?: string | null
}

export interface StartSessionResponse {
  session_id: string
  target_position: string
  status: string
  message: string
}

export interface ChatRequest {
  message: string
}

export interface ChatResponse {
  session_id: string
  reply: string
  current_stage: string
  total_turns: number
  is_finished: boolean
  stages_reached?: string[]   // Stages the interview went through; the others before the current one were skipped
  report_summary?: {
    overall_score: number | null   // null when the report could not be generated
    strengths: string[]
    improvements: string[]
  }
}

export interface ChatDonePayload {
  type: 'done'
  reply: string          // The interviewer's whole reply for this turn
  current_stage: string
  total_turns: number
  is_finished: boolean
  stages_reached?: string[]   // Stages the interview went through; the others before the current one were skipped
  report_summary?: {
    overall_score: number | null
    strengths: string[]
    improvements: string[]
  }
}

export interface ChatStreamCallbacks {
  onToken: (chunk: string) => void
  onDone: (payload: ChatDonePayload) => void
  onError: (err: Error) => void
}

export interface DimensionEval {
  dimension: string
  score: number
  comment: string
  highlights?: string[]
  weaknesses?: string[]
}

export interface InterviewReport {
  session_id: string
  target_position: string
  overall_score: number | null
  dimensions: DimensionEval[]
  strengths: string[]
  improvements: string[]
  overall_comment: string
  recommended_topics: string[]
  next_step_advice: string
  fallback: boolean     // true when the report could not be generated
  stages_reached?: string[]
}

export interface SessionListItem {
  session_id: string
  target_position: string
  overall_score?: number
  status: string
  finished_at?: string
  created_at: string
}

export const interviewApi = {
  startSession: (data: StartSessionRequest) =>
    client.post<StartSessionResponse>('/interview/sessions', data),

  chat: (sessionId: string, data: ChatRequest) =>
    client.post<ChatResponse>(`/interview/sessions/${sessionId}/chat`, data),

  /**
   * Streamed chat (SSE) with fetch + ReadableStream, so tokens render as they arrive.
   * onToken fires per token; onDone fires at the end with the turn's final state.
   */
  chatStream(sessionId: string, message: string, callbacks: ChatStreamCallbacks): void {
    if (DEMO) {
      callbacks.onError(new Error(DEMO_NOTICE))
      return
    }
    const token = localStorage.getItem('edu-agent-token') || ''
    // Whether done or error was dispatched, for the check when the stream closes
    let doneCalled = false

    const dispatch = (line: string) => {
      if (!line.startsWith('data:')) return
      try {
        const payload = JSON.parse(line.slice(5).trim())
        if (payload.type === 'token') {
          callbacks.onToken(payload.content)
        } else if (payload.type === 'done') {
          doneCalled = true
          callbacks.onDone(payload as ChatDonePayload)
        } else if (payload.type === 'error') {
          doneCalled = true
          callbacks.onError(new Error(payload.message ?? 'Streaming failed'))
        }
      } catch {
        // Ignore keep-alive pings and other non-JSON lines
      }
    }

    fetch(`${API_BASE}/api/v1/interview/sessions/${sessionId}/chat/stream`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    })
      .then(async (resp) => {
        if (!resp.ok || !resp.body) {
          const body = await resp.json().catch(() => null)
          callbacks.onError(new Error(errorText(body?.detail) ?? `HTTP ${resp.status}`))
          return
        }

        const reader = resp.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          // The server separates SSE frames with \r\n\r\n; normalize so \n\n splits them
          // (on the whole buffer, in case a chunk ends between \r and \n)
          buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n/g, '\n')
          const parts = buffer.split('\n\n')
          buffer = parts.pop() ?? ''   // Keep an incomplete last frame for the next read

          for (const part of parts) {
            for (const line of part.split('\n')) {
              dispatch(line)
            }
          }
        }

        // A last frame without a trailing blank line
        if (buffer.trim()) {
          for (const line of buffer.split('\n')) {
            dispatch(line)
          }
        }

        // The connection closed without done or error: stop waiting
        if (!doneCalled) {
          callbacks.onError(new Error('The connection closed before the reply finished'))
        }
      })
      .catch((err) => {
        if (!doneCalled) callbacks.onError(err)
      })
  },

  getReport: (sessionId: string) =>
    client.get<InterviewReport>(`/interview/sessions/${sessionId}/report`),

  listSessions: () =>
    client.get<{ items: SessionListItem[]; total: number }>('/interview/sessions'),
}

import axios, { type AxiosError, type InternalAxiosRequestConfig } from 'axios'
import { ElMessage } from 'element-plus'
import router from '@/router'
import { useAuthStore } from '@/stores/auth'

/** Backend origin. Unset: same origin, through the Vite dev proxy or the Docker nginx.
 *  Local dev sets it in .env.local so streams skip the Vite proxy, which buffers SSE. */
export const API_BASE: string = import.meta.env.VITE_API_BASE_URL ?? ''

const client = axios.create({
  baseURL: API_BASE + '/api/v1',
  timeout: 30_000,
})

// Request interceptor: add the JWT
client.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = localStorage.getItem('edu-agent-token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// Response interceptor: one place for error messages
client.interceptors.response.use(
  res => res,
  (error: AxiosError<{ detail: unknown }>) => {
    const status = error.response?.status
    const detail = errorText(error.response?.data?.detail)

    if (status === 401) {
      // Clear the store as well as localStorage; otherwise auth.isLoggedIn stays true and the
      // navigation guard sends /login straight back to /dashboard
      useAuthStore().logout()
      router.push('/login').catch(() => {})
      return Promise.reject(error)
    }

    if (status === 403) {
      ElMessage.error('You do not have permission to do that')
    } else if (status === 429) {
      ElMessage.warning('Too many requests; please wait a moment')
    } else if (status && status >= 500) {
      ElMessage.error(detail ?? 'Something went wrong; please try again')
    } else if (detail) {
      ElMessage.error(detail)
    }

    return Promise.reject(error)
  },
)

/** The API's error detail as text: a string, {message}, or FastAPI's validation error list. */
export function errorText(detail: unknown): string | undefined {
  if (!detail) return undefined
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map(d => (d && typeof d === 'object' && 'msg' in d ? String(d.msg) : String(d))).join('; ')
  }
  if (typeof detail === 'object' && 'message' in detail) return String((detail as { message: unknown }).message)
  return undefined
}

export default client

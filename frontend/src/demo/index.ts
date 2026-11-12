// Read-only demo: a static build (VITE_DEMO_MODE=true) that needs no backend. GET requests are
// answered from fixtures.json, recorded from the real API by
// scripts/demo/export_frontend_fixtures.py; anything that would run an agent is refused.
import { AxiosError, type AxiosAdapter, type AxiosResponse, type InternalAxiosRequestConfig } from 'axios'
import fixtures from './fixtures.json'

export const DEMO = import.meta.env.VITE_DEMO_MODE === 'true'

export const DEMO_NOTICE =
  'This is a read-only demo with recorded results. Run the project locally to use the AI agents.'

export const REPO_URL = 'https://github.com/cruilnsy/edu-agent'

const responses = fixtures as Record<string, unknown>

function settle(config: InternalAxiosRequestConfig, status: number, data: unknown): Promise<AxiosResponse> {
  const response: AxiosResponse = { data, status, statusText: String(status), headers: {}, config }
  const ok = config.validateStatus ? config.validateStatus(status) : status >= 200 && status < 300
  return ok
    ? Promise.resolve(response)
    : Promise.reject(new AxiosError(`Request failed with status code ${status}`, AxiosError.ERR_BAD_REQUEST,
                                    config, null, response))
}

/** Replaces the HTTP adapter: recorded responses for reads, a notice for everything else. */
export const demoAdapter: AxiosAdapter = (config) => {
  const method = (config.method ?? 'get').toUpperCase()
  const path = (config.url ?? '').split('?')[0]
  const key = `${method} ${path}`
  if (key in responses) return settle(config, 200, responses[key])
  if (method === 'GET') return settle(config, 404, { detail: 'Not part of the demo data' })
  return settle(config, 400, { detail: DEMO_NOTICE })
}

/** The signed-in user of the demo: the fictional student the fixtures belong to. */
export function demoUser() {
  const me = responses['GET /auth/me'] as { user_id: string; tenant_id: string }
  return { userId: me.user_id, role: 'student' as const, tenantId: me.tenant_id, username: 'demo student' }
}

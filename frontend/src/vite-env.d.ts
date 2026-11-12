/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string
  readonly VITE_DEMO_MODE?: string      // 'true': read-only demo without a backend (src/demo)
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        // Backend binds 127.0.0.1 (APP_HOST); "localhost" may resolve to ::1 in Node and miss it
        target: process.env.API_TARGET ?? 'http://127.0.0.1:8010',
        changeOrigin: true,
      },
    },
  },
})

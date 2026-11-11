import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import 'element-plus/dist/index.css'
import 'highlight.js/styles/github.css'

import App from './App.vue'
import router from './router'

const app = createApp(App)

// Register every Element Plus icon
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

// Global error handler: keeps an error in the Vue scheduler or a component update from
// crashing the whole app as an unhandled rejection. AppLayout's onErrorCaptured recovers.
app.config.errorHandler = (err, _instance, info) => {
  console.error('[EduAgent] Vue error:', info, err)
}

app.use(createPinia())
app.use(router)
app.use(ElementPlus, { locale: undefined })

app.mount('#app')

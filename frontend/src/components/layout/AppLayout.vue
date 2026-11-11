<template>
  <el-container class="app-layout">
    <el-aside width="200px" class="sidebar-aside">
      <Sidebar />
    </el-aside>
    <el-container direction="vertical">
      <el-header class="app-header">
        <span class="header-title">EduAgent</span>
        <div class="header-right">
          <span class="username">{{ auth.user?.username ?? auth.user?.userId }}</span>
          <el-dropdown @command="handleCommand">
            <el-button text :icon="ArrowDown" />
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item command="logout">Log out</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </el-header>
      <!-- Progress bar while a lazy route loads, so a click shows feedback at once -->
      <div class="nav-progress-bar" :class="{ active: navigating }" />
      <el-main class="app-main">
        <!-- keep-alive keeps only QAChatView, so its sessions and messages survive switching
             pages. A routerViewKey change (error recovery) drops the cache; acceptable. -->
        <router-view v-slot="{ Component }" :key="routerViewKey">
          <keep-alive :include="['QAChatView']">
            <component :is="Component" />
          </keep-alive>
        </router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { ref, onErrorCaptured } from 'vue'
import { ArrowDown } from '@element-plus/icons-vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import Sidebar from './Sidebar.vue'

const auth = useAuthStore()
const router = useRouter()

// Show the top progress bar while a route changes (including loading its JS chunk)
const navigating = ref(false)
router.beforeEach(() => { navigating.value = true })
router.afterEach(() => { navigating.value = false })

// Error boundary for RouterView and its children. If a render fails (e.g. el = null in
// componentUpdateFn), changing the key remounts RouterView on the current route instead of
// leaving the page stuck.
const routerViewKey = ref(0)
let lastRecoveryAt = 0
onErrorCaptured((_err, _instance, _info) => {
  const now = Date.now()
  // 500 ms cool-down so the same error cannot loop
  if (now - lastRecoveryAt > 500) {
    lastRecoveryAt = now
    routerViewKey.value++
  }
  return false // Stop the error here
})

function handleCommand(cmd: string) {
  if (cmd === 'logout') {
    auth.logout()
    router.push('/login')
  }
}
</script>

<style scoped>
.app-layout {
  height: 100vh;
}
.sidebar-aside {
  background: #001529;
  overflow: hidden;
}
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #fff;
  border-bottom: 1px solid #f0f0f0;
  padding: 0 24px;
  height: 56px;
}
.header-title {
  font-size: 18px;
  font-weight: 600;
  color: #1677ff;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 8px;
}
.username {
  font-size: 14px;
  color: #595959;
}
/* Route progress bar: 3px, animated while a lazy route loads */
.nav-progress-bar {
  height: 3px;
  background: transparent;
  overflow: hidden;
  flex-shrink: 0;
}
.nav-progress-bar.active {
  background: #e8f0fe;
}
.nav-progress-bar.active::after {
  content: '';
  display: block;
  height: 100%;
  width: 40%;
  background: #1677ff;
  animation: nav-scan 0.9s ease-in-out infinite;
}
@keyframes nav-scan {
  0%   { transform: translateX(-100%); }
  100% { transform: translateX(350%); }
}
.app-main {
  background: #f5f7fa;
  overflow-y: auto;
  padding: 24px;
}
</style>

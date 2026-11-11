<template>
  <div class="sidebar">
    <div class="logo">
      <span>🎓 EduAgent</span>
    </div>
    <nav class="nav-list">
      <RouterLink to="/dashboard" class="nav-item" :class="{ 'nav-item--active': isActive('/dashboard') }">
        <el-icon><House /></el-icon>
        <span>Home</span>
      </RouterLink>

      <RouterLink to="/chat" class="nav-item" :class="{ 'nav-item--active': isActive('/chat') }">
        <el-icon><MagicStick /></el-icon>
        <span>Assistant</span>
      </RouterLink>

      <RouterLink to="/qa" class="nav-item" :class="{ 'nav-item--active': isActive('/qa') }">
        <el-icon><ChatDotRound /></el-icon>
        <span>Knowledge Q&amp;A</span>
      </RouterLink>

      <RouterLink to="/exam" class="nav-item" :class="{ 'nav-item--active': isActive('/exam') }">
        <el-icon><Document /></el-icon>
        <span>Exam Grading</span>
      </RouterLink>

      <RouterLink to="/resume" class="nav-item" :class="{ 'nav-item--active': isActive('/resume') }">
        <el-icon><Postcard /></el-icon>
        <span>Resume Review</span>
      </RouterLink>

      <RouterLink to="/interview" class="nav-item" :class="{ 'nav-item--active': isActive('/interview') }">
        <el-icon><Microphone /></el-icon>
        <span>Mock Interview</span>
      </RouterLink>

      <!-- Teacher menu (teacher or admin only) -->
      <template v-if="auth.isTeacher">
        <div class="nav-divider" />
        <RouterLink to="/teacher/exam-review" class="nav-item" :class="{ 'nav-item--active': isActive('/teacher/exam-review') }">
          <el-icon><EditPen /></el-icon>
          <span>Review Grading</span>
        </RouterLink>
        <RouterLink to="/teacher/knowledge-pending" class="nav-item" :class="{ 'nav-item--active': isActive('/teacher/knowledge-pending') }">
          <el-icon><Collection /></el-icon>
          <span>Knowledge Gaps</span>
        </RouterLink>
      </template>
    </nav>
  </div>
</template>

<script setup lang="ts">
import { useRoute } from 'vue-router'
import {
  House, ChatDotRound, Document, Postcard,
  Microphone, EditPen, Collection, MagicStick,
} from '@element-plus/icons-vue'
import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
const route = useRoute()

// Highlight for child routes too (visual only). RouterLink's active class does not cover
// sibling routes such as /resume/:id
function isActive(prefix: string) {
  return route.path === prefix || route.path.startsWith(prefix + '/')
}
</script>

<style scoped>
.sidebar {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.logo {
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-size: 16px;
  font-weight: 600;
  border-bottom: 1px solid #ffffff1a;
  flex-shrink: 0;
}

.nav-list {
  display: flex;
  flex-direction: column;
  padding: 4px 0;
  flex: 1;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 13px 20px;
  color: #ffffffa6;
  text-decoration: none;
  font-size: 14px;
  cursor: pointer;
  transition: background-color 0.2s, color 0.2s;
  user-select: none;
}

.nav-item:hover {
  background-color: #ffffff14;
  color: #fff;
}

.nav-item--active {
  background-color: #1677ff;
  color: #fff;
}

.nav-item .el-icon {
  font-size: 16px;
  flex-shrink: 0;
}

.nav-divider {
  border: none;
  border-top: 1px solid #ffffff1a;
  margin: 8px 0;
}
</style>

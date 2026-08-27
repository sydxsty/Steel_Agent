<script setup>
/**
 * WelcomeState.vue — 欢迎页面组件
 * =================================
 * 功能 / Features:
 *   - 显示助手头像、欢迎语和建议提示卡片
 *   - 点击建议提示时 emit 'select-prompt' 事件，自动发起对话
 *
 * 该组件在消息列表为空时显示（由 ChatView 的 showWelcome 控制）
 */
import { suggestedPrompts } from '../../data/mock.js'
import UserAvatar from '../common/UserAvatar.vue'

// ============================================================
// 事件定义 / Emits
// ============================================================
const emit = defineEmits(['select-prompt'])

// ============================================================
// 方法 / Methods
// ============================================================

/**
 * 处理建议提示点击 — emit 文本给父组件 ChatView
 * Handle suggested prompt click — emit text to parent ChatView
 * ChatView 收到事件后调用 handleSend() 发起流式对话
 *
 * @param {string} text - 选中的建议提示文本
 */
function handlePromptClick(text) {
  emit('select-prompt', text)
}

// ============================================================
// SVG 图标路径映射（与 mock.js 中的 icon 字段对应）
// ============================================================
const iconMap = {
  lightbulb:
    'M9.663 17h4.674M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z',
  code: 'M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4',
  book: 'M4 19.5A2.5 2.5 0 016.5 17H20 M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z',
  chart:
    'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2z M23 19v-10a2 2 0 00-2-2h-2a2 2 0 00-2 2v10a2 2 0 002 2h2a2 2 0 002-2z M16 19V5a2 2 0 00-2-2h-2a2 2 0 00-2 2v14a2 2 0 002 2h2a2 2 0 002-2z',
  globe:
    'M21 12a9 9 0 11-18 0 9 9 0 0118 0z M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 M12 2a15.3 15.3 0 00-4 10 15.3 15.3 0 004 10 M2.458 7h19.084 M2.458 17h19.084',
  calendar:
    'M19 4H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V6a2 2 0 00-2-2z M16 2v4 M8 2v4 M3 10h18 M8 14h.01 M12 14h.01 M16 14h.01 M8 18h.01 M12 18h.01 M16 18h.01',
}
</script>

<template>
  <div class="welcome-state">
    <!-- 助手头像 / Bot avatar -->
    <UserAvatar :size="72" role="bot" />

    <!-- 欢迎语 / Greeting -->
    <h1 class="greeting">你好！我是 Steel Multi-Agent System (SMAS)，专注于管线钢</h1>
    <p class="subtitle">选择一个话题开始对话，或者直接输入你的问题</p>

    <!-- 建议提示卡片网格 / Suggested prompts grid -->
    <div class="prompts-grid">
      <button
        v-for="prompt in suggestedPrompts"
        :key="prompt.id"
        class="prompt-card"
        @click="handlePromptClick(prompt.text)"
      >
        <!-- 图标 / Icon -->
        <svg
          class="prompt-icon"
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="1.5"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path :d="iconMap[prompt.icon]" />
        </svg>
        <span class="prompt-text">{{ prompt.text }}</span>
      </button>
    </div>
  </div>
</template>

<style scoped>
.welcome-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  padding: 40px var(--space-xl);
  overflow-y: auto;
}

.greeting {
  font-size: var(--font-size-2xl);
  font-weight: 700;
  color: var(--text-primary);
  margin-top: var(--space-xl);
}

.subtitle {
  font-size: 15px;
  color: var(--text-secondary);
  margin-top: var(--space-sm);
}

.prompts-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: var(--space-md);
  max-width: 640px;
  width: 100%;
  margin-top: var(--space-3xl);
}

.prompt-card {
  display: flex;
  align-items: flex-start;
  gap: var(--space-md);
  padding: 14px 18px;
  background: var(--bg-white);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  text-align: left;
  cursor: pointer;
  transition: all 0.2s ease;
}

.prompt-card:hover {
  border-color: var(--primary-light);
  background: #f8f7ff;
  box-shadow: 0 2px 8px rgba(108, 92, 231, 0.1);
}

.prompt-icon {
  color: var(--primary);
  flex-shrink: 0;
  margin-top: 1px;
}

.prompt-text {
  font-size: var(--font-size-sm);
  color: var(--text-primary);
  line-height: 1.5;
}
</style>

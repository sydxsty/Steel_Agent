<script setup>
/**
 * MessageList.vue — 消息列表组件
 * ===============================
 * 功能 / Features:
 *   - 渲染消息气泡列表
 *   - 自动滚动到底部（新消息到达或流式内容更新时）
 *   - 智能滚动：用户手动上滚查看历史消息时暂停自动滚动
 *   - 将重发事件向上传递给 ChatView
 *
 * Props:
 *   messages:    消息数组
 *   isStreaming: 是否正在流式生成（触发持续滚动）
 *
 * Emits:
 *   resend: 重发消息事件（携带消息 ID）
 */
import { ref, watch, nextTick, onMounted } from 'vue'
import MessageBubble from './MessageBubble.vue'

// ============================================================
// Props / 属性定义
// ============================================================
const props = defineProps({
  /** 消息列表 / Message list */
  messages: {
    type: Array,
    required: true,
  },
  /** 是否正在流式生成中 / Whether streaming is in progress */
  isStreaming: {
    type: Boolean,
    default: false,
  },
})

// ============================================================
// Emits / 事件定义
// ============================================================
const emit = defineEmits(['resend'])

// ============================================================
// 响应式状态 / Reactive State
// ============================================================

/** 消息列表容器 DOM 引用 / Message list container element ref */
const listRef = ref(null)

/** 用户是否手动上滚了（上滚后暂停自动滚动）/ Whether user scrolled up manually */
const userScrolledUp = ref(false)

// ============================================================
// 方法 / Methods
// ============================================================

/**
 * 判断用户是否在滚动容器底部附近
 * Check if the user is near the bottom of the scroll container
 *
 * 在底部 60px 范围内视为"在底部"，用于决定是否自动滚动
 * Within 60px of the bottom is considered "at bottom" for auto-scroll decisions
 *
 * @returns {boolean} true = 在底部附近
 */
function isNearBottom() {
  const el = listRef.value
  if (!el) return true
  // scrollHeight - scrollTop - clientHeight = 距底部的距离
  return el.scrollHeight - el.scrollTop - el.clientHeight < 60
}

/**
 * 滚动到底部
 * Scroll to the bottom of the message list
 *
 * @param {boolean} smooth - 是否使用平滑滚动（新消息时 true，流式更新时 false）
 */
function scrollToBottom(smooth = false) {
  nextTick(() => {
    const el = listRef.value
    if (!el) return
    el.scrollTo({
      top: el.scrollHeight,
      behavior: smooth ? 'smooth' : 'auto',
    })
  })
}

/**
 * 处理滚动事件 — 检测用户是否手动上滚
 * Handle scroll event — detect if user scrolled up manually
 */
function handleScroll() {
  userScrolledUp.value = !isNearBottom()
}

/**
 * 处理重发事件 — 向上传递给 ChatView
 * Handle resend event — bubble up to ChatView
 */
function handleResend(msgId) {
  emit('resend', msgId)
}

// ============================================================
// 监听器 / Watchers
// ============================================================

/**
 * 监听消息数量变化 — 新消息时滚动到底部
 * Watch message count changes — scroll to bottom on new messages
 */
watch(
  () => props.messages.length,
  () => {
    // 新消息到达时始终滚动（使用平滑动画）
    // Always scroll on new message (with smooth animation)
    userScrolledUp.value = false
    scrollToBottom(true)
  }
)

/**
 * 监听最后一条消息的内容长度 — 流式生成期间持续滚动
 * Watch last message content length — continuously scroll during streaming
 *
 * 仅在 isStreaming 为 true 且用户未手动上滚时才自动滚动
 * 流式更新使用 instant 滚动（behavior: 'auto'），避免平滑动画堆积
 */
watch(
  () => {
    const msgs = props.messages
    if (msgs.length === 0) return ''
    const last = msgs[msgs.length - 1]
    if (last.role !== 'assistant') return ''
    return `${last.content}|${last.retrieval?.status || ''}|${last.retrieval?.documents?.length || 0}`
  },
  () => {
    if (props.isStreaming && !userScrolledUp.value) {
      scrollToBottom(false) // 流式期间用 instant 滚动
    }
  }
)

// ============================================================
// 生命周期 / Lifecycle
// ============================================================
onMounted(() => {
  // 组件挂载后滚动到底部（显示最新消息）
  scrollToBottom()
})
</script>

<template>
  <div
    ref="listRef"
    class="message-list"
    @scroll="handleScroll"
  >
    <MessageBubble
      v-for="msg in messages"
      :key="msg.id"
      :message="msg"
      @resend="handleResend"
    />
  </div>
</template>

<style scoped>
.message-list {
  flex: 1;
  overflow-y: auto;
  padding: var(--space-2xl) 0;

  /*
   * 在流式输出时添加微弱的滚动条高亮效果
   * 提示用户内容正在更新中（UI 细节）
   */
  scroll-behavior: auto;
}
</style>

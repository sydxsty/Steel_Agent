<script setup>
/**
 * MessageBubble.vue — 消息气泡组件
 * ==================================
 * 功能 / Features:
 *   - 渲染用户消息和 AI 消息气泡
 *   - AI 消息三种显示模式：
 *       1. 思考中（isThinking）  → 三点跳动动画
 *       2. 流式输出（isStreaming）→ 纯文本 + 闪烁光标
 *       3. 完成态               → v-html 渲染 markdown
 *   - 复制按钮（仅 AI 消息、完成态可见）
 *   - 重发按钮（仅用户消息，hover 时可见）
 *
 * Props:
 *   message: { id, role, content, time, isThinking?, isStreaming? }
 *
 * Emits:
 *   resend: 用户点击重发按钮时触发，携带消息 ID
 */
import { ref, computed } from 'vue'
import { marked } from 'marked'
import UserAvatar from '../common/UserAvatar.vue'

// 配置 marked：支持表格、图片、换行
marked.setOptions({
  breaks: true,        // 单个换行符也转 <br>
  gfm: true,           // GitHub Flavored Markdown（表格、任务列表等）
})

// ============================================================
// Props / 属性定义
// ============================================================
const props = defineProps({
  message: {
    type: Object,
    required: true,
  },
})

// ============================================================
// Emits / 事件定义
// ============================================================
const emit = defineEmits(['resend'])

// ============================================================
// 响应式状态 / Reactive State
// ============================================================

/** 复制按钮状态：'idle' | 'copied' */
const copyState = ref('idle')

/** 将 Markdown 渲染为 HTML，并为宽表格提供气泡内横向滚动容器。 */
function renderMarkdown(markdown) {
  if (!markdown) return ''
  // 工程报告常用“1060~920℃”表示数值范围，但 Markdown 会把不同范围中的
  // 波浪号配对成删除线，导致中间整段文字出现很长的横线。仅规范数字范围，
  // 保留其它位置合法的 Markdown 删除线语法。
  const numericRangePattern = new RegExp(
    '(\\d)\\s*[~\\uFF5E]{1,2}\\s*(?=[+-]?\\d)',
    'g',
  )
  const normalizedMarkdown = String(markdown).replace(numericRangePattern, '$1-')
  return marked.parse(normalizedMarkdown).replace(
    /<table([\s\S]*?)<\/table>/g,
    '<div class="markdown-table-scroll"><table$1</table></div>',
  )
}

/** 将 markdown 内容渲染为 HTML */
const renderedContent = computed(() => {
  return renderMarkdown(props.message.content)
})

const renderedRetrievalMarkdown = computed(() => {
  return renderMarkdown(props.message.retrieval?.markdown)
})

// ============================================================
// 方法 / Methods
// ============================================================

/**
 * 将 HTML 内容提取为纯文本（用于复制）
 * Extract plain text from HTML content (for copying)
 *
 * 使用临时 DOM 元素提取文本，去除所有 HTML 标签
 *
 * @param {string} html - HTML 字符串
 * @returns {string} 纯文本内容
 */
function htmlToPlainText(html) {
  if (!html) return ''
  const div = document.createElement('div')
  div.innerHTML = html
  return div.textContent || div.innerText || ''
}

/**
 * 复制消息内容到剪贴板
 * Copy message content to clipboard
 *
 * 优先使用 navigator.clipboard API，降级到 execCommand 兼容旧浏览器
 * 复制成功后显示 ✓ 图标 2 秒
 */
async function copyContent() {
  const text = htmlToPlainText(props.message.content)

  try {
    // 首选方案：Clipboard API（现代浏览器）
    // Preferred: Clipboard API (modern browsers)
    await navigator.clipboard.writeText(text)
    copyState.value = 'copied'
  } catch {
    // 降级方案：execCommand（旧浏览器兼容）
    // Fallback: execCommand (legacy browser support)
    const textarea = document.createElement('textarea')
    textarea.value = text
    textarea.style.position = 'fixed'
    textarea.style.opacity = '0'
    document.body.appendChild(textarea)
    textarea.select()
    try {
      document.execCommand('copy')
      copyState.value = 'copied'
    } catch {
      // 复制失败，静默处理 / Copy failed, silently ignore
    }
    document.body.removeChild(textarea)
  }

  // 2 秒后恢复图标 / Restore icon after 2 seconds
  if (copyState.value === 'copied') {
    setTimeout(() => {
      copyState.value = 'idle'
    }, 2000)
  }
}

function buildPdfHtml() {
  const parts = []
  // retrieval 保存的是网页端“模型处理过程/思维链”。导出文件只包含最终正文报告，
  // 页面仍继续按当前形式展示 retrieval，不改变计算过程的实时可见性。
  if (renderedContent.value) {
    parts.push(`<section class="pdf-section">${renderedContent.value}</section>`)
  }
  return parts.join('\n')
}

function waitForPdfImages(doc, timeoutMs = 15000) {
  const images = Array.from(doc.images || [])
  if (!images.length) return Promise.resolve()

  const waitOne = (img) => new Promise((resolve) => {
    if (img.complete && img.naturalWidth > 0) {
      if (typeof img.decode === 'function') {
        img.decode().then(resolve).catch(resolve)
      } else {
        resolve()
      }
      return
    }

    const done = () => {
      img.removeEventListener('load', done)
      img.removeEventListener('error', done)
      resolve()
    }
    img.addEventListener('load', done, { once: true })
    img.addEventListener('error', done, { once: true })
  })

  return Promise.race([
    Promise.all(images.map(waitOne)),
    new Promise((resolve) => setTimeout(resolve, timeoutMs)),
  ])
}

function downloadPdf() {
  const html = buildPdfHtml()
  if (!html) return

  const iframe = document.createElement('iframe')
  iframe.style.position = 'fixed'
  iframe.style.right = '0'
  iframe.style.bottom = '0'
  iframe.style.width = '1px'
  iframe.style.height = '1px'
  iframe.style.border = '0'
  iframe.style.opacity = '0'
  document.body.appendChild(iframe)

  const doc = iframe.contentDocument || iframe.contentWindow?.document
  if (!doc) {
    document.body.removeChild(iframe)
    return
  }

  doc.open()
  doc.write(`<!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>AI 报告</title>
        <style>
          @page { margin: 16mm 14mm; }
          * { box-sizing: border-box; }
          body {
            margin: 0;
            color: #111827;
            font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
            font-size: 12px;
            line-height: 1.72;
          }
          h1, h2, h3, h4 {
            margin: 18px 0 10px;
            color: #111827;
            line-height: 1.35;
            page-break-after: avoid;
          }
          h1 { font-size: 22px; }
          h2 { font-size: 18px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }
          h3 { font-size: 15px; }
          p { margin: 0 0 10px; }
          table {
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0 16px;
            page-break-inside: auto;
          }
          tr { page-break-inside: avoid; page-break-after: auto; }
          th, td {
            border: 1px solid #d1d5db;
            padding: 6px 8px;
            vertical-align: top;
            text-align: center;
            word-break: break-word;
          }
          th { background: #f3f4f6; font-weight: 700; }
          code {
            padding: 1px 4px;
            border-radius: 3px;
            background: #f3f4f6;
            font-family: Consolas, "Courier New", monospace;
          }
          pre {
            white-space: pre-wrap;
            word-break: break-word;
            padding: 10px;
            border: 1px solid #e5e7eb;
            background: #f9fafb;
          }
          blockquote {
            margin: 10px 0;
            padding: 8px 12px;
            border-left: 3px solid #9ca3af;
            background: #f9fafb;
          }
          img {
            display: block;
            max-width: 100%;
            height: auto;
            margin: 10px 0;
            page-break-inside: avoid;
          }
          img[src*="/generated-images/"] {
            width: 80%;
            max-width: 80%;
            margin: 10px auto;
          }
          .report-figure-caption {
            display: block;
            width: 80%;
            max-width: 80%;
            margin: 4px auto 12px;
            text-align: center;
            text-indent: 0;
            font-size: 13px;
            line-height: 1.5;
            letter-spacing: 0;
            page-break-before: avoid;
            break-before: avoid;
          }
          .report-table-caption {
            display: block;
            width: 80%;
            max-width: 80%;
            margin: 12px auto 6px;
            text-align: center;
            text-indent: 0;
            font-size: 13px;
            line-height: 1.5;
            letter-spacing: 0;
            page-break-after: avoid;
            break-after: avoid;
          }
          .pdf-section { margin-bottom: 16px; }
          .retrieval {
            padding: 10px 12px;
            border-left: 3px solid #6366f1;
            background: #f8fafc;
          }
          @media print {
            body { print-color-adjust: exact; -webkit-print-color-adjust: exact; }
          }
        </style>
      </head>
      <body>${html}</body>
    </html>
  `)
  doc.close()

  const cleanup = () => {
    setTimeout(() => {
      if (iframe.parentNode) iframe.parentNode.removeChild(iframe)
    }, 5000)
  }

  iframe.onload = async () => {
    await waitForPdfImages(doc)
    setTimeout(() => {
      iframe.contentWindow?.addEventListener('afterprint', cleanup, { once: true })
      iframe.contentWindow?.focus()
      iframe.contentWindow?.print()
      cleanup()
    }, 300)
  }
}

/**
 * 重发消息
 * Resend message — emit 事件给父组件 MessageList → ChatView
 */
function handleResend() {
  emit('resend', props.message.id)
}
</script>

<template>
  <div
    class="message-wrapper"
    :class="message.role"
  >
    <!-- ============================================ -->
    <!-- AI 头像（仅 AI 消息显示）                      -->
    <!-- ============================================ -->
    <UserAvatar
      v-if="message.role === 'assistant'"
      :size="32"
      role="bot"
      class="msg-avatar"
    />

    <div class="message-body">
      <!-- ============================================ -->
      <!-- AI 消息气泡                                 -->
      <!-- ============================================ -->
      <div
        v-if="message.role === 'assistant'"
        class="message-bubble assistant-bubble"
      >
        <div
          v-if="message.designVersion"
          class="design-version-badge"
          :title="`当前成功设计版本：方案${message.designVersion}`"
        >
          方案{{ message.designVersion }}
        </div>
        <div
          v-if="message.retrieval"
          class="retrieval-panel"
        >
          <div class="retrieval-status">
            {{ message.retrieval.message }}
          </div>
          <div
            v-if="message.retrieval.markdown"
            class="retrieval-markdown markdown-body"
            v-html="renderedRetrievalMarkdown"
          ></div>
          <div
            v-else-if="message.retrieval.content"
            class="retrieval-streaming"
          >{{ message.retrieval.content }}</div>
          <ol
            v-else-if="message.retrieval.documents && message.retrieval.documents.length"
            class="retrieval-list"
          >
            <li
              v-for="doc in message.retrieval.documents"
              :key="doc.chunk_id || `${doc.source}-${doc.rank}`"
              class="retrieval-item"
            >
              <div class="retrieval-title">{{ doc.title || `检索结果 ${doc.rank}` }}</div>
              <div class="retrieval-summary">{{ doc.summary }}</div>
              <div class="retrieval-meta">
                <span v-if="doc.source">来源：{{ doc.source }}</span>
                <span v-if="doc.section">section：{{ doc.section }}</span>
                <span v-if="doc.page">页码：{{ doc.page }}</span>
                <span v-if="doc.chunk_id">chunk_id：{{ doc.chunk_id }}</span>
              </div>
            </li>
          </ol>
        </div>

        <!--
          模式 1: 思考中 — 三点跳动动画
          Mode 1: Thinking — three-dot bouncing animation
          三个圆点使用不同的 animation-delay 实现错开弹跳效果
        -->
        <div v-if="message.isThinking" class="thinking-indicator">
          <span class="dot"></span>
          <span class="dot"></span>
          <span class="dot"></span>
        </div>

        <!--
          模式 2: 流式输出中 — 纯文本 + 闪烁光标
          Mode 2: Streaming — plain text + blinking cursor
          使用 {{ }} 文本插值而非 v-html，避免不完整 markdown 被错误渲染
          streaming-cursor 类通过 ::after 伪元素显示闪烁的 | 光标
        -->
        <div
          v-else-if="message.isStreaming"
          class="message-content streaming-cursor"
        >{{ message.content }}</div>

        <!--
          模式 3: 完成态 — markdown 渲染（支持表格、图片、代码块等）
          Mode 3: Complete — markdown rendered (supports tables, images, code blocks)
        -->
        <div
          v-else
          class="message-content markdown-body"
          v-html="renderedContent"
        ></div>

        <!--
          AI 消息操作按钮栏（hover 时显示）
          AI message action buttons (visible on hover)
          包含复制按钮
        -->
        <div class="message-actions">
          <!-- 复制按钮 / Copy button -->
          <button
            class="action-btn"
            :class="{ copied: copyState === 'copied' }"
            title="复制"
            @click="copyContent"
          >
            <!-- 默认：复制图标 -->
            <svg
              v-if="copyState === 'idle'"
              width="15"
              height="15"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
              <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
            </svg>
            <!-- 复制成功：勾选图标 -->
            <svg
              v-else
              width="15"
              height="15"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2.5"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <polyline points="20 6 9 17 4 12" />
            </svg>
          </button>

          <!-- PDF 下载按钮 / PDF download button -->
          <button
            class="action-btn"
            title="下载 PDF"
            @click="downloadPdf"
          >
            <svg
              width="15"
              height="15"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
          </button>
        </div>
      </div>

      <!-- ============================================ -->
      <!-- 用户消息气泡                                 -->
      <!-- ============================================ -->
      <div
        v-else
        class="message-bubble user-bubble"
      >
        {{ message.content }}
      </div>

      <!-- ============================================ -->
      <!-- 时间戳 + 用户消息操作按钮                     -->
      <!-- ============================================ -->
      <div class="message-meta">
        <!-- 时间戳 / Timestamp -->
        <span class="message-time">{{ message.time }}</span>

        <!--
          重发按钮（仅用户消息）
          Resend button (user messages only)
          点击后删除该消息及后续对话，用相同文本重新发送
        -->
        <button
          v-if="message.role === 'user'"
          class="action-btn resend-btn"
          title="重新发送"
          @click="handleResend"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <polyline points="23 4 23 10 17 10" />
            <path d="M20.49 15a9 9 0 11-2.12-9.36L23 10" />
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* ============================================================
   消息布局 / Message Layout
   ============================================================ */
.message-wrapper {
  display: flex;
  padding: 0 var(--space-xl);
  margin-bottom: var(--space-2xl);
  max-width: 768px;
  margin-left: auto;
  margin-right: auto;
}

.message-wrapper.assistant {
  justify-content: flex-start;
}

.message-wrapper.user {
  justify-content: flex-end;
}

.msg-avatar {
  margin-right: var(--space-md);
  margin-top: 2px;
}

.message-body {
  display: flex;
  flex-direction: column;
  max-width: 75%;
}

.message-wrapper.user .message-body {
  align-items: flex-end;
}

/* ============================================================
   气泡样式 / Bubble Styles
   ============================================================ */
.message-bubble {
  padding: var(--space-md) var(--space-xl);
  font-size: var(--font-size-base);
  line-height: 1.7;
}

.assistant-bubble {
  background: var(--bg-white);
  border: 1px solid var(--border-color);
  border-radius: 4px 16px 16px 16px;
  box-shadow: var(--shadow-card);
  color: var(--text-primary);
}

.design-version-badge {
  display: inline-flex;
  align-items: center;
  min-height: 22px;
  margin-bottom: var(--space-sm);
  padding: 2px 8px;
  border: 1px solid #c7d2fe;
  border-radius: 4px;
  background: #eef2ff;
  color: #4338ca;
  font-size: var(--font-size-xs);
  font-weight: 600;
  line-height: 1.4;
}

.retrieval-panel {
  margin-bottom: var(--space-md);
  padding: var(--space-md) var(--space-lg);
  background: #f1f3f8;
  border-left: 3px solid #c7d2fe;
  border-radius: var(--radius-md);
  color: var(--text-secondary);
}

.retrieval-status {
  margin-bottom: var(--space-sm);
  font-size: var(--font-size-sm);
  font-weight: 600;
  color: var(--text-primary);
}

.retrieval-markdown {
  font-size: var(--font-size-sm);
  line-height: 1.7;
}

.retrieval-markdown :deep(hr) {
  margin: var(--space-md) 0;
  border: none;
  border-top: 1px solid #d8deea;
}

.retrieval-markdown :deep(p) {
  margin-bottom: var(--space-sm);
}

.retrieval-streaming {
  white-space: pre-wrap;
  font-size: var(--font-size-sm);
  line-height: 1.7;
  color: var(--text-primary);
}

.retrieval-streaming::after {
  content: '|';
  display: inline-block;
  margin-left: 2px;
  color: var(--primary);
  animation: blink-cursor 1s step-end infinite;
}

.retrieval-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-sm);
  margin: 0;
  padding-left: 18px;
}

.retrieval-item {
  padding-bottom: var(--space-sm);
  border-bottom: 1px solid rgba(148, 163, 184, 0.28);
}

.retrieval-item:last-child {
  padding-bottom: 0;
  border-bottom: 0;
}

.retrieval-title {
  font-size: var(--font-size-sm);
  font-weight: 600;
  color: var(--primary);
}

.retrieval-summary {
  margin-top: 2px;
  font-size: var(--font-size-sm);
  line-height: 1.6;
  color: var(--text-primary);
}

.retrieval-meta {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-xs) var(--space-md);
  margin-top: 4px;
  font-size: var(--font-size-xs);
  color: var(--text-secondary);
}

.user-bubble {
  background: var(--primary);
  color: white;
  border-radius: 16px 16px 4px 16px;
}

/* ============================================================
   三点思考动画 / Three-Dot Thinking Animation
   参考 DeepSeek/Coze 的等待样式
   ============================================================ */
.thinking-indicator {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 6px 0;
  min-height: 24px;
}

.thinking-indicator .dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--primary-light);
  /* 动画：循环缩放 + 透明度变化 / Animation: cyclic scale + opacity */
  animation: thinking-bounce 1.4s infinite ease-in-out both;
}

/* 三个圆点错开延迟，实现依次弹跳效果 */
.thinking-indicator .dot:nth-child(1) {
  animation-delay: -0.32s;
}
.thinking-indicator .dot:nth-child(2) {
  animation-delay: -0.16s;
}
.thinking-indicator .dot:nth-child(3) {
  animation-delay: 0s;
}

@keyframes thinking-bounce {
  0%,
  80%,
  100% {
    transform: scale(0.6);
    opacity: 0.35;
  }
  40% {
    transform: scale(1.25);
    opacity: 1;
  }
}

/* ============================================================
   流式输出光标 / Streaming Cursor
   ::after 伪元素显示闪烁的 | 符号
   ============================================================ */
.streaming-cursor::after {
  content: '|';
  display: inline;
  animation: blink-cursor 1s step-end infinite;
  color: var(--primary);
  font-weight: 300;
  margin-left: 1px;
}

@keyframes blink-cursor {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0;
  }
}

/* ============================================================
   消息元信息行（时间戳 + 操作按钮）/ Message Meta Row
   ============================================================ */
.message-meta {
  display: flex;
  align-items: center;
  gap: var(--space-xs);
  margin-top: var(--space-xs);
  padding: 0 4px;
}

.message-time {
  font-size: var(--font-size-xs);
  color: var(--text-secondary);
}

/* ============================================================
   操作按钮 / Action Buttons
   ============================================================ */
.message-actions {
  display: flex;
  gap: 4px;
  margin-top: 6px;
  /* 默认透明，hover 时显示 / Hidden by default, show on hover */
  opacity: 0;
  transition: opacity 0.2s ease;
}

/* 鼠标悬停在消息体上时显示操作按钮 */
.message-body:hover .message-actions {
  opacity: 1;
}

.action-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  transition: all 0.15s ease;
  flex-shrink: 0;
}

.action-btn:hover {
  background: var(--hover-bg);
  color: var(--text-primary);
}

/* 复制成功状态 / Copy success state */
.action-btn.copied {
  color: #10b981;
}

/* 重发按钮始终显示 / Resend button always visible */
.resend-btn {
  opacity: 0;
  transition: opacity 0.2s ease;
}

.message-body:hover .resend-btn {
  opacity: 1;
}
</style>

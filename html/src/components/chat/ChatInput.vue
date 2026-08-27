<script setup>
/**
 * 聊天输入框与临时附件队列。
 *
 * 附件按选择顺序逐个上传、逐个解析。组件只向聊天主流程传递已完成附件的
 * attachment_id，不在浏览器内保存可能很长的 Markdown 正文。
 */
import { computed, nextTick, ref, watch } from 'vue'
import ModelSelector from '../common/ModelSelector.vue'

const API_BASE_URL = 'http://localhost:8000'
const MAX_FILE_BYTES = 10 * 1024 * 1024
const MAX_FILES = 5
const MAX_PDFS = 2
const ACCEPTED_EXTENSIONS = [
  '.docx', '.xlsx', '.pdf', '.md', '.markdown', '.txt',
  '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff',
]

const props = defineProps({
  disabled: { type: Boolean, default: false },
  sessionId: { type: String, required: true },
})

const emit = defineEmits(['send'])
const inputText = ref('')
const textareaRef = ref(null)
const fileInputRef = ref(null)
const attachments = ref([])
const attachmentNotice = ref('')
const queueRunning = ref(false)

const hasBlockingAttachment = computed(() => attachments.value.some(
  item => item.status !== 'ready',
))
const sendDisabled = computed(() => (
  props.disabled
  || !inputText.value.trim()
  || hasBlockingAttachment.value
))

function extensionOf(name) {
  const dot = name.lastIndexOf('.')
  return dot >= 0 ? name.slice(dot).toLowerCase() : ''
}

function formatFileSize(size) {
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / 1024 / 1024).toFixed(1)} MB`
}

function statusLabel(item) {
  const labels = {
    queued: '等待上传',
    uploading: '正在上传',
    parsing: item.message || '正在解析',
    ready: '解析完成',
    error: item.message || '解析失败',
  }
  return labels[item.status] || item.status
}

function openFilePicker() {
  if (!props.disabled) fileInputRef.value?.click()
}

/**
 * 前端先做即时限制校验；相同规则仍会在后端再次执行，避免绕过。
 */
function handleFileSelection(event) {
  attachmentNotice.value = ''
  const selected = Array.from(event.target.files || [])
  let pdfCount = attachments.value.filter(item => item.extension === '.pdf').length

  for (const file of selected) {
    if (attachments.value.length >= MAX_FILES) {
      attachmentNotice.value = `每次最多上传 ${MAX_FILES} 个附件。`
      break
    }
    const extension = extensionOf(file.name)
    if (!ACCEPTED_EXTENSIONS.includes(extension)) {
      attachmentNotice.value = `不支持 ${file.name}；请使用 DOCX、XLSX、PDF、Markdown、TXT 或图片。`
      continue
    }
    if (file.size > MAX_FILE_BYTES) {
      attachmentNotice.value = `${file.name} 超过 10 MiB，未加入附件。`
      continue
    }
    if (file.size <= 0) {
      attachmentNotice.value = `${file.name} 是空文件，未加入附件。`
      continue
    }
    if (extension === '.pdf' && pdfCount >= MAX_PDFS) {
      attachmentNotice.value = `每次最多上传 ${MAX_PDFS} 个 PDF。`
      continue
    }
    if (extension === '.pdf') pdfCount += 1
    attachments.value.push({
      localId: crypto.randomUUID(),
      attachmentId: '',
      file,
      name: file.name,
      size: file.size,
      extension,
      status: 'queued',
      progress: 0,
      message: '',
      cancelled: false,
      xhr: null,
    })
  }
  // 允许用户再次选择同名文件。
  event.target.value = ''
  drainQueue()
}

function parseErrorPayload(text, fallback) {
  try {
    const payload = JSON.parse(text)
    return payload.detail || payload.error || fallback
  } catch {
    return fallback
  }
}

/** 使用 XHR 才能获得浏览器原生的上传字节进度。 */
function uploadFile(item) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    item.xhr = xhr
    xhr.open('POST', `${API_BASE_URL}/attachments/upload`)
    xhr.upload.onprogress = event => {
      if (event.lengthComputable && !item.cancelled) {
        item.progress = Math.round((event.loaded / event.total) * 40)
      }
    }
    xhr.onload = () => {
      item.xhr = null
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText))
        } catch {
          reject(new Error('上传接口返回了无效数据'))
        }
      } else {
        reject(new Error(parseErrorPayload(xhr.responseText, `上传失败 (${xhr.status})`)))
      }
    }
    xhr.onerror = () => reject(new Error('附件上传连接失败'))
    xhr.onabort = () => reject(new DOMException('附件上传已取消', 'AbortError'))
    const form = new FormData()
    form.append('session_id', props.sessionId)
    form.append('file', item.file, item.name)
    xhr.send(form)
  })
}

async function startParsing(item) {
  const response = await fetch(
    `${API_BASE_URL}/attachments/${encodeURIComponent(item.attachmentId)}/parse`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: props.sessionId }),
    },
  )
  if (!response.ok) {
    throw new Error(parseErrorPayload(await response.text(), `无法开始解析 (${response.status})`))
  }
}

function delay(milliseconds) {
  return new Promise(resolve => setTimeout(resolve, milliseconds))
}

async function waitForParsing(item) {
  while (!item.cancelled) {
    const response = await fetch(
      `${API_BASE_URL}/attachments/${encodeURIComponent(item.attachmentId)}/status?session_id=${encodeURIComponent(props.sessionId)}`,
      { cache: 'no-store' },
    )
    if (!response.ok) {
      throw new Error(parseErrorPayload(await response.text(), `无法读取解析状态 (${response.status})`))
    }
    const status = await response.json()
    item.message = status.message || ''
    item.progress = Math.min(100, 40 + Math.round((Number(status.progress) || 0) * 0.6))
    if (status.status === 'ready') {
      item.status = 'ready'
      item.progress = 100
      return
    }
    if (status.status === 'error') {
      throw new Error(status.message || '附件解析失败')
    }
    if (status.status === 'cancelled') {
      throw new DOMException('附件解析已取消', 'AbortError')
    }
    item.status = status.status === 'queued' ? 'parsing' : status.status
    await delay(700)
  }
}

/**
 * 严格串行消费队列：一个附件完全 ready 后才处理下一个附件。
 */
async function drainQueue() {
  if (queueRunning.value) return
  queueRunning.value = true
  try {
    while (true) {
      const item = attachments.value.find(candidate => candidate.status === 'queued' && !candidate.cancelled)
      if (!item) break
      try {
        item.status = 'uploading'
        item.progress = 1
        const uploaded = await uploadFile(item)
        if (item.cancelled) continue
        item.attachmentId = uploaded.attachment_id
        item.status = 'parsing'
        item.progress = 40
        item.message = '等待解析'
        await startParsing(item)
        await waitForParsing(item)
      } catch (error) {
        if (!item.cancelled && error?.name !== 'AbortError') {
          item.status = 'error'
          item.progress = 100
          item.message = error.message || '附件处理失败'
        }
      }
    }
  } finally {
    queueRunning.value = false
    // 处理循环退出瞬间新加入的文件。
    if (attachments.value.some(item => item.status === 'queued' && !item.cancelled)) {
      drainQueue()
    }
  }
}

async function removeAttachment(item) {
  item.cancelled = true
  item.xhr?.abort()
  attachments.value = attachments.value.filter(candidate => candidate.localId !== item.localId)
  if (item.attachmentId) {
    try {
      await fetch(
        `${API_BASE_URL}/attachments/${encodeURIComponent(item.attachmentId)}?session_id=${encodeURIComponent(props.sessionId)}`,
        { method: 'DELETE' },
      )
    } catch {
      // 后端定时清理会兜底删除断网期间未能取消的临时附件。
    }
  }
}

function clearAcceptedDraft() {
  inputText.value = ''
  attachments.value = []
  attachmentNotice.value = ''
  if (textareaRef.value) textareaRef.value.style.height = 'auto'
}

function handleSend() {
  const text = inputText.value.trim()
  if (!text || sendDisabled.value) return
  emit('send', {
    text,
    attachmentIds: attachments.value.map(item => item.attachmentId),
    onAccepted: clearAcceptedDraft,
  })
}

function handleKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    handleSend()
  }
}

function autoResize() {
  const element = textareaRef.value
  if (!element) return
  element.style.height = 'auto'
  element.style.height = `${Math.min(element.scrollHeight, 120)}px`
}

watch(inputText, () => nextTick(autoResize))
</script>

<template>
  <div class="chat-input-container">
    <div class="input-card">
      <input
        ref="fileInputRef"
        class="file-input"
        type="file"
        multiple
        :accept="ACCEPTED_EXTENSIONS.join(',')"
        @change="handleFileSelection"
      />

      <div v-if="attachments.length" class="attachment-list" aria-label="附件列表">
        <div
          v-for="item in attachments"
          :key="item.localId"
          class="attachment-item"
          :class="`attachment-${item.status}`"
        >
          <div class="file-icon" aria-hidden="true">{{ item.extension.replace('.', '').slice(0, 4).toUpperCase() }}</div>
          <div class="attachment-main">
            <div class="attachment-title-row">
              <span class="attachment-name" :title="item.name">{{ item.name }}</span>
              <span class="attachment-size">{{ formatFileSize(item.size) }}</span>
            </div>
            <div class="attachment-status" :title="statusLabel(item)">{{ statusLabel(item) }}</div>
            <div class="progress-track" aria-hidden="true">
              <span class="progress-value" :style="{ width: `${item.progress}%` }"></span>
            </div>
          </div>
          <button
            class="remove-attachment"
            type="button"
            title="取消并移除附件"
            aria-label="取消并移除附件"
            @click="removeAttachment(item)"
          >×</button>
        </div>
      </div>

      <div v-if="attachmentNotice" class="attachment-notice">{{ attachmentNotice }}</div>

      <div class="composer-row">
        <button
          class="tool-btn"
          type="button"
          title="添加附件"
          :disabled="disabled"
          @click="openFilePicker"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
          </svg>
        </button>

        <textarea
          ref="textareaRef"
          v-model="inputText"
          class="input-textarea"
          placeholder="输入你的问题，按 Enter 发送..."
          rows="1"
          :disabled="disabled"
          @keydown="handleKeydown"
        ></textarea>

        <ModelSelector />

        <button
          class="send-btn"
          :class="{ disabled: sendDisabled }"
          type="button"
          title="发送"
          :disabled="sendDisabled"
          @click="handleSend"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="12" y1="19" x2="12" y2="5" />
            <polyline points="5 12 12 5 19 12" />
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat-input-container {
  padding: 0 var(--space-xl) var(--space-xl);
  max-width: 808px;
  width: 100%;
  align-self: center;
}

.input-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  background: var(--bg-white);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-xl);
  padding: 10px var(--space-lg);
  box-shadow: var(--shadow-input);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.input-card:focus-within {
  border-color: var(--primary-light);
  box-shadow: 0 4px 24px rgba(108, 92, 231, 0.12);
}

.file-input { display: none; }

.attachment-list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 7px;
  padding-bottom: 2px;
}

.attachment-item {
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr) 24px;
  align-items: center;
  gap: 8px;
  min-height: 56px;
  padding: 7px 6px 7px 8px;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  background: #fafbfc;
}

.attachment-error { border-color: #e8a4a4; background: #fff8f8; }
.attachment-ready { border-color: #b8d8c5; background: #f8fcfa; }

.file-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border-radius: 5px;
  background: #ece9ff;
  color: var(--primary);
  font-size: 10px;
  font-weight: 700;
  overflow: hidden;
}

.attachment-main { min-width: 0; }
.attachment-title-row { display: flex; align-items: baseline; gap: 6px; }
.attachment-name {
  min-width: 0;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--text-primary);
  font-size: 12px;
}
.attachment-size { flex-shrink: 0; color: var(--text-placeholder); font-size: 10px; }
.attachment-status {
  margin-top: 2px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--text-secondary);
  font-size: 10px;
}

.progress-track {
  height: 3px;
  margin-top: 5px;
  overflow: hidden;
  border-radius: 2px;
  background: #e7e7ec;
}
.progress-value {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--primary);
  transition: width 0.25s ease;
}
.attachment-error .progress-value { background: #c94b4b; }
.attachment-ready .progress-value { background: #3b9466; }

.remove-attachment {
  width: 24px;
  height: 24px;
  color: var(--text-secondary);
  font-size: 18px;
  line-height: 1;
}
.remove-attachment:hover { color: #b93636; background: #f5eaea; border-radius: 50%; }

.attachment-notice { color: #b4473b; font-size: 12px; }

.composer-row {
  display: flex;
  align-items: flex-end;
  gap: var(--space-sm);
}

.tool-btn, .send-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: var(--radius-round);
  flex-shrink: 0;
  transition: all 0.2s ease;
}
.tool-btn { color: var(--text-secondary); }
.tool-btn:hover:not(:disabled) { background: var(--hover-bg); color: var(--text-primary); }
.tool-btn:disabled { opacity: 0.45; cursor: not-allowed; }

.input-textarea {
  flex: 1;
  min-height: 24px;
  max-height: 120px;
  padding: 6px 4px;
  font-size: var(--font-size-base);
  line-height: 1.5;
  resize: none;
  background: transparent;
}
.input-textarea:disabled { opacity: 0.6; cursor: not-allowed; }
.input-textarea::placeholder { color: var(--text-placeholder); }

.send-btn { background: var(--primary); color: white; }
.send-btn:hover:not(.disabled) { background: var(--primary-hover); transform: scale(1.05); }
.send-btn.disabled { opacity: 0.4; cursor: not-allowed; }

@media (max-width: 680px) {
  .attachment-list { grid-template-columns: minmax(0, 1fr); }
  .chat-input-container { padding-inline: var(--space-md); }
}
</style>

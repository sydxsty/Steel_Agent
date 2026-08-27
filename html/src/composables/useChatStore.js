import { computed, reactive, ref, watch } from 'vue'

const NEW_SESSION_TITLE = '新会话'
const MAX_HISTORY_ITEMS = 5
const HISTORY_STORAGE_KEY = 'metal_chat_sessions_v1'
const MAX_PERSISTED_STATE_BYTES = 2 * 1024 * 1024
const INLINE_IMAGE_PLACEHOLDER = '> 旧版内联图片已清理，请重新生成报告查看图片。'

function generateSessionId() {
  try {
    return crypto.randomUUID()
  } catch {
    return `session_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`
  }
}

let _idCounter = 1

function generateId() {
  return `msg_${Date.now()}_${_idCounter++}`
}

function getCurrentTime() {
  const now = new Date()
  const h = String(now.getHours()).padStart(2, '0')
  const m = String(now.getMinutes()).padStart(2, '0')
  return `${h}:${m}`
}

function getCurrentDateLabel(timestamp = Date.now()) {
  const date = new Date(timestamp)
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${month}-${day}`
}

function escapeHtml(str) {
  const div = document.createElement('div')
  div.appendChild(document.createTextNode(str))
  return div.innerHTML
}

function normalizeTitle(text) {
  const title = text.trim().replace(/\s+/g, ' ')
  return title.length > 24 ? `${title.slice(0, 24)}...` : title
}

function sanitizeMessageContent(content = '') {
  const text = String(content || '')
  if (!text.includes('data:image/')) return text
  return text.replace(/!\[[^\]]*\]\(data:image\/[^)]+\)/g, INLINE_IMAGE_PLACEHOLDER)
    .replace(/<img\b[^>]*src=["']data:image\/[^"']+["'][^>]*>/gi, INLINE_IMAGE_PLACEHOLDER)
}

function hydrateMessage(message) {
  // 页面刷新后先保留已经显示的内容并结束本地动画；ChatView 会查询
  // 后端独立计算任务。任务仍在运行时重新订阅，已结束或后端关闭时
  // 继续展示这里保留的上次内容，不再先入为主地标记“计算中断”。
  let retrieval = message.retrieval
    ? { ...message.retrieval }
    : null
  const wasInterrupted = Boolean(message.isThinking)
    || Boolean(message.isStreaming)
    || ['searching', 'streaming'].includes(retrieval?.status)

  if (wasInterrupted && retrieval) {
    retrieval.status = 'done'
    retrieval.message = retrieval.message || '模型处理过程'
    retrieval.markdown = retrieval.markdown || retrieval.content || ''
  }

  return reactive({
    id: message.id || generateId(),
    role: message.role,
    content: sanitizeMessageContent(message.content),
    time: message.time || '',
    isThinking: false,
    isStreaming: false,
    retrieval,
    designId: message.designId || null,
    designVersion: message.designVersion || null,
  })
}

function createSessionRecord(title = NEW_SESSION_TITLE) {
  const now = Date.now()
  return reactive({
    id: generateSessionId(),
    title,
    createdAt: now,
    updatedAt: now,
    messages: [],
    activeDesignId: null,
    activeDesignVersion: null,
  })
}

function loadPersistedState() {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY)
    if (!raw) return null
    if (raw.length > MAX_PERSISTED_STATE_BYTES) {
      localStorage.removeItem(HISTORY_STORAGE_KEY)
      return null
    }

    const parsed = JSON.parse(raw)
    const loadedSessions = Array.isArray(parsed.sessions)
      ? parsed.sessions.map((session) => reactive({
        id: session.id || generateSessionId(),
        title: session.title || NEW_SESSION_TITLE,
        createdAt: session.createdAt || Date.now(),
        updatedAt: session.updatedAt || session.createdAt || Date.now(),
        activeDesignId: session.activeDesignId || null,
        activeDesignVersion: session.activeDesignVersion || null,
        messages: Array.isArray(session.messages)
          ? session.messages.map(hydrateMessage)
          : [],
      }))
      : []

    return {
      activeSessionId: parsed.activeSessionId,
      sessions: loadedSessions,
    }
  } catch {
    return null
  }
}

function persistState() {
  try {
    const payload = JSON.stringify({
      activeSessionId: activeSessionId.value,
      sessions: sessions.value.map(session => ({
        ...session,
        messages: session.messages.map(message => ({
          ...message,
          content: sanitizeMessageContent(message.content),
        })),
      })),
    })
    if (payload.length > MAX_PERSISTED_STATE_BYTES) return
    localStorage.setItem(HISTORY_STORAGE_KEY, payload)
  } catch {
    // Local storage can be unavailable in restricted browser contexts.
  }
}

const persistedState = loadPersistedState()
const initialSessions = persistedState?.sessions?.length
  ? persistedState.sessions
  : [createSessionRecord()]

const sessions = ref(initialSessions)
const activeSessionId = ref(
  initialSessions.some(session => session.id === persistedState?.activeSessionId)
    ? persistedState.activeSessionId
    : initialSessions[0].id
)
const isStreaming = ref(false)
const error = ref(null)

const currentSession = computed(() => {
  return sessions.value.find(session => session.id === activeSessionId.value) || sessions.value[0]
})

const sessionId = computed(() => currentSession.value?.id || '')
const activeDesignId = computed(() => currentSession.value?.activeDesignId || '')
const messages = computed(() => currentSession.value?.messages || [])
const showWelcome = computed(() => messages.value.length === 0)
const historyItems = computed(() => {
  return [...sessions.value]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, MAX_HISTORY_ITEMS)
    .map(session => ({
      id: session.id,
      title: session.title,
      date: getCurrentDateLabel(session.updatedAt),
    }))
})

function trimHistoryLimit() {
  if (sessions.value.length <= MAX_HISTORY_ITEMS) return

  const removable = [...sessions.value]
    .filter(session => session.id !== activeSessionId.value)
    .sort((a, b) => a.updatedAt - b.updatedAt)

  while (sessions.value.length > MAX_HISTORY_ITEMS && removable.length) {
    const session = removable.shift()
    const index = sessions.value.findIndex(item => item.id === session.id)
    if (index !== -1) sessions.value.splice(index, 1)
  }
}

function touchSession(session = currentSession.value) {
  if (!session) return
  session.updatedAt = Date.now()
}

function createNewSession() {
  const session = createSessionRecord()
  sessions.value.unshift(session)
  activeSessionId.value = session.id
  error.value = null
  isStreaming.value = false
  trimHistoryLimit()
  return session
}

function selectSession(id) {
  const session = sessions.value.find(item => item.id === id)
  if (!session || isStreaming.value) return
  activeSessionId.value = id
  error.value = null
}

function addUserMessage(text) {
  const session = currentSession.value || createNewSession()

  if (session.messages.length === 0 && session.title === NEW_SESSION_TITLE) {
    session.title = normalizeTitle(text)
  }

  const msg = reactive({
    id: generateId(),
    role: 'user',
    content: text,
    time: getCurrentTime(),
  })

  session.messages.push(msg)
  touchSession(session)
  error.value = null
  return msg
}

function addAssistantPlaceholder() {
  const session = currentSession.value || createNewSession()
  const msg = reactive({
    id: generateId(),
    role: 'assistant',
    content: '',
    time: '',
    isThinking: true,
    isStreaming: false,
    retrieval: null,
    designId: null,
    designVersion: null,
  })

  session.messages.push(msg)
  touchSession(session)
  return msg
}

function prepareAssistantComputationResume() {
  const session = currentSession.value
  if (!session) return

  let assistant = null
  for (let i = session.messages.length - 1; i >= 0; i--) {
    if (session.messages[i].role === 'assistant') {
      assistant = session.messages[i]
      break
    }
  }
  if (!assistant) {
    assistant = reactive({
      id: generateId(),
      role: 'assistant',
      content: '',
      time: '',
      isThinking: true,
      isStreaming: false,
      retrieval: null,
      designId: null,
      designVersion: null,
    })
    session.messages.push(assistant)
  } else {
    // 后端会从第一个事件完整重放本轮计算，因此先清空本地半成品，避免重复拼接。
    assistant.content = ''
    assistant.time = ''
    assistant.isThinking = true
    assistant.isStreaming = false
    assistant.retrieval = null
  }
  error.value = null
  isStreaming.value = true
  touchSession(session)
}

function startAssistantRetrieval(message = '正在检索资料...') {
  const msgs = messages.value
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === 'assistant') {
      msgs[i].retrieval = {
        status: 'searching',
        message,
        documents: [],
        content: '',
        markdown: '',
      }
      touchSession()
      return
    }
  }
}

function setAssistantRetrievalResults(documents = [], markdown = '') {
  const msgs = messages.value
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === 'assistant') {
      const hasResult = documents.length || markdown
      msgs[i].retrieval = {
        status: 'done',
        message: hasResult ? '检索到以下资料' : '未检索到工程机械用钢知识库资料',
        documents,
        content: '',
        markdown,
      }
      touchSession()
      return
    }
  }
}

function setAssistantRetrievalStatus(message = '已检索参考资料') {
  const msgs = messages.value
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === 'assistant') {
      msgs[i].retrieval = {
        status: 'done',
        message,
        documents: [],
        content: '',
        markdown: '',
      }
      touchSession()
      return
    }
  }
}

function startAssistantRetrievalPreview(message = '正在生成材料设计初步方案...') {
  const msgs = messages.value
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === 'assistant') {
      msgs[i].retrieval = {
        status: 'streaming',
        message,
        documents: [],
        content: '',
        markdown: '',
      }
      touchSession()
      return
    }
  }
}

function appendAssistantRetrievalPreview(text) {
  const msgs = messages.value
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === 'assistant') {
      if (!msgs[i].retrieval) {
        msgs[i].retrieval = {
          status: 'streaming',
          message: '正在生成材料设计初步方案...',
          documents: [],
          content: '',
          markdown: '',
        }
      }
      msgs[i].retrieval.status = 'streaming'
      msgs[i].retrieval.content = `${msgs[i].retrieval.content || ''}${text}`
      // 每次阶段信息追加后同步刷新 markdown，避免等到 design_preview_done 才渲染。
      msgs[i].retrieval.markdown = msgs[i].retrieval.content
      touchSession()
      return
    }
  }
}

function appendAssistantReasoning(text) {
  const msgs = messages.value
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === 'assistant') {
      if (!msgs[i].retrieval) {
        msgs[i].retrieval = {
          status: 'streaming',
          message: '模型思维链',
          documents: [],
          content: '',
          markdown: '',
        }
      }
      msgs[i].retrieval.status = 'streaming'
      msgs[i].retrieval.content = `${msgs[i].retrieval.content || ''}${text}`
      msgs[i].retrieval.markdown = msgs[i].retrieval.content
      touchSession()
      return
    }
  }
}

function finishAssistantRetrievalPreview() {
  const msgs = messages.value
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === 'assistant' && msgs[i].retrieval) {
      const content = msgs[i].retrieval.content || ''
      msgs[i].retrieval.status = 'done'
      msgs[i].retrieval.message = '材料设计初步方案'
      msgs[i].retrieval.markdown = content
      touchSession()
      return
    }
  }
}

function appendToLastAssistant(text) {
  const msgs = messages.value
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === 'assistant') {
      if (msgs[i].isThinking) {
        msgs[i].isThinking = false
        msgs[i].isStreaming = true
      }
      msgs[i].content += text
      touchSession()
      return
    }
  }
}

function replaceLastAssistantContent(text) {
  const msgs = messages.value
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === 'assistant') {
      msgs[i].isThinking = false
      msgs[i].isStreaming = true
      msgs[i].content = text || ''
      touchSession()
      return
    }
  }
}

function setActiveDesignContext(payload = {}) {
  const session = currentSession.value
  if (!session || !payload.design_id) return
  session.activeDesignId = payload.design_id
  session.activeDesignVersion = payload.version || null

  const msgs = session.messages
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role !== 'assistant') continue
    msgs[i].designId = payload.design_id
    msgs[i].designVersion = payload.version || null
    break
  }
  touchSession(session)
}

function settleLastAssistantRetrieval(status = 'done', message = '') {
  const msgs = messages.value
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role !== 'assistant' || !msgs[i].retrieval) continue
    const retrieval = msgs[i].retrieval
    if (!['searching', 'streaming'].includes(retrieval.status)) return
    retrieval.status = status
    retrieval.message = message || (
      status === 'error' ? '计算已中断' : '模型处理过程已结束'
    )
    retrieval.markdown = retrieval.markdown || retrieval.content || ''
    touchSession()
    return
  }
}

function finishStreaming() {
  isStreaming.value = false
  settleLastAssistantRetrieval('done')

  const msgs = messages.value
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === 'assistant') {
      msgs[i].isThinking = false
      msgs[i].isStreaming = false
      msgs[i].time = getCurrentTime()
      touchSession()
      return
    }
  }
}

function setError(msg) {
  error.value = msg
  isStreaming.value = false
  settleLastAssistantRetrieval('error', '后端连接已断开，计算已停止')

  const msgs = messages.value
  const last = msgs[msgs.length - 1]
  if (last && last.role === 'assistant' && (last.content || last.retrieval)) {
    last.isThinking = false
    last.isStreaming = false
    last.content += `\n\n> 错误：${escapeHtml(msg)}`
    last.time = getCurrentTime()
    touchSession()
    return
  }

  if (last && last.role === 'assistant' && last.isThinking && !last.content) {
    last.isThinking = false
    last.isStreaming = false
    last.content = `<em style="color: #e11d48;">错误：${escapeHtml(msg)}</em>`
    last.time = getCurrentTime()
    touchSession()
  }
}

function clearError() {
  error.value = null
}

function resendMessage(msgId) {
  const msgs = messages.value
  const idx = msgs.findIndex(m => m.id === msgId)
  if (idx === -1) return null

  const userMsg = msgs[idx]
  const text = userMsg.content

  msgs.splice(idx)
  // 回退到较早消息重发时，同步恢复该位置之前最近的成功设计版本，避免
  // 把已被删除的后续方案UUID继续作为“当前方案”传给后端。
  const remainingDesignMessage = [...msgs]
    .reverse()
    .find(message => message.role === 'assistant' && message.designId)
  const session = currentSession.value
  if (session) {
    session.activeDesignId = remainingDesignMessage?.designId || null
    session.activeDesignVersion = remainingDesignMessage?.designVersion || null
  }
  touchSession()
  error.value = null
  isStreaming.value = false
  return text
}

watch(
  [sessions, activeSessionId],
  persistState,
  // 初始化时立即把已规范化的中断状态写回，避免下次刷新再次读到
  // isThinking/isStreaming=true 的旧数据。
  { deep: true, immediate: true }
)

export function useChatStore() {
  return {
    sessionId,
    activeDesignId,
    activeSessionId,
    sessions,
    historyItems,
    messages,
    isStreaming,
    error,
    showWelcome,
    maxHistoryItems: MAX_HISTORY_ITEMS,

    createNewSession,
    selectSession,
    addUserMessage,
    addAssistantPlaceholder,
    prepareAssistantComputationResume,
    appendToLastAssistant,
    startAssistantRetrieval,
    setAssistantRetrievalResults,
    setAssistantRetrievalStatus,
    startAssistantRetrievalPreview,
    appendAssistantRetrievalPreview,
    appendAssistantReasoning,
    finishAssistantRetrievalPreview,
    finishStreaming,
    setError,
    clearError,
    resendMessage,
    replaceLastAssistantContent,
    setActiveDesignContext,
  }
}

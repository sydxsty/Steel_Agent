<script setup>
/**
 * ChatView.vue — 聊天主视图组件
 * ==============================
 * 功能 / Features:
 *   - 聊天应用的核心编排组件
 *   - 管理消息状态和流式 API 调用的完整生命周期
 *   - 协调 WelcomeState、MessageList、ChatInput 三个子组件
 *
 * 数据流 / Data Flow:
 *   ChatInput (@send)        → handleSend() → fetch /chat → ReadableStream
 *   WelcomeState (@select-prompt) → handleSend()
 *   MessageList (@resend)     → handleResend() → handleSend()
 *
 * 流式处理流程 / Streaming Flow:
 *   1. addUserMessage(text)           — 立即显示用户消息
 *   2. addAssistantPlaceholder()      — 显示三点思考动画
 *   3. fetch POST /chat               — 发起流式请求
 *   4. response.body.getReader()      — 获取 ReadableStream reader
 *   5. 逐行解析 NDJSON                — 按 \n 分割 + JSON.parse
 *   6. appendToLastAssistant(content) — 实时追加文本
 *   7. finishStreaming()              — 完成流式，切换到 markdown 渲染
 */

import { nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import WelcomeState from './WelcomeState.vue'
import MessageList from './MessageList.vue'
import ChatInput from './ChatInput.vue'
import { useChatStore } from '../../composables/useChatStore.js'

// ============================================================
// 从 Store 获取响应式状态和操作方法
// ============================================================
const {
  sessionId,
  activeDesignId,
  messages,
  isStreaming,
  showWelcome,
  addUserMessage,
  addAssistantPlaceholder,
  prepareAssistantComputationResume,
  appendToLastAssistant,
  replaceLastAssistantContent,
  setActiveDesignContext,
  startAssistantRetrieval,
  setAssistantRetrievalResults,
  setAssistantRetrievalStatus,
  startAssistantRetrievalPreview,
  appendAssistantRetrievalPreview,
  appendAssistantReasoning,
  finishAssistantRetrievalPreview,
  finishStreaming,
  setError,
  resendMessage,
} = useChatStore()

// ============================================================
// 常量 / Constants
// ============================================================

/**
 * 后端 API 基础 URL
 * 开发阶段指向本地 FastAPI 服务器（端口 8000）
 */
const API_BASE_URL = 'http://localhost:8000'
const BACKEND_HEALTHCHECK_INTERVAL_MS = 5000
const BACKEND_HEALTHCHECK_TIMEOUT_MS = 4000
const BACKEND_HEALTHCHECK_FAILURE_LIMIT = 2
// 后端每15秒发送一次心跳。超过45秒未收到任何字节，说明浏览器最小化后
// 当前 ReadableStream 很可能已经成为“假活连接”，需要从后端缓存完整重放。
const STREAM_STALE_TIMEOUT_MS = 45000
const RECONNECT_RELEASE_TIMEOUT_MS = 5000

let activeRequestController = null
let healthMonitorIntervalId = null
let healthMonitorProbing = false
let healthMonitorFailures = 0
let resumeInProgress = false
let detachedByHealthMonitor = false
let reconnectInProgress = false
let componentUnmounting = false
let lastStreamActivityAt = 0
const plannedReconnectControllers = new WeakSet()
const designReferenceDialog = ref(null)

function markStreamActivity() {
  lastStreamActivityAt = Date.now()
}

function isStreamSubscriptionStale() {
  return Boolean(
    activeRequestController
    && lastStreamActivityAt > 0
    && Date.now() - lastStreamActivityAt >= STREAM_STALE_TIMEOUT_MS,
  )
}

async function waitForSubscriptionRelease(controller) {
  const deadline = Date.now() + RECONNECT_RELEASE_TIMEOUT_MS
  while (activeRequestController === controller && Date.now() < deadline) {
    await new Promise(resolve => setTimeout(resolve, 50))
  }
  return activeRequestController !== controller
}

async function fetchComputationStatus(targetSessionId = sessionId.value) {
  if (!targetSessionId) return { exists: false, status: 'idle' }
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), BACKEND_HEALTHCHECK_TIMEOUT_MS)
  try {
    const response = await fetch(
      `${API_BASE_URL}/computation-status/${encodeURIComponent(targetSessionId)}`,
      { method: 'GET', cache: 'no-store', signal: controller.signal },
    )
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    return await response.json()
  } finally {
    clearTimeout(timeoutId)
  }
}

// ============================================================
// 方法 / Methods
// ============================================================

/**
 * 判断后端是否返回了工程机械用钢的实绩匹配对象。
 * 这类结果不是普通流式 content，而是包含 isState/arrBody 的完整 JSON。
 *
 * @param {object} data - NDJSON 单行解析后的对象
 * @returns {boolean} 是否为 Oracle 实绩匹配结果
 */
function isProcessMatchResult(data) {
  return data
    && typeof data === 'object'
    && Object.prototype.hasOwnProperty.call(data, 'isState')
    && Array.isArray(data.arrBody)
}

/**
 * 将 Oracle 实绩匹配对象转成 Markdown 文本，保证消息气泡能正常显示。
 * 同时保留完整 JSON，方便复制给后续接口或人工排查。
 *
 * @param {object} data - 后端 match_engineering_steel_process 返回对象
 * @returns {string} 可渲染的 Markdown 文本
 */
function formatProcessMatchResult(data) {
  const hasRow = data.arrBody.length > 0
  const status = data.isState
    ? '严格条件匹配成功'
    : hasRow
      ? '放宽条件后匹配成功'
      : '未查询到匹配实绩数据'

  const lines = [
    '### 工程机械用钢匹配结果',
    `- 匹配状态：${status}`,
    `- 钢卷号：${data.strCoil || '-'}`,
    `- 钢种：${data.strSteel || '-'}`,
    `- 会话：${data.session_key || '-'}`,
  ]

  if (data.message) {
    lines.push(`- 说明：${data.message}`)
  }

  if (data.error) {
    lines.push(`- 错误：${data.error}`)
  }

  lines.push('', '```json', JSON.stringify(data, null, 2), '```')
  return `${lines.join('\n')}\n`
}

/**
 * 统一处理后端 NDJSON 单行数据。
 * 兼容普通 content 流、分类提示、规格 JSON 和 Oracle 匹配 JSON。
 *
 * @param {object} parsed - NDJSON 单行解析后的对象
 */
async function flushRetrievalPaint() {
  await nextTick()
  await new Promise(resolve => {
    if (typeof requestAnimationFrame === 'function') {
      requestAnimationFrame(() => resolve())
    } else {
      setTimeout(resolve, 0)
    }
  })
}

async function handleParsedResponse(parsed) {
  if (parsed.event === 'design_reference_required') {
    designReferenceDialog.value = {
      message: parsed.message || '请选择本轮需要修改的历史设计方案。',
      candidates: Array.isArray(parsed.candidates) ? parsed.candidates : [],
      originalPrompt: parsed.original_prompt || '',
      resumeToken: parsed.resume_token || '',
    }
    await flushRetrievalPaint()
    return
  }

  if (parsed.event === 'design_context') {
    // UUID只在会话状态和下一次请求中使用；页面仅展示简短的方案V序号。
    setActiveDesignContext(parsed)
    await flushRetrievalPaint()
    return
  }

  if (parsed.event === 'search_start') {
    startAssistantRetrieval(parsed.message || '正在检索资料...')
    await flushRetrievalPaint()
    return
  }

  if (parsed.event === 'retrieval_result') {
    setAssistantRetrievalResults(parsed.documents || [], parsed.markdown || '')
    await flushRetrievalPaint()
    return
  }

  if (parsed.event === 'search_done') {
    setAssistantRetrievalStatus(parsed.message || '已检索参考资料，正在生成初步方案...')
    await flushRetrievalPaint()
    return
  }

  if (parsed.event === 'design_preview_start') {
    startAssistantRetrievalPreview(parsed.message || '正在生成材料设计初步方案...')
    await flushRetrievalPaint()
    return
  }

  if (parsed.event === 'design_preview_delta') {
    if (parsed.content) {
      appendAssistantRetrievalPreview(parsed.content)
      await flushRetrievalPaint()
    }
    return
  }

  if (parsed.event === 'design_preview_done') {
    finishAssistantRetrievalPreview()
    await flushRetrievalPaint()
    return
  }

  if (parsed.event === 'answer_start') {
    return
  }

  // 后端长计算期间的保活事件，不进入用户可见内容。
  if (parsed.event === 'heartbeat') {
    return
  }

  if (parsed.event === 'answer_delta') {
    if (parsed.content) {
      appendToLastAssistant(parsed.content)
    }
    return
  }

  if (parsed.event === 'answer_replace') {
    replaceLastAssistantContent(parsed.content || '')
    return
  }

  if (parsed.event === 'answer_done') {
    return
  }

  if (parsed.event === 'error') {
    appendToLastAssistant(`\n\n> 错误：${parsed.message || parsed.error || '请求失败'}`)
    return
  }

  // 工程机械用钢实绩匹配结果没有 content 字段，需要主动格式化显示。
  if (isProcessMatchResult(parsed)) {
    appendToLastAssistant(formatProcessMatchResult(parsed))
    return
  }

  // 普通错误流仍按错误处理，交给 setError 渲染。
  if (parsed.error) {
    appendToLastAssistant(`\n\n> 错误：${parsed.error}`)
    return
  }

  // 思维链内容（来自 /chat 端点的 reasoning_content 透传）
  if (parsed.reasoning || parsed.reasoning_start) {
    appendAssistantReasoning(parsed.reasoning || '')
    await flushRetrievalPaint()
    return
  }

  // 意图分类结果 (来自 /classify 端点)
  if (parsed.intent) {
    console.log('[意图分类]', parsed.intent)
    appendToLastAssistant(`[意图: ${parsed.intent}] `)
  }

  // 钢材用途分类结果 (来自二级分类)
  if (parsed.purpose) {
    console.log('[用途分类]', parsed.purpose)
    appendToLastAssistant(`[用途: ${parsed.purpose}] `)
  }

  // 钢材规格提取结果 (含 用途/C_max/C_min 等字段)
  if (parsed["用途"] && !parsed.purpose) {
    console.log('[钢规格]', parsed)
    appendToLastAssistant(JSON.stringify(parsed, null, 2))
  }

  // 流式内容块 (来自 /chat 端点)
  if (parsed.content) {
    appendToLastAssistant(parsed.content)
  }
}

async function consumeComputationResponse(response) {
  if (!response.body) {
    throw new Error('后端未返回可读取的流式响应')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let answerDoneReceived = false
  markStreamActivity()
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      markStreamActivity()
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop()
      for (const line of lines) {
        if (!line.trim()) continue
        try {
          const parsed = JSON.parse(line)
          if (parsed.event === 'answer_done') answerDoneReceived = true
          await handleParsedResponse(parsed)
        } catch (parseErr) {
          console.warn('[NDJSON] 解析行失败:', line, parseErr)
        }
      }
    }

    if (buffer.trim()) {
      try {
        const parsed = JSON.parse(buffer)
        if (parsed.event === 'answer_done') answerDoneReceived = true
        await handleParsedResponse(parsed)
      } catch {
        // 忽略连接结束时残留的不完整 NDJSON 行。
      }
    }
    return answerDoneReceived
  } finally {
    try {
      reader.releaseLock()
    } catch {
      // 流异常关闭时 reader 可能已自动释放。
    }
  }
}

async function resumeRunningComputation(targetSessionId = sessionId.value) {
  if (
    !targetSessionId
    || componentUnmounting
    || resumeInProgress
    || activeRequestController
  ) return
  resumeInProgress = true
  const controller = new AbortController()
  activeRequestController = controller
  markStreamActivity()
  prepareAssistantComputationResume()

  try {
    const response = await fetch(
      `${API_BASE_URL}/computation-stream/${encodeURIComponent(targetSessionId)}?from_event=0`,
      { method: 'GET', cache: 'no-store', signal: controller.signal },
    )
    if (!response.ok) throw new Error(`恢复计算流失败 (${response.status})`)
    const answerDoneReceived = await consumeComputationResponse(response)
    if (!answerDoneReceived) {
      throw new Error('后端连接在计算完成前中断')
    }
  } catch (err) {
    const plannedReconnect = plannedReconnectControllers.has(controller)
    if (err?.name !== 'AbortError' && !plannedReconnect) {
      console.error('[ChatView] 恢复计算流失败:', err)
      setError(err.message || '恢复计算连接失败')
    }
  } finally {
    if (activeRequestController === controller) {
      activeRequestController = null
    }
    resumeInProgress = false
    finishStreaming()
  }
}

/**
 * 浏览器最小化或后台冻结后，旧 fetch 可能既不报错也不再收包。
 * 此处先查询后台任务，再只断开网页订阅并从事件0完整重放；Store 会先清空
 * 当前半成品，所以不会重复拼接，后台计算任务本身不会被取消。
 */
async function recoverComputationSubscription({ force = false } = {}) {
  if (componentUnmounting || reconnectInProgress || resumeInProgress) return
  if (!sessionId.value || (!activeRequestController && !isStreaming.value)) return

  reconnectInProgress = true
  try {
    const targetSessionId = sessionId.value
    const status = await fetchComputationStatus(targetSessionId)
    if (!status.exists || !['running', 'completed'].includes(status.status)) return

    const currentController = activeRequestController
    const needsReplay = Boolean(currentController || isStreaming.value)
    const shouldReplaceActive = Boolean(
      currentController && (force || isStreamSubscriptionStale()),
    )
    if (shouldReplaceActive) {
      plannedReconnectControllers.add(currentController)
      detachedByHealthMonitor = true
      currentController.abort()
      const released = await waitForSubscriptionRelease(currentController)
      if (!released) {
        console.warn('[ChatView] 旧计算流未及时释放，本轮稍后继续重试')
        return
      }
    }

    if (!activeRequestController && (status.status === 'running' || needsReplay)) {
      // 不等待整个长计算完成，避免占住健康检查锁；resumeInProgress 会防止重复订阅。
      void resumeRunningComputation(targetSessionId)
    }
  } catch (err) {
    if (err?.name !== 'AbortError') {
      console.warn('[ChatView] 计算流恢复检查失败:', err)
    }
  } finally {
    reconnectInProgress = false
  }
}

async function probeBackendAndComputation() {
  if (healthMonitorProbing) return
  healthMonitorProbing = true
  try {
    const targetSessionId = sessionId.value
    const status = await fetchComputationStatus(targetSessionId)
    healthMonitorFailures = 0
    if (status.status === 'running' && !activeRequestController) {
      void resumeRunningComputation(targetSessionId)
    } else if (
      ['running', 'completed'].includes(status.status)
      && activeRequestController
      && isStreamSubscriptionStale()
    ) {
      await recoverComputationSubscription({ force: true })
    } else if (
      status.status === 'completed'
      && !activeRequestController
      && isStreaming.value
    ) {
      void resumeRunningComputation(targetSessionId)
    }
  } catch {
    healthMonitorFailures += 1
    if (
      healthMonitorFailures >= BACKEND_HEALTHCHECK_FAILURE_LIMIT
      && activeRequestController
    ) {
      // 只停止当前网页订阅。计算任务由后端独立持有，不会随连接取消。
      detachedByHealthMonitor = true
      activeRequestController.abort()
    }
  } finally {
    healthMonitorProbing = false
  }
}

function handlePageVisibilityRecovery() {
  if (document.visibilityState !== 'visible') return
  // visibilitychange 在浏览器从最小化/后台标签恢复时触发。即使旧请求对象
  // 仍存在也主动更换订阅，解决 activeRequestController 阻止自动恢复的问题。
  recoverComputationSubscription({ force: true })
}

function handlePageShowRecovery() {
  recoverComputationSubscription({ force: true })
}

function startBackendHealthMonitor() {
  if (healthMonitorIntervalId) return
  probeBackendAndComputation()
  healthMonitorIntervalId = setInterval(
    probeBackendAndComputation,
    BACKEND_HEALTHCHECK_INTERVAL_MS,
  )
}

/**
 * 处理发送消息 — 完整的 send → stream → render 流程
 * Handle sending a message — complete send → stream → render pipeline
 *
 * 这是聊天应用最核心的函数，负责：
 *   1. 添加用户消息和 AI 占位消息到界面
 *   2. 通过 fetch 发起流式请求到后端 /chat API
 *   3. 使用 ReadableStream 逐块读取 NDJSON 响应
 *   4. 实时追加每个 token 到 AI 消息气泡
 *   5. 处理完成、网络错误、HTTP 错误等各种情况
 *
 * @param {string|object} payload - 文本，或输入框提交的文本、附件ID和接收回调
 */
async function handleSend(payload) {
  const text = typeof payload === 'string' ? payload : String(payload?.text || '')
  const attachmentIds = typeof payload === 'object' && Array.isArray(payload?.attachmentIds)
    ? payload.attachmentIds
    : []
  const onAccepted = typeof payload === 'object' && typeof payload?.onAccepted === 'function'
    ? payload.onAccepted
    : null
  const reuseExistingMessage = typeof payload === 'object' && Boolean(payload?.reuseExistingMessage)
  const referenceDesignId = typeof payload === 'object'
    ? String(payload?.referenceDesignId || '')
    : ''
  const referenceResumeToken = typeof payload === 'object'
    ? String(payload?.referenceResumeToken || '')
    : ''
  // 防止空消息和重复发送 / Guard against empty message and duplicate sends
  if (!text.trim() || isStreaming.value) return

  // ==========================================================
  // 步骤 1: 添加用户消息到界面
  // Step 1: Add user message to the UI
  // ==========================================================
  if (!reuseExistingMessage) {
    addUserMessage(text)
  }

  // ==========================================================
  // 步骤 2: 添加 AI 占位消息（显示三点思考动画）
  // Step 2: Add AI placeholder message (shows thinking dots)
  // ==========================================================
  if (reuseExistingMessage) {
    prepareAssistantComputationResume()
  } else {
    addAssistantPlaceholder()
  }

  // ==========================================================
  // 步骤 3: 标记流式生成开始（禁用输入框）
  // Step 3: Mark streaming as started (disables input)
  // ==========================================================
  isStreaming.value = true

  const requestController = new AbortController()
  activeRequestController = requestController
  detachedByHealthMonitor = false
  markStreamActivity()
  let reader = null
  let answerDoneReceived = false

  try {
    // ==========================================================
    // 步骤 4: 发起 POST 请求到 /chat
    // Step 4: Initiate POST request to /chat
    // ==========================================================
    const response = await fetch(`${API_BASE_URL}/classify`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: text,
        session_id: sessionId.value,
        active_design_id: activeDesignId.value || null,
        reference_design_id: referenceDesignId || null,
        reference_resume_token: referenceResumeToken || null,
        attachment_ids: attachmentIds,
      }),
      signal: requestController.signal,
    })

    // 检查 HTTP 状态码 — 非 2xx 表示请求失败
    // Check HTTP status — non-2xx means request failed
    if (!response.ok) {
      let errorMsg = `服务器错误 (${response.status})`
      try {
        // 尝试解析错误响应体中的详细错误信息
        // Try to parse detailed error message from response body
        const errData = await response.json()
        errorMsg = errData.error || errorMsg
      } catch {
        // 响应体不是 JSON（如 502 Bad Gateway 的 HTML 页面）
        // Response body is not JSON (e.g., 502 HTML page)
      }
      throw new Error(errorMsg)
    }

    // 服务端已将附件Markdown复制进当前后台任务，此时清除当前轮附件草稿。
    onAccepted?.()

    // ==========================================================
    // 步骤 5: 使用 ReadableStream 逐块读取流式响应
    // Step 5: Read streaming response chunk by chunk via ReadableStream
    // ==========================================================
    // response.body 是一个 ReadableStream<Uint8Array>
    // getReader() 返回一个 reader，用于逐块读取二进制数据
    if (!response.body) {
      throw new Error('后端未返回可读取的流式响应')
    }

    reader = response.body.getReader()
    // TextDecoder 用于将 Uint8Array 字节数据解码为 UTF-8 字符串
    // { stream: true } 参数确保多字节字符（如中文）在跨 chunk 边界时被正确处理
    const decoder = new TextDecoder('utf-8')
    // 缓冲区：存储不完整的行（可能在 chunk 边界处被截断）
    // Buffer: holds incomplete lines (may be truncated at chunk boundaries)
    let buffer = ''

    while (true) {
      // reader.read() 返回 { done: boolean, value: Uint8Array }
      // done=true 表示流已结束
      const { done, value } = await reader.read()
      if (done) break
      markStreamActivity()

      // 将本次读取的字节数据解码为文本并追加到缓冲区
      // Decode current chunk's bytes to text and append to buffer
      buffer += decoder.decode(value, { stream: true })

      // ==========================================================
      // 步骤 6: 按行分割缓冲区，逐行解析 NDJSON
      // Step 6: Split buffer by newline, parse NDJSON line by line
      // ==========================================================
      // NDJSON 格式：每行一个完整的 JSON 对象，以 \n 分隔
      // 最后一行可能不完整（被 chunk 边界截断），保留到下次循环
      const lines = buffer.split('\n')
      // 最后一行可能不完整，保留在 buffer 中
      // Last line may be incomplete, keep it in buffer for next iteration
      buffer = lines.pop()

      for (const line of lines) {
        // 跳过空行 / Skip empty lines
        if (!line.trim()) continue

        try {
          const parsed = JSON.parse(line)
          if (parsed.event === 'answer_done') answerDoneReceived = true
          await handleParsedResponse(parsed)
        } catch (parseErr) {
          // JSON 解析失败（不应该发生，防御性处理）
          // JSON parse failure (shouldn't happen, defensive handling)
          console.warn('[NDJSON] 解析行失败:', line, parseErr)
        }
      }
    }

    // ==========================================================
    // 步骤 8: 处理缓冲区中剩余的数据
    // Step 8: Process any remaining data in buffer
    // ==========================================================
    // 流结束后，buffer 中可能还有最后一整行数据
    if (buffer.trim()) {
      try {
        const parsed = JSON.parse(buffer)
        if (parsed.event === 'answer_done') answerDoneReceived = true
        await handleParsedResponse(parsed)
      } catch {
        // 忽略最后不完整的数据 / Ignore remaining incomplete data
      }
    }

    if (!answerDoneReceived) {
      throw new Error('后端连接在计算完成前中断')
    }

    // ==========================================================
    // 步骤 9: 完成流式生成
    // Step 9: Finish streaming
    // ==========================================================
    // 将 isStreaming 设为 false，设置时间戳，
    // MessageBubble 切换到 v-html markdown 渲染模式
  } catch (err) {
    // ==========================================================
    // 错误处理 / Error Handling
    // ==========================================================
    // 可能的错误场景：
    //   - 网络断开（fetch 抛出 TypeError）
    //   - 服务器返回非 2xx 状态码
    //   - NDJSON 行中包含 error 字段
    //   - 其他未知异常
    const plannedReconnect = plannedReconnectControllers.has(requestController)
    if (!plannedReconnect) {
      console.error('[ChatView] 流式请求失败:', err)
    }

    if (!plannedReconnect) {
      const errorMessage = err?.name === 'AbortError'
        ? (detachedByHealthMonitor
          ? '前端连接暂时中断；后台任务不受影响，连接恢复后将自动继续显示。'
          : '后端服务已关闭或连接中断，已保留上次显示内容。')
        : (err.message || '连接失败，请检查网络后重试')
      setError(errorMessage)
    }
  } finally {
    if (reader) {
      try {
        reader.releaseLock()
      } catch {
        // 流已异常关闭时 reader 可能已经自动释放。
      }
    }
    if (activeRequestController === requestController) {
      activeRequestController = null
    }
    finishStreaming()
  }
  // 注意：无论成功、断流或失败，finally 都会统一结束消息和计算步骤的加载状态。
  // 这确保了输入框在流式结束后恢复可用
}

function confirmDesignReference(candidate) {
  const dialog = designReferenceDialog.value
  if (!dialog || isStreaming.value || !candidate?.design_id) return
  designReferenceDialog.value = null
  handleSend({
    text: dialog.originalPrompt,
    referenceDesignId: candidate.design_id,
    referenceResumeToken: dialog.resumeToken,
    reuseExistingMessage: true,
  })
}

function cancelDesignReference() {
  if (isStreaming.value) return
  designReferenceDialog.value = null
  appendToLastAssistant('\n\n> 已取消本次历史方案续改。')
}

/**
 * 处理来自 WelcomeState 的建议提示点击
 * Handle suggested prompt click from WelcomeState
 *
 * @param {string} text - 选中的提示文本
 */
function handleSelectPrompt(text) {
  handleSend(text)
}

/**
 * 处理来自 MessageBubble（通过 MessageList）的重发请求
 * Handle resend request from MessageBubble (via MessageList)
 *
 * 流程:
 *   1. 通过 resendMessage() 删除该消息及后续所有消息
 *   2. 用相同的文本重新调用 handleSend()
 *
 * @param {string} msgId - 要重发的（用户）消息 ID
 */
function handleResend(msgId) {
  const text = resendMessage(msgId)
  if (text) {
    // 使用相同的文本重新发送 / Re-send with the same text
    handleSend(text)
  }
}

onMounted(() => {
  componentUnmounting = false
  startBackendHealthMonitor()
  document.addEventListener('visibilitychange', handlePageVisibilityRecovery)
  window.addEventListener('pageshow', handlePageShowRecovery)
  window.addEventListener('online', handlePageShowRecovery)
})

onBeforeUnmount(() => {
  componentUnmounting = true
  document.removeEventListener('visibilitychange', handlePageVisibilityRecovery)
  window.removeEventListener('pageshow', handlePageShowRecovery)
  window.removeEventListener('online', handlePageShowRecovery)
  if (healthMonitorIntervalId) {
    clearInterval(healthMonitorIntervalId)
    healthMonitorIntervalId = null
  }
  // 这里只断开当前网页订阅；后端独立计算任务继续执行并缓存输出。
  activeRequestController?.abort()
  activeRequestController = null
})

</script>

<template>
  <div class="chat-view">
    <!--
      欢迎页面：消息列表为空时显示
      Welcome page: shown when no messages exist
      点击建议提示 → @select-prompt → handleSelectPrompt → handleSend
    -->
    <WelcomeState
      v-if="showWelcome"
      @select-prompt="handleSelectPrompt"
    />

    <!--
      消息列表：有消息时显示
      Message list: shown when messages exist
      - :messages 传入响应式消息数组
      - :is-streaming 传入流式状态（触发持续滚动）
      - @resend 处理重发请求
    -->
    <MessageList
      v-else
      :key="sessionId"
      :messages="messages"
      :is-streaming="isStreaming"
      @resend="handleResend"
    />

    <!--
      聊天输入框：始终显示
      Chat input: always visible
      - :disabled 流式生成期间禁用输入
      - @send 处理发送事件
    -->
    <ChatInput
      :disabled="isStreaming"
      :session-id="sessionId"
      @send="handleSend"
    />

    <div v-if="designReferenceDialog" class="design-reference-backdrop">
      <section class="design-reference-dialog" role="dialog" aria-modal="true" aria-labelledby="design-reference-title">
        <header>
          <h2 id="design-reference-title">选择历史设计方案</h2>
          <button type="button" class="dialog-close" aria-label="取消选择" @click="cancelDesignReference">×</button>
        </header>
        <p>{{ designReferenceDialog.message }}</p>
        <div class="design-reference-list">
          <button
            v-for="candidate in designReferenceDialog.candidates"
            :key="candidate.design_id"
            type="button"
            class="design-reference-option"
            :disabled="isStreaming"
            @click="confirmDesignReference(candidate)"
          >
            <strong>方案{{ candidate.version || '-' }}</strong>
            <span>{{ candidate.grade || '未标注牌号' }}</span>
            <span>成品 {{ candidate.aim_thick ?? '-' }} mm</span>
            <span>板坯 {{ candidate.slab_thick ?? '-' }} mm</span>
          </button>
        </div>
        <p v-if="!designReferenceDialog.candidates.length" class="dialog-empty">
          当前会话没有可选择的成功设计，请重新发送完整设计条件。
        </p>
      </section>
    </div>

    <!--
      连接状态提示（开发调试用）
      Connection status hint (for dev debugging)
      当后端不可达时在 fetch 中会被 catch，不需要额外 UI
    -->
  </div>
</template>

<style scoped>
.chat-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--bg-main);
  overflow: hidden;
}

.design-reference-backdrop {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: grid;
  place-items: center;
  padding: 20px;
  background: rgba(20, 24, 32, 0.38);
}

.design-reference-dialog {
  width: min(560px, 100%);
  max-height: min(680px, 86vh);
  overflow: auto;
  padding: 20px;
  border: 1px solid var(--border-color, #dfe3ea);
  border-radius: 8px;
  background: var(--bg-main, #fff);
  box-shadow: 0 18px 48px rgba(20, 24, 32, 0.18);
}

.design-reference-dialog header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.design-reference-dialog h2 {
  margin: 0;
  font-size: 18px;
  letter-spacing: 0;
}

.design-reference-dialog p {
  margin: 12px 0;
  color: var(--text-secondary, #5d6573);
}

.dialog-close {
  width: 32px;
  height: 32px;
  border: 0;
  background: transparent;
  font-size: 24px;
  line-height: 1;
  cursor: pointer;
}

.design-reference-list {
  display: grid;
  gap: 8px;
}

.design-reference-option {
  display: grid;
  grid-template-columns: minmax(80px, auto) repeat(3, minmax(0, 1fr));
  align-items: center;
  gap: 12px;
  width: 100%;
  min-height: 52px;
  padding: 10px 12px;
  border: 1px solid var(--border-color, #dfe3ea);
  border-radius: 6px;
  background: var(--bg-main, #fff);
  color: inherit;
  text-align: left;
  cursor: pointer;
}

.design-reference-option:hover {
  border-color: #6b5ce7;
  background: #f7f6ff;
}

.design-reference-option span {
  min-width: 0;
  overflow-wrap: anywhere;
}

.dialog-empty {
  padding: 12px;
  border: 1px solid var(--border-color, #dfe3ea);
}

@media (max-width: 640px) {
  .design-reference-option {
    grid-template-columns: 1fr 1fr;
  }
}
</style>

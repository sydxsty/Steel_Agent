<script setup>
import { useChatStore } from '../../composables/useChatStore.js'

const {
  activeSessionId,
  historyItems,
  isStreaming,
  createNewSession,
  selectSession,
} = useChatStore()
</script>

<template>
  <div class="chat-history">
    <div class="history-header">
      <span class="history-title">对话历史</span>
      <button
        class="new-chat-btn"
        title="新对话"
        :disabled="isStreaming"
        @click="createNewSession"
      >
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <line x1="12" y1="5" x2="12" y2="19" />
          <line x1="5" y1="12" x2="19" y2="12" />
        </svg>
      </button>
    </div>

    <div class="history-list">
      <div
        v-for="item in historyItems"
        :key="item.id"
        class="history-item"
        :class="{ active: item.id === activeSessionId }"
        @click="selectSession(item.id)"
      >
        <svg
          class="item-icon"
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path
            d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"
          />
        </svg>
        <span class="item-title">{{ item.title }}</span>
        <span class="item-date">{{ item.date }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat-history {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.history-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-md) var(--space-lg);
}

.history-title {
  font-size: var(--font-size-xs);
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.new-chat-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  transition: all 0.2s ease;
}

.new-chat-btn:hover:not(:disabled) {
  background: var(--hover-bg);
  color: var(--primary);
}

.new-chat-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.history-list {
  flex: 1;
  overflow-y: auto;
  padding: 0 var(--space-sm);
}

.history-item {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: 0 var(--space-md);
  height: 40px;
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all 0.15s ease;
}

.history-item:hover {
  background: var(--hover-bg);
}

.history-item.active {
  background: var(--active-bg);
}

.history-item.active .item-icon,
.history-item.active .item-title {
  color: var(--primary);
}

.item-icon {
  color: var(--text-secondary);
  flex-shrink: 0;
}

.item-title {
  flex: 1;
  font-size: var(--font-size-sm);
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.item-date {
  font-size: var(--font-size-xs);
  color: var(--text-secondary);
  flex-shrink: 0;
}
</style>

<script setup>
import BotInfo from '../sidebar/BotInfo.vue'
import ChatHistory from '../sidebar/ChatHistory.vue'
</script>

<template>
  <aside class="app-sidebar">
    <BotInfo />
    <ChatHistory />
  </aside>
</template>

<style scoped>
.app-sidebar {
  width: var(--sidebar-width);
  height: 100%;
  display: flex;
  flex-direction: column;
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border-color);
  overflow: hidden;
}
</style>

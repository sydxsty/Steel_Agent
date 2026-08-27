<script setup>
import AppHeader from './components/layout/AppHeader.vue'
import AppSidebar from './components/layout/AppSidebar.vue'
import ChatView from './components/chat/ChatView.vue'
</script>

<template>
  <div class="app-container">
    <AppHeader />
    <div class="main-layout">
      <AppSidebar />
      <ChatView />
    </div>
  </div>
</template>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

.main-layout {
  display: grid;
  grid-template-columns: var(--sidebar-width) 1fr;
  flex: 1;
  overflow: hidden;
}
</style>

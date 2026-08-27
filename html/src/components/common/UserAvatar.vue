<script setup>
defineProps({
  size: { type: Number, default: 32 },
  role: { type: String, default: 'user' },
})
</script>

<template>
  <div
    class="user-avatar"
    :class="role"
    :style="{ width: size + 'px', height: size + 'px' }"
  >
    <img
      v-if="role === 'bot'"
      src="/DSL图标_透明.ico"
      :width="Math.round(size * 0.7)"
      :height="Math.round(size * 0.7)"
      alt="DSL"
      class="bot-image"
    />
    <svg
      v-else
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="white"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  </div>
</template>

<style scoped>
.user-avatar {
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--radius-round);
  flex-shrink: 0;
}
.user-avatar.bot {
  background: #F4F2FF;
}
.bot-image {
  object-fit: contain;
}
.user-avatar.user {
  background: linear-gradient(135deg, #667eea, #764ba2);
}
</style>

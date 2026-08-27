<script setup>
import UserAvatar from '../common/UserAvatar.vue'
import { botInfo } from '../../data/mock.js'
</script>

<template>
  <div class="bot-info">
    <UserAvatar :size="48" role="bot" />
    <h2 class="bot-name">{{ botInfo.name }}</h2>
    <p class="bot-desc">{{ botInfo.description }}</p>
  </div>
</template>

<style scoped>
.bot-info {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: var(--space-xl) var(--space-lg);
  border-bottom: 1px solid var(--border-color);
  text-align: center;
}

.bot-name {
  font-size: var(--font-size-lg);
  font-weight: 600;
  color: var(--text-primary);
  margin-top: var(--space-md);
}

.bot-desc {
  font-size: var(--font-size-sm);
  color: var(--text-secondary);
  margin-top: var(--space-sm);
  line-height: 1.5;
}
</style>

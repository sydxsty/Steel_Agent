<script setup>
const links = [
  {
    label: '创建 Bot',
    icon: 'plus-circle',
  },
  {
    label: 'Bot 商店',
    icon: 'grid',
  },
  {
    label: '帮助中心',
    icon: 'help',
  },
]
</script>

<template>
  <div class="sidebar-footer">
    <a
      v-for="link in links"
      :key="link.label"
      class="footer-link"
      href="#"
    >
      <svg
        v-if="link.icon === 'plus-circle'"
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="16" />
        <line x1="8" y1="12" x2="16" y2="12" />
      </svg>
      <svg
        v-else-if="link.icon === 'grid'"
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </svg>
      <svg
        v-else
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        <circle cx="12" cy="12" r="10" />
        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
      </svg>
      <span>{{ link.label }}</span>
    </a>
  </div>
</template>

<style scoped>
.sidebar-footer {
  padding: var(--space-md) var(--space-lg);
  border-top: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.footer-link {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-sm) var(--space-md);
  border-radius: var(--radius-md);
  font-size: var(--font-size-sm);
  color: var(--text-secondary);
  transition: all 0.2s ease;
}

.footer-link:hover {
  background: var(--hover-bg);
  color: var(--primary);
}
</style>

<script setup>
import UserAvatar from '../common/UserAvatar.vue'
</script>

<template>
  <header class="app-header">
    <div class="header-left">
      <div class="logo-icon">
        <img src="/DSL图标_透明.ico" width="28" height="28" alt="DSL" class="logo-img" />
      </div>
      <span class="logo-text">Steel Multi-Agent System (SMAS)</span>
    </div>

    <nav class="header-nav">
      <a class="nav-item active" href="#">首页</a>
      <a class="nav-item" href="#">API开放平台</a>
    </nav>

    <div class="header-right">
      <div class="user-menu">
        <UserAvatar :size="32" role="user" />
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2.5"
          stroke-linecap="round"
          stroke-linejoin="round"
          class="chevron"
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </div>
    </div>
  </header>
</template>

<style scoped>
.app-header {
  height: var(--header-height);
  display: flex;
  align-items: center;
  padding: 0 var(--space-xl);
  background: var(--bg-white);
  border-bottom: 1px solid var(--border-color);
  position: sticky;
  top: 0;
  z-index: 100;
}

.header-left {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
}

.logo-icon {
  display: flex;
  align-items: center;
}
.logo-img {
  border-radius: 6px;
  object-fit: cover;
}

.logo-text {
  font-size: var(--font-size-xl);
  font-weight: 700;
  color: var(--text-primary);
}

.header-nav {
  display: flex;
  align-items: center;
  gap: var(--space-3xl);
  margin-left: 48px;
}

.nav-item {
  font-size: var(--font-size-base);
  color: var(--text-secondary);
  padding: 4px 0;
  position: relative;
  transition: color 0.2s ease;
}

.nav-item:hover {
  color: var(--text-primary);
}

.nav-item.active {
  color: var(--primary);
  font-weight: 500;
}

.nav-item.active::after {
  content: '';
  position: absolute;
  bottom: -2px;
  left: 50%;
  transform: translateX(-50%);
  width: 4px;
  height: 4px;
  border-radius: var(--radius-round);
  background: var(--primary);
}

.header-right {
  display: flex;
  align-items: center;
  gap: var(--space-lg);
  margin-left: auto;
}

.icon-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  transition: all 0.2s ease;
}

.icon-btn:hover {
  background: var(--hover-bg);
  color: var(--text-primary);
}

.user-menu {
  display: flex;
  align-items: center;
  gap: var(--space-xs);
  cursor: pointer;
  padding: 2px 4px;
  border-radius: var(--radius-md);
  transition: background 0.2s ease;
}

.user-menu:hover {
  background: var(--hover-bg);
}

.chevron {
  color: var(--text-secondary);
}
</style>

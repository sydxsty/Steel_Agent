<script setup>
import { ref } from 'vue'
import { modelOptions } from '../../data/mock.js'

const isOpen = ref(false)
const selected = ref(modelOptions[0])

function toggleDropdown() {
  isOpen.value = !isOpen.value
}

function selectModel(model) {
  selected.value = model
  isOpen.value = false
}
</script>

<template>
  <div class="model-selector">
    <!-- DeepSeek-style pill toggle -->
    <button
      class="think-toggle"
      :class="{ active: isOpen }"
      @click="toggleDropdown"
    >
      <!-- Brain/think icon -->
      <svg
        class="think-icon"
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.5"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        <path d="M9.663 17h4.674M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
      <span class="think-text">{{ selected.name }}</span>
      <svg
        class="think-chevron"
        :class="{ rotated: isOpen }"
        width="10"
        height="10"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2.5"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        <path d="M6 9l6 6 6-6" />
      </svg>
    </button>

    <!-- Dropdown -->
    <Transition name="dropdown">
      <div
        v-if="isOpen"
        class="model-dropdown"
      >
        <div
          v-for="model in modelOptions"
          :key="model.id"
          class="model-option"
          :class="{ selected: model.id === selected.id }"
          @click="selectModel(model)"
        >
          <div class="option-info">
            <span class="option-name">{{ model.name }}</span>
            <span class="option-desc">{{ model.description }}</span>
          </div>
          <svg
            v-if="model.id === selected.id"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2.5"
            stroke-linecap="round"
            stroke-linejoin="round"
            class="check-icon"
          >
            <path d="M20 6L9 17l-5-5" />
          </svg>
        </div>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.model-selector {
  position: relative;
  flex-shrink: 0;
}

/* DeepSeek-style pill toggle button */
.think-toggle {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 36px;
  min-width: 164px;
  padding: 0 12px;
  border: 1px solid var(--border-color);
  border-radius: 17px;
  background: var(--bg-white);
  color: var(--text-secondary);
  font-size: 13px;
  white-space: nowrap;
  cursor: pointer;
  transition: all 0.2s ease;
}

.think-toggle:hover {
  border-color: #d0d0da;
  background: #f9f9fb;
  color: var(--text-primary);
}

.think-toggle.active {
  border-color: var(--primary-light);
  background: var(--primary-bg);
  color: var(--primary);
}

.think-icon {
  flex-shrink: 0;
  opacity: 0.7;
}

.think-toggle.active .think-icon {
  opacity: 1;
}

.think-text {
  flex: 1;
  font-weight: 500;
  text-align: left;
}

.think-chevron {
  flex-shrink: 0;
  opacity: 0.5;
  transition: transform 0.2s ease;
}

.think-chevron.rotated {
  transform: rotate(180deg);
}

/* Dropdown */
.model-dropdown {
  position: absolute;
  bottom: calc(100% + 6px);
  left: 0;
  min-width: 230px;
  background: var(--bg-white);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-elevated);
  padding: var(--space-xs);
  z-index: 200;
}

.model-option {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-sm) var(--space-md);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: background 0.15s ease;
}

.model-option:hover {
  background: var(--hover-bg);
}

.model-option.selected {
  background: var(--active-bg);
}

.option-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.option-name {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary);
}

.option-desc {
  font-size: 11px;
  color: var(--text-secondary);
}

.check-icon {
  color: var(--primary);
  flex-shrink: 0;
}

/* Transition */
.dropdown-enter-active,
.dropdown-leave-active {
  transition: all 0.2s ease;
}

.dropdown-enter-from,
.dropdown-leave-to {
  opacity: 0;
  transform: translateY(4px);
}
</style>

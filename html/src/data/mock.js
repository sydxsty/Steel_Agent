// Bot info
export const botInfo = {
  name: 'Steel Multi-Agent System (SMAS)',
  description: '我可以帮助你回答管线钢的相关问题，针对成分和工艺进行设计优化',
}

// Chat history list
export const chatHistory = [
  { id: 1, title: '关于 Vue 3 的讨论', date: '05-30' },
  { id: 2, title: '帮我写一段 Python 代码', date: '05-29' },
  { id: 3, title: '翻译一篇英文文章', date: '05-28' },
  { id: 4, title: '数据分析方法咨询', date: '05-27' },
  { id: 5, title: '旅行规划建议', date: '05-26' },
]

// Suggested prompts for welcome state
export const suggestedPrompts = [
  {
    id: 1,
    icon: 'lightbulb',
    text: '设计一组管线钢的成分工艺',
  },
  {
    id: 2,
    icon: 'code',
    text: '设计一组厚度22mm的，屈服强度大于550MPa的管线钢成分工艺',
  },
  {
    id: 3,
    icon: 'book',
    text: '设计一组X80管线钢的成分工艺，目标屈服强度大于500MPa',
  },
  {
    id: 4,
    icon: 'chart',
    text: 'X65、X70和X80管线钢的成分设计差异有哪些？',
  },
  {
    id: 5,
    icon: 'globe',
    text: '管线钢TMCP控轧控冷中FET、FDT和返红温度如何协同控制？',
  },
  {
    id: 6,
    icon: 'calendar',
    text: '设计一组低碳高韧、适用于低温服役的管线钢成分与工艺',
  },
]

// Pre-rendered messages (AI content uses HTML for visual completeness)
export const messages = [
  {
    id: 1,
    role: 'assistant',
    content: '<p>你可以向我提问管线钢相关问题。</p>',
    time: '',
  },
]

// Model options
export const modelOptions = [
  { id: 'deepseek-v4-flash', name: 'DeepSeek V4 Flash', description: 'DeepSeek V4 Flash' },
  { id: 'qwen3-8-max', name: 'Qwen3.8 Max', description: 'Qwen3.8 Max' },
]

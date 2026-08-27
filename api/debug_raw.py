"""临时调试脚本：查看 LLM 原始输出"""
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from my_llm import deepseek_Llm
from intent_classifier import get_session_history, _parse_json_response

system_prompt = (
    "你是钢材意图识别助手。涉及钢材成分设计、工艺优化、性能分析输出DESIGN，否则输出CHAT。"
    '请严格只输出JSON格式，不要任何解释: {"intent": "DESIGN"} 或 {"intent": "CHAT"}'
)
user_message = "Q460钢的热处理工艺如何优化？"

prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
chain = prompt | deepseek_Llm | StrOutputParser()
chain_h = RunnableWithMessageHistory(
    chain, get_session_history,
    input_messages_key="input", history_messages_key="history"
)

raw = chain_h.invoke(
    {"system_prompt": system_prompt, "input": user_message},
    config={"configurable": {"session_id": "debug-raw-1"}}
)
print("=== LLM 原始输出 ===")
print(repr(raw))
print()
print("=== 解析后 ===")
print(_parse_json_response(raw, {"intent": "DESIGN"}))




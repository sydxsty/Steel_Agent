"""面向七类钢材用途的 LangChain RAG 检索工具。

每个工具固定连接一个用途知识库。LLM 只负责根据当前用户提示词选择工具
和组织检索词，不能在工具调用时覆盖数据库名称或集合名称。
"""

from langchain_core.tools import tool


KNOWLEDGE_BASE_COLLECTION = "documents"
KNOWLEDGE_BASE_TOP_K = 5


def _search_steel_knowledge_base(query: str, *, db_name: str, label: str) -> str:
    """执行固定用途知识库检索；知识库不可用时返回可供模型降级回答的文本。"""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return (
            f"【{label}知识库检索状态】未提供有效检索词。"
            "请不要改用其他用途知识库，直接依据专业知识保守回答。"
        )

    try:
        from hybrid_retriever import hybrid_search

        docs = hybrid_search(
            query=normalized_query,
            k=KNOWLEDGE_BASE_TOP_K,
            db_name=db_name,
            db_collection=KNOWLEDGE_BASE_COLLECTION,
        )
    except Exception as exc:
        print(
            f"[知识库工具] {label}检索不可用: db={db_name}, "
            f"collection={KNOWLEDGE_BASE_COLLECTION}, error={type(exc).__name__}: {exc}"
        )
        return (
            f"【{label}知识库检索状态】对应知识库暂不可用，未获得参考文献。"
            "请不要调用或引用其他用途知识库；可以依据专业知识保守回答，并明确说明缺少该用途知识库依据。"
        )

    if not docs:
        return (
            f"【{label}知识库检索状态】未检索到与当前问题相关的资料。"
            "请不要改用其他用途知识库，可以依据专业知识保守回答。"
        )

    sections = []
    for doc in docs:
        source = doc.get("source", "unknown") if isinstance(doc, dict) else "unknown"
        content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
        sections.append(f"[来源: {source}]\n{content}")
    return "\n\n---\n\n".join(sections)


@tool
def search_engineering_machinery_wear_steel_knowledge_base(query: str) -> str:
    """检索工程机械耐磨钢知识库。仅用于工程机械耐磨钢、耐磨板、NM系列及耐磨服役相关问题。"""
    return _search_steel_knowledge_base(
        query,
        db_name="gcjxyg_Know_db",
        label="工程机械耐磨钢",
    )


@tool
def search_pipeline_steel_knowledge_base(query: str) -> str:
    """检索管线钢知识库。仅用于管线钢、油气输送管、X系列管线钢及TMCP管线生产相关问题。"""
    return _search_steel_knowledge_base(
        query,
        db_name="gxg_Know_db",
        label="管线钢",
    )


@tool
def search_offshore_steel_knowledge_base(query: str) -> str:
    """检索海工钢知识库。仅用于海工钢、海洋工程平台和海洋工程结构用钢相关问题。"""
    return _search_steel_knowledge_base(
        query,
        db_name="hgg_Know_db",
        label="海工钢",
    )


@tool
def search_building_steel_knowledge_base(query: str) -> str:
    """检索建筑钢知识库。仅用于建筑工程、房建钢材、建筑型钢和钢筋相关问题。"""
    return _search_steel_knowledge_base(
        query,
        db_name="jzyg_Know_db",
        label="建筑钢",
    )


@tool
def search_structural_steel_knowledge_base(query: str) -> str:
    """检索结构钢知识库。仅用于通用高强度结构钢、低合金结构钢和结构件用钢相关问题。"""
    return _search_steel_knowledge_base(
        query,
        db_name="jgyg_Know_db",
        label="结构钢",
    )


@tool
def search_wind_power_steel_knowledge_base(query: str) -> str:
    """检索风电用钢知识库。仅用于陆上风电塔筒、风机塔架及其 TMCP 钢板的成分、工艺、组织、性能和焊接问题。"""
    return _search_steel_knowledge_base(
        query,
        db_name="jgyg_Know_db",
        label="风电用钢",
    )


@tool
def search_automotive_steel_knowledge_base(query: str) -> str:
    """检索汽车钢知识库。仅用于汽车钢、汽车板、大梁钢、DP、TRIP、QP和热成形钢相关问题。"""
    return _search_steel_knowledge_base(
        query,
        db_name="qcyg_Know_db",
        label="汽车钢",
    )


@tool
def search_bridge_steel_knowledge_base(query: str) -> str:
    """检索桥梁钢知识库。仅用于桥梁钢、桥梁结构和桥梁耐候钢相关问题。"""
    return _search_steel_knowledge_base(
        query,
        db_name="qlyg_Know_db",
        label="桥梁钢",
    )


KNOWLEDGE_BASE_TOOLS = [
    search_engineering_machinery_wear_steel_knowledge_base,
    search_pipeline_steel_knowledge_base,
    search_offshore_steel_knowledge_base,
    search_building_steel_knowledge_base,
    search_structural_steel_knowledge_base,
    search_wind_power_steel_knowledge_base,
    search_automotive_steel_knowledge_base,
    search_bridge_steel_knowledge_base,
]

KNOWLEDGE_BASE_TOOL_MAP = {tool_item.name: tool_item for tool_item in KNOWLEDGE_BASE_TOOLS}

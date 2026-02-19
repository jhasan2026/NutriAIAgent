from __future__ import annotations

from typing import TypedDict, Annotated, Dict, Any, List, Optional
import json

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import Tool
from langchain_core.runnables import RunnableConfig

from llm_manager import get_llm_instance
from tools import get_user_location, get_weather_and_season, get_local_food_prices
from rag import build_ehr_vectorstore, retrieve_ehr_context
from prompts import PROMPT
from safety import basic_safety_scan, medical_disclaimer, enforce_dietary_constraints


# -----------------------
# STATE
# -----------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    # user-provided profile and constraints
    user_profile: Dict[str, Any]
    hard_constraints: List[str]

    # EHR / RAG
    ehr_json: Dict[str, Any]
    ehr_ready: bool
    ehr_context: str

    # tool context
    tool_context: Dict[str, Any]

    # final text
    final_answer: str


# -----------------------
# LLM + Tools
# -----------------------
llm = get_llm_instance()

TOOLS = [
    get_user_location,
    get_weather_and_season,
    get_local_food_prices,
]

llm_with_tools = llm.bind_tools(TOOLS)


# -----------------------
# NODES
# -----------------------
def ensure_ehr_vectorstore(state: AgentState) -> AgentState:
    """
    Build vectorstore once when EHR exists.
    Store it in checkpointer memory by embedding into state as a serialized flag.
    (We rebuild in memory per thread; for production, persist to disk/db.)
    """
    # We'll store a minimal flag only. The actual store will be kept in a module-level cache by thread_id.
    return state


# Cache per thread_id (simple demo)
_VECTORSTORE_CACHE: Dict[str, Any] = {}


def build_or_get_vs(thread_id: str, ehr_json: Dict[str, Any]):
    if thread_id not in _VECTORSTORE_CACHE:
        _VECTORSTORE_CACHE[thread_id] = build_ehr_vectorstore(ehr_json)
    return _VECTORSTORE_CACHE[thread_id]


def tool_context_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Calls tools to get location -> weather/season -> local prices (CSV) context.
    """
    thread_id = str(config.get("configurable", {}).get("thread_id", "default"))
    user_profile = state.get("user_profile", {}) or {}

    # Optional overrides
    manual_location = user_profile.get("manual_location")  # e.g., "Dhaka"
    hemisphere = user_profile.get("hemisphere", "north")

    loc = {}
    if manual_location:
        # If user provided a manual location string, we won't geocode here; keep it simple.
        loc = {"status": "ok", "city": manual_location, "country": user_profile.get("country"), "latitude": None, "longitude": None}
    else:
        loc = get_user_location.invoke({"country_hint": user_profile.get("country")})

    weather = {}
    # Only call weather if we have lat/lon
    if loc.get("latitude") is not None and loc.get("longitude") is not None:
        weather = get_weather_and_season.invoke(
            {"latitude": float(loc["latitude"]), "longitude": float(loc["longitude"]), "hemisphere": hemisphere}
        )
    else:
        # still infer season without coords
        weather = {"status": "ok", "season": user_profile.get("season_hint", "unknown"), "note": "No lat/lon available."}

    # Prices: use city or manual location label
    price_loc = loc.get("city") or manual_location or "Dhaka"
    prices = get_local_food_prices.invoke({"location_name": price_loc})

    tool_context = {
        "location": loc,
        "weather": weather,
        "prices": prices,
    }

    return {"tool_context": tool_context}


def rag_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Retrieve relevant EHR snippets for the current user query.
    """
    thread_id = str(config.get("configurable", {}).get("thread_id", "default"))
    ehr_json = state.get("ehr_json", {}) or {}
    if not ehr_json:
        return {"ehr_context": "No EHR uploaded by the user."}

    # Use latest human message as query
    last_user = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user = m.content
            break
    query = last_user or "personalized nutrition recommendation"

    vs = build_or_get_vs(thread_id, ehr_json)
    docs = retrieve_ehr_context(vs, query=query, k=6)

    ehr_context = "\n".join([f"- ({d.metadata.get('type','ehr')}) {d.page_content}" for d in docs])
    return {"ehr_context": ehr_context}


def generate_answer_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Produce final nutrition plan using prompt + tool context + RAG.
    """
    user_profile = state.get("user_profile", {}) or {}
    tool_context = state.get("tool_context", {}) or {}
    ehr_context = state.get("ehr_context", "") or ""
    hard_constraints = state.get("hard_constraints", []) or []

    # Latest user message
    user_message = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_message = m.content
            break

    # Compose prompt
    prompt = PROMPT.format(
        user_message=user_message,
        user_profile=json.dumps(user_profile, ensure_ascii=False, indent=2),
        tool_context=json.dumps(tool_context, ensure_ascii=False, indent=2),
        ehr_context=ehr_context,
        hard_constraints=json.dumps(hard_constraints, ensure_ascii=False),
    )

    # Use tool-enabled LLM (it can still call tools if needed later)
    resp = llm.invoke(prompt)

    text = resp.content if hasattr(resp, "content") else str(resp)

    # Safety scan
    scan = basic_safety_scan(text)
    if not scan["ok"]:
        safe_text = (
            "I can’t help with that request because it may be unsafe medically.\n\n"
            "If you want, tell me your condition and goals, and I’ll provide a safe meal plan "
            "within clinical dietary guidelines.\n\n"
            f"Flagged content: {scan['hits']}\n\n"
            + medical_disclaimer()
        )
        return {"final_answer": safe_text, "messages": [AIMessage(content=safe_text)]}

    # Constraint check (basic)
    constraint_check = enforce_dietary_constraints(text, hard_constraints=hard_constraints)
    if not constraint_check["ok"]:
        text += (
            "\n\n⚠️ **Constraint warning:** I might have violated these constraints: "
            + ", ".join(constraint_check["violated"])
            + "\nPlease tell me which foods are strictly forbidden, and I will regenerate."
        )

    # Always add disclaimer
    if "Medical note" not in text and "Medical" not in text:
        text += "\n\n" + medical_disclaimer()

    return {"final_answer": text, "messages": [AIMessage(content=text)]}


# -----------------------
# GRAPH
# -----------------------
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("tool_context", tool_context_node)
    graph.add_node("rag", rag_node)
    graph.add_node("generate", generate_answer_node)

    graph.add_edge(START, "tool_context")
    graph.add_edge("tool_context", "rag")
    graph.add_edge("rag", "generate")
    graph.add_edge("generate", END)

    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)


chatbot = build_agent()

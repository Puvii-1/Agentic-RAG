"""
The agent: a loop between "call the LLM" and "run whatever tools it asked
for", until the LLM responds with a final answer (no more tool calls).

Uses LangGraph's MemorySaver checkpointer so conversation history persists
across turns within a session (keyed by thread_id) — this is what makes
follow-up questions like "what about its sequel?" work.
"""
import os
import logging
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful research assistant with access to tools.

- Use `search_documents` when the question is likely answered by the user's
  uploaded document.
- Use `web_search` for anything current, general, or outside the document's
  scope.
- Use `calculator` for any arithmetic — never compute math yourself.
- You can use more than one tool if the question needs it (e.g. look up a
  number in the document, then calculate with it).
- If tools return nothing useful, say so honestly rather than guessing.
- Cite whether your answer came from the document, the web, or both.
"""


@retry(
    retry=retry_if_exception_type(ChatGoogleGenerativeAIError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    reraise=True,
)
def _invoke_with_retry(llm, messages):
    """
    Retries transient failures (e.g. 503 UNAVAILABLE 'high demand') with
    exponential backoff: waits ~2s, then ~4s, then ~8s between the 3
    attempts before giving up and re-raising.

    Note: a 429 RESOURCE_EXHAUSTED for a *daily* quota won't be fixed by
    a short retry — but retrying briefly is harmless, and we still want
    to convert the eventual failure into a friendly message rather than
    crashing the app (see agent_node below).
    """
    return llm.invoke(messages)


def _friendly_error_message(error: Exception) -> str:
    text = str(error)
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return (
            "⚠️ I've hit the Gemini API's free-tier request limit for this model. "
            "This isn't a bug in the app — daily quotas reset at midnight Pacific Time. "
            "In the meantime, you can try setting `GEMINI_MODEL=gemini-2.5-flash-lite` "
            "in your `.env` file, which typically has a higher free daily quota."
        )
    if "UNAVAILABLE" in text or "503" in text:
        return (
            "⚠️ Gemini's servers are temporarily overloaded. I retried a few times "
            "automatically, but it's still unavailable — please try asking again in a minute."
        )
    return f"⚠️ The model call failed after retrying: {text[:300]}"


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def extract_text(content) -> str:
    """
    Newer versions of langchain-google-genai return message content as a
    list of content blocks (e.g. [{"type": "text", "text": "...", "extras": {...}}])
    instead of a plain string, especially when the response includes extra
    metadata like grounding signatures. This normalizes either shape into
    plain text for display.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def build_agent(tools: list):
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    ).bind_tools(tools)

    tools_by_name = {t.name: t for t in tools}

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        # Ensure the system prompt is present exactly once, at the start
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        try:
            response = _invoke_with_retry(llm, messages)
        except ChatGoogleGenerativeAIError as e:
            logger.warning(f"Gemini call failed after retries: {e}")
            return {"messages": [AIMessage(content=_friendly_error_message(e))]}
        return {"messages": [response]}

    def tool_node(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        results = []
        for call in last_message.tool_calls:
            tool_fn = tools_by_name[call["name"]]
            output = tool_fn.invoke(call["args"])
            results.append(ToolMessage(content=str(output), tool_call_id=call["id"], name=call["name"]))
        return {"messages": results}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=MemorySaver())


def run_agent(app, user_input: str, thread_id: str = "default"):
    """
    thread_id: identifies the conversation. Use the same thread_id across
    turns in one chat session so memory persists; use a new one to start fresh.
    Returns the full list of new messages produced this turn (including any
    tool calls/results) so the UI can show the reasoning trace.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [("user", user_input)]}, config=config)
    return result["messages"]
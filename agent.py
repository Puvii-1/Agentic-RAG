"""
The agent: a loop between "call the LLM" and "run whatever tools it asked
for", until the LLM responds with a final answer (no more tool calls).

Uses LangGraph's MemorySaver checkpointer so conversation history persists
across turns within a session (keyed by thread_id) — this is what makes
follow-up questions like "what about its sequel?" work.
"""
import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

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
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    ).bind_tools(tools)

    tools_by_name = {t.name: t for t in tools}

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        # Ensure the system prompt is present exactly once, at the start
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm.invoke(messages)
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
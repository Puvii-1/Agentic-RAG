"""
Tools available to the agent. Each is a plain function decorated with
@tool so the LLM can call it via native tool-calling. The agent decides
which of these to use (or none) based on the user's question.
"""
import ast
import operator as op
from langchain_core.tools import tool
from ddgs import DDGS
from retrieval import hybrid_search


def make_document_search_tool(retriever):
    """
    Factory because this tool needs access to your indexed document's
    retriever, which only exists after a PDF is uploaded.
    """

    @tool
    def search_documents(query: str) -> str:
        """Search the uploaded PDF document for information relevant to the query.
        Use this when the question is likely answered by the document's content
        (e.g. specific facts, figures, or statements the document would contain)."""
        docs = hybrid_search(retriever, query, top_n=4)
        if not docs:
            return "No relevant content found in the document."
        formatted = []
        for d in docs:
            page = d.metadata.get("page", 0) + 1
            formatted.append(f"[Page {page}] {d.page_content[:500]}")
        return "\n\n".join(formatted)

    return search_documents


@tool
def web_search(query: str) -> str:
    """Search the web for current or general information NOT found in the
    uploaded document — e.g. recent events, facts outside the document's
    scope, or anything the document doesn't cover."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No web results found."
        formatted = [f"{r['title']}: {r['body']}" for r in results]
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Web search failed: {e}"


# --- Safe calculator: only arithmetic, no eval() of arbitrary code ---
_ALLOWED_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg,
    ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. '12 * (4 + 3)' or '150 / 3 - 2'.
    Use this for any arithmetic instead of computing it yourself."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Could not evaluate expression: {e}"
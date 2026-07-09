"""
Small evaluation harness: run a fixed set of test queries through the
agent and score each answer with an LLM-as-judge. This gives you a
measurable quality signal instead of eyeballing a few manual tries —
worth having for a portfolio project and for catching regressions
after you change prompts or retrieval settings.

Run with: python eval.py
(requires a document already indexed at ./chroma_db, and GOOGLE_API_KEY set)
"""
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from rag_pipeline import load_vectorstore
from retrieval import build_hybrid_retriever
from tools import make_document_search_tool, web_search, calculator
from agent import build_agent, run_agent

load_dotenv()

# Edit this list to match questions relevant to whatever document you've indexed.
# expected_tool is just a hint for you to sanity-check routing; the judge
# doesn't use it, it's informational when you read the results.
TEST_QUERIES = [
    {"query": "What is the main topic of this document?", "expected_tool": "search_documents"},
    {"query": "What is 45 * 12 + 7?", "expected_tool": "calculator"},
    {"query": "What is today's top news headline?", "expected_tool": "web_search"},
]

JUDGE_PROMPT = """You are grading an AI assistant's answer for quality.
Score from 1 (bad) to 5 (excellent) on:
- Relevance: does it actually answer the question?
- Groundedness: does it seem to use retrieved information rather than guessing?
- Honesty: if it couldn't find something, does it say so rather than fabricate?

Question: {question}
Answer: {answer}

Respond in EXACTLY this JSON format, nothing else:
{{"score": <1-5>, "reasoning": "<one sentence>"}}
"""


def judge(question: str, answer: str) -> dict:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))
    response = llm.invoke([HumanMessage(content=JUDGE_PROMPT.format(question=question, answer=answer))])
    text = response.content.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"score": None, "reasoning": f"Could not parse judge output: {text}"}


def run_evaluation():
    vectordb = load_vectorstore()
    # NOTE: for hybrid retrieval BM25 needs the raw chunks, not just vectordb.
    # If you're running eval right after indexing in the same session, pass
    # the chunks you already have. This standalone script re-fetches them
    # from Chroma's stored documents instead.
    all_docs = vectordb.get()
    from langchain_core.documents import Document
    chunks = [Document(page_content=doc, metadata=meta) for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])]

    retriever = build_hybrid_retriever(vectordb, chunks)
    doc_tool = make_document_search_tool(retriever)
    agent_app = build_agent([doc_tool, web_search, calculator])

    results = []
    for i, item in enumerate(TEST_QUERIES):
        messages = run_agent(agent_app, item["query"], thread_id=f"eval-{i}")
        final_answer = messages[-1].content
        tools_used = [m.name for m in messages if hasattr(m, "tool_call_id")]
        score = judge(item["query"], final_answer)

        results.append({
            "query": item["query"],
            "expected_tool": item["expected_tool"],
            "tools_actually_used": tools_used,
            "answer": final_answer,
            "score": score.get("score"),
            "judge_reasoning": score.get("reasoning"),
        })

    print(json.dumps(results, indent=2))
    valid_scores = [r["score"] for r in results if r["score"] is not None]
    if valid_scores:
        print(f"\nAverage score: {sum(valid_scores) / len(valid_scores):.2f} / 5")
    return results


if __name__ == "__main__":
    run_evaluation()
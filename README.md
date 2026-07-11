# Agentic RAG

Evolution of the Document Q&A RAG app into an **agent** that decides what
to do with each question — search the document, search the web, do math,
or some combination — instead of always just retrieving-then-answering.

## What's new vs. the original RAG app

|                 | Original RAG                                  | Agentic RAG                                                   |
| --------------- | --------------------------------------------- | ------------------------------------------------------------- |
| Retrieval       | Vector search only                            | **Hybrid**: BM25 + vector, then **cross-encoder re-ranked**   |
| Decision making | Always retrieves, always answers from context | **Agent decides** which tool(s) to use per question           |
| Tools           | None                                          | Document search, web search, calculator                       |
| Memory          | None (each question independent)              | **Persists across turns** in a conversation                   |
| Evaluation      | Manual testing                                | **LLM-as-judge harness** (`eval.py`) with scored test queries |
| UI              | Answer + sources                              | Answer + **reasoning trace** (which tools fired, why)         |

`rag_pipeline.py` (loading/splitting/embedding/vectorstore) is unchanged
from the original — the agent is built as a new layer around it.

## Architecture

```
User question
     │
     ▼
┌─────────┐   tool call    ┌───────────────┐
│  Agent   │ ─────────────▶│     Tools      │
│ (Gemini) │                │ • search_documents (hybrid + rerank)
│          │◀───────────────│ • web_search (DuckDuckGo)
└─────────┘   tool result   │ • calculator (safe eval)
     │                       └───────────────┘
     │ no more tool calls
     ▼
 Final answer (+ reasoning trace shown in UI)
```

The agent loop (`agent.py`) is a small LangGraph: an `agent` node calls
the LLM with tools bound; if the response includes tool calls, a `tools`
node executes them and loops back to the agent; otherwise it ends.
Conversation memory is handled by LangGraph's `MemorySaver` checkpointer,
keyed by a `thread_id` per chat session.

## Setup

1. **Same Gemini key as before** — reuse your existing `.env` file, or:

   ```
   copy .env.example .env
   ```

   and paste in `GOOGLE_API_KEY=your_key`

2. **Install dependencies** (fresh venv recommended):

   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the app**:

   ```
   streamlit run app.py
   ```

4. Upload a PDF, click **Index document**, then try questions like:
   - _"What does this document say about X?"_ → routes to document search
   - _"What's 340 _ 12?"\* → routes to calculator
   - _"What's the latest news on Y?"_ → routes to web search
   - A follow-up like _"and what about its cause?"_ → uses memory to know what "its" refers to

   Expand **"Reasoning trace"** under any answer to see exactly which
   tool(s) fired and what they returned.

## Running the evaluation harness

```
python eval.py
```

This runs a small fixed set of test questions through the agent and
scores each answer 1-5 using Gemini as a judge (relevance, groundedness,
honesty). Edit `TEST_QUERIES` in `eval.py` to match whatever document
you've indexed. Useful for catching regressions if you tweak prompts or
retrieval settings later.

## Project structure

```
agentic-rag/
├── rag_pipeline.py   # unchanged: load/split/embed/vectorstore (original RAG app)
├── retrieval.py        # NEW: hybrid retrieval (BM25 + vector) + cross-encoder rerank
├── tools.py              # NEW: document search, web search, calculator tools
├── agent.py                # NEW: LangGraph agent loop + memory
├── eval.py                    # NEW: evaluation harness (LLM-as-judge)
├── app.py                        # Streamlit UI (updated with reasoning trace)
├── requirements.txt
└── .env.example
```

## Notes on things worth knowing before you demo this

- **Hybrid retrieval weighting**: `retrieval.py` weights vector search
  0.6 and BM25 0.4 — tune this in `build_hybrid_retriever` if you find
  one method dominating unhelpfully for your documents.
- **Web search** uses DuckDuckGo via the `ddgs` package — no API key
  needed, but results can be rate-limited if you hit it very rapidly.
- **Calculator is a safe AST-based evaluator**, not raw `eval()` — it
  only allows arithmetic operations, so it can't be used to run arbitrary
  code. Worth mentioning if asked about security in an interview.
- **Memory resets** when you index a new document (fresh `thread_id`),
  so old conversation context doesn't leak into a new document's context.

## Ideas to extend further

- Add a "fact-check" step: after the agent answers, have it critique its
  own answer against the retrieved sources before showing it to the user
- Stream the agent's tokens live instead of waiting for the full response
- Add a proper retrieval metric (e.g. recall@k on a labeled Q&A set) to
  `eval.py` alongside the LLM-judge score

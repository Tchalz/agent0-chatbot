"""
Core ReAct-style agent loop, ported from agentWorkflow.ipynb.
Plan -> Execute -> Observe -> Reflect, looping until the model finishes.
"""
import os
import json
import re
import textwrap
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# NOTE: 'openrouter/owl-alpha' was a placeholder in the original notebook.
# Set a real OpenRouter model id, e.g. "anthropic/claude-3.5-sonnet" or "openai/gpt-4o".
MODEL = os.getenv("AGENT_MODEL", "openrouter/owl-alpha")
MAX_ITERATIONS = 6  # safety ceiling on the loop


# ── Tools the agent can call ────────────────────────────────────────────────

async def tool_search(query: str) -> str:
    """
    Real web search via Tavily (https://tavily.com), which returns an
    LLM-ready answer plus a few supporting sources rather than raw HTML.

    Falls back to a clearly-labeled placeholder if TAVILY_API_KEY isn't set,
    so the app still runs (with degraded, non-factual answers) for anyone
    who hasn't configured a search key yet.
    """
    if not TAVILY_API_KEY:
        return (
            f"[No search provider configured — set TAVILY_API_KEY to enable "
            f"real search] Unable to search for: {query}"
        )

    try:
        async with httpx.AsyncClient(timeout=15) as http_client:
            resp = await http_client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "include_answer": True,
                    "max_results": 3,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        return f"Search failed: {e}"

    parts = []
    if data.get("answer"):
        parts.append(f"Summary: {data['answer']}")

    for r in data.get("results", [])[:3]:
        title = r.get("title", "")
        url = r.get("url", "")
        content = (r.get("content") or "")[:300]
        parts.append(f"- {title} ({url}): {content}")

    if not parts:
        return f"No search results found for: {query}"

    return "\n".join(parts)


def tool_calculate(expression: str) -> str:
    """Safe arithmetic evaluator."""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: unsafe expression"
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


async def tool_summarise(text: str) -> str:
    """Summarise text using the LLM."""
    resp = await client.chat.completions.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": f"Summarise this in 2 sentences:\n{text}"}],
    )
    return resp.choices[0].message.content


TOOLS = {
    "search":    (tool_search,    "Search the web for information. Input: query string."),
    "calculate": (tool_calculate, "Evaluate a math expression. Input: expression string."),
    "summarise": (tool_summarise, "Summarise a block of text. Input: text string."),
}


async def run_tool(name: str, input_str: str) -> str:
    if name not in TOOLS:
        return f"Unknown tool '{name}'"
    fn, _ = TOOLS[name]
    result = fn(input_str)
    # tool_search and tool_calculate are sync, tool_summarise is async
    if hasattr(result, "__await__"):
        result = await result
    return result


# ── System prompt ────────────────────────────────────────────────────────────

def parse_step(raw: str) -> dict | None:
    """
    Try to pull a single JSON object out of a model response.

    Handles three real-world failure modes seen in production:
      - clean JSON (the happy path)
      - JSON wrapped in markdown ```json fences
      - JSON with stray prose before/after it
      - an empty / whitespace-only response (returns None instead of raising)

    Returns the parsed dict, or None if no valid JSON object could be found.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # 1. try as-is
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. strip markdown code fences if present
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    if fenced != text:
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass

    # 3. last resort: grab the first {...} block anywhere in the text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def build_system_prompt() -> str:
    tool_descriptions = "\n".join(
        f"  - {name}: {desc}" for name, (_, desc) in TOOLS.items()
    )
    return textwrap.dedent(f"""
        You are a goal-oriented AI agent. You work in a loop:
        PLAN -> EXECUTE -> OBSERVE -> REFLECT -> repeat until done.

        Available tools:
        {tool_descriptions}

        At each step respond with EXACTLY one JSON object (no markdown, no extra text):

        If you need to call a tool:
        {{"action": "tool", "tool": "<name>", "input": "<input>", "thought": "<why>"}}

        If you have the final answer:
        {{"action": "finish", "answer": "<final answer>", "thought": "<summary>"}}

        Rules:
        - Use tools when you need external information or computation.
        - Finish as soon as you have enough information to answer fully.
        - Never guess when a tool can give you the real answer.
    """).strip()


# ── Core agent loop ──────────────────────────────────────────────────────────

async def run_agent(goal: str, on_step=None, is_cancelled=None) -> dict:
    """
    Run the agent loop for a single goal.

    on_step: optional callback(dict) invoked after every iteration with a
             step record, useful for streaming progress to a UI.
    is_cancelled: optional async callable returning True if the caller wants
             the loop to stop early (e.g. the client disconnected after
             clicking Stop). Checked before each LLM call and after each
             tool call, so a cancellation lands within one step instead of
             waiting for the full 6-iteration loop to finish.

    Returns a dict: {"answer": str, "steps": [...]} or, if cancelled,
    {"answer": None, "steps": [...], "cancelled": True}.
    """
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": f"Goal: {goal}"},
    ]

    steps = []

    async def cancelled() -> bool:
        return bool(is_cancelled) and await is_cancelled()

    for iteration in range(1, MAX_ITERATIONS + 1):
        if await cancelled():
            return {"answer": None, "steps": steps, "cancelled": True}

        response = await client.chat.completions.create(
            model=MODEL,
            max_tokens=500,
            messages=messages,
        )
        raw = (response.choices[0].message.content or "").strip()

        step = parse_step(raw)

        if step is None:
            # The model returned something we couldn't parse as JSON (empty
            # response, prose, a refusal, etc). Give it one corrective nudge
            # rather than crashing the whole request with a 500.
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    "Your last response was not valid JSON, or was empty. "
                    "Respond with EXACTLY one JSON object as instructed, "
                    "and nothing else."
                ),
            })

            retry_response = await client.chat.completions.create(
                model=MODEL,
                max_tokens=500,
                messages=messages,
            )
            raw = (retry_response.choices[0].message.content or "").strip()
            step = parse_step(raw)

            if step is None:
                # Still couldn't get usable JSON after a retry. Stop cleanly
                # instead of crashing, and surface this to the UI as the
                # final answer so the user sees *something* instead of a 500.
                fallback = (
                    "The agent's underlying model returned an unreadable "
                    "response and could not continue. Try again, or try a "
                    "different model in AGENT_MODEL."
                )
                steps.append({
                    "iteration": iteration,
                    "type": "finish",
                    "thought": "",
                    "answer": fallback,
                })
                return {"answer": fallback, "steps": steps}

        if "action" not in step:
            # Malformed but parseable JSON (missing required field) — treat
            # the same as an unparseable response rather than KeyError-ing.
            fallback = (
                "The agent's underlying model returned an incomplete step "
                "and could not continue. Try again, or try a different "
                "model in AGENT_MODEL."
            )
            steps.append({
                "iteration": iteration,
                "type": "finish",
                "thought": step.get("thought", ""),
                "answer": fallback,
            })
            return {"answer": fallback, "steps": steps}

        thought = step.get("thought", "")

        if step["action"] == "finish":
            record = {
                "iteration": iteration,
                "type": "finish",
                "thought": thought,
                "answer": step.get("answer", "(no answer provided)"),
            }
            steps.append(record)
            if on_step:
                on_step(record)
            return {"answer": record["answer"], "steps": steps}

        if await cancelled():
            return {"answer": None, "steps": steps, "cancelled": True}

        tool_name = step.get("tool", "")
        tool_input = step.get("input", "")
        observation = await run_tool(tool_name, tool_input)

        record = {
            "iteration": iteration,
            "type": "tool",
            "thought": thought,
            "tool": tool_name,
            "input": tool_input,
            "observation": observation,
        }
        steps.append(record)
        if on_step:
            on_step(record)

        if await cancelled():
            return {"answer": None, "steps": steps, "cancelled": True}

        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    fallback = "Max iterations reached without a final answer."
    steps.append({"iteration": MAX_ITERATIONS, "type": "finish", "thought": "", "answer": fallback})
    return {"answer": fallback, "steps": steps}

"""
Core ReAct-style agent loop, ported from agentWorkflow.ipynb.
Plan -> Execute -> Observe -> Reflect, looping until the model finishes.
"""
import os
import json
import textwrap
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# NOTE: 'openrouter/owl-alpha' was a placeholder in the original notebook.
# Set a real OpenRouter model id, e.g. "anthropic/claude-3.5-sonnet" or "openai/gpt-4o".
MODEL = os.getenv("AGENT_MODEL", "openrouter/owl-alpha")
MAX_ITERATIONS = 6  # safety ceiling on the loop


# ── Tools the agent can call ────────────────────────────────────────────────
# tool_search is a SIMULATED tool carried over from the notebook — it does not
# actually search the web. Replace with a real search API call if you need
# real answers.

def tool_search(query: str) -> str:
    """Simulated search tool. Replace with a real API call."""
    return f"[Search result for '{query}']: Found relevant information about {query}."


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


def tool_summarise(text: str) -> str:
    """Summarise text using the LLM."""
    resp = client.chat.completions.create(
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


def run_tool(name: str, input_str: str) -> str:
    if name not in TOOLS:
        return f"Unknown tool '{name}'"
    fn, _ = TOOLS[name]
    return fn(input_str)


# ── System prompt ────────────────────────────────────────────────────────────

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

def run_agent(goal: str, on_step=None) -> dict:
    """
    Run the agent loop for a single goal.

    on_step: optional callback(dict) invoked after every iteration with a
             step record, useful for streaming progress to a UI.

    Returns a dict: {"answer": str, "steps": [list of step records]}
    """
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": f"Goal: {goal}"},
    ]

    steps = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=500,
            messages=messages,
        )
        raw = response.choices[0].message.content.strip()

        try:
            step = json.loads(raw)
        except json.JSONDecodeError:
            cleaned = raw.strip("`").replace("json\n", "", 1)
            step = json.loads(cleaned)

        thought = step.get("thought", "")

        if step["action"] == "finish":
            record = {
                "iteration": iteration,
                "type": "finish",
                "thought": thought,
                "answer": step["answer"],
            }
            steps.append(record)
            if on_step:
                on_step(record)
            return {"answer": step["answer"], "steps": steps}

        tool_name = step["tool"]
        tool_input = step["input"]
        observation = run_tool(tool_name, tool_input)

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

        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    fallback = "Max iterations reached without a final answer."
    steps.append({"iteration": MAX_ITERATIONS, "type": "finish", "thought": "", "answer": fallback})
    return {"answer": fallback, "steps": steps}

"""
FastAPI app that wraps the ReAct agent loop (agent.py) as a chatbot.
Serves a static chat UI and a /chat endpoint that runs the agent loop.
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from agent import run_agent

app = FastAPI(title="Agent Chatbot")

app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    """
    Runs the full plan/execute/observe/reflect loop for the given message
    and returns the final answer plus the step-by-step trace.

    If the client disconnects (e.g. the Stop button was clicked, which
    aborts the fetch), the agent loop stops making further LLM/tool calls
    instead of running to completion in the background.
    """

    async def is_cancelled() -> bool:
        return await request.is_disconnected()

    result = await run_agent(req.message, is_cancelled=is_cancelled)

    if result.get("cancelled"):
        # Client already disconnected; nothing to send, but return cleanly
        # in case this code path is ever hit with a still-open connection.
        return JSONResponse(result, status_code=499)

    return result

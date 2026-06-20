"""
FastAPI app that wraps the ReAct agent loop (agent.py) as a chatbot.
Serves a static chat UI and a /chat endpoint that runs the agent loop.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
def chat(req: ChatRequest):
    """
    Runs the full plan/execute/observe/reflect loop for the given message
    and returns the final answer plus the step-by-step trace.
    """
    result = run_agent(req.message)
    return result

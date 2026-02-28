"""
AI Agents Module -- Autonomous Reasoning Systems
==================================================

WHAT ARE AI AGENTS?
  An AI agent is a system that uses a Large Language Model (LLM) as its
  "brain" to autonomously decide what actions to take, execute those
  actions, observe the results, and iterate until a task is complete.

  Unlike simple chat bots that just generate text, agents can:
    - Call tools (search, calculate, browse the web)
    - Reason about multi-step problems
    - Adapt their approach based on intermediate results
    - Collaborate with other agents

  Think of it this way:
    - **LLM** = a brain that can think and speak
    - **Tools** = hands that can interact with the world
    - **Agent** = a person with a brain AND hands, who can plan and act

THE AGENT SPECTRUM (from simple to complex):
  ┌─────────────────────────────────────────────────────────────────┐
  │  Simple Chat    Tool-Calling    ReAct Agent    Multi-Agent     │
  │  (no tools)     (1-2 calls)     (loop)         (team)          │
  │                                                                 │
  │  "What is       "Calculate      "Research      "Research X,    │
  │   Python?"       sqrt(144)"      topic X,       analyze data,  │
  │                                  verify it,     write report"  │
  │                                  summarize"                     │
  │                                                                 │
  │  ◄──────── Increasing Complexity + Capability ──────────►      │
  └─────────────────────────────────────────────────────────────────┘

AGENTS IN THIS MODULE:

  1. **ReActAgent** -- Reasoning + Acting loop (Yao et al., 2022)
     The agent explicitly THINKS, then ACTS (calls tools), then OBSERVES
     results, repeating until it has a complete answer. Best for complex
     research and multi-step reasoning tasks.

  2. **ToolAgent** -- Simple tool-calling agent
     The LLM directly decides which tools to call based on the query.
     No explicit reasoning loop -- the LLM handles it internally.
     Best for straightforward tool-dispatch tasks.

  3. **MultiAgentSupervisor** -- Multi-agent collaboration
     A supervisor agent routes tasks to specialist agents (researcher,
     analyst, writer). Each specialist is optimized for its role.
     Best for complex tasks requiring diverse skills.

TOOLS AVAILABLE:
  All agents can use these tools (from ``agentexplorr.agents.tools``):
    - ``web_search`` -- DuckDuckGo search (no API key needed)
    - ``calculator`` -- Safe math evaluation (no eval()!)
    - ``web_scrape`` -- Extract text from URLs (httpx + BeautifulSoup)

TECHNOLOGY STACK (100% open source, no API keys):
  - **LangGraph** -- Agent orchestration as state machines (graphs)
  - **langchain-ollama** -- LLM inference via local Ollama server
  - **ChatOllama** -- LangChain chat model wrapper for Ollama
  - **DuckDuckGo** -- Free web search, no API key required
  - **Ollama** -- Run LLMs locally (llama3.2, mistral, etc.)

GETTING STARTED:
  1. Install Ollama: https://ollama.com/download
  2. Pull a model: ``ollama pull llama3.2``
  3. Start Ollama: ``ollama serve``
  4. Run an agent:

     >>> from agentexplorr.agents import ReActAgent
     >>> agent = ReActAgent()
     >>> result = agent.run("What is the population of Japan?")
     >>> print(result.final_answer)

LEARNING RESOURCES:
  - LangGraph docs: https://langchain-ai.github.io/langgraph/
  - Ollama docs: https://github.com/ollama/ollama
  - LangChain Agents: https://python.langchain.com/docs/concepts/agents/
  - PAPER: "ReAct" (Yao et al., 2022) -- https://arxiv.org/abs/2210.03629
  - PAPER: "Toolformer" (Schick et al., 2023) -- https://arxiv.org/abs/2302.04761
  - PAPER: "A Survey on LLM-based Agents" -- https://arxiv.org/abs/2308.11432
  - VIDEO: "What are AI Agents?" -- https://www.youtube.com/watch?v=F8NKVhkZZWI
  - VIDEO: "LangGraph Crash Course" -- https://www.youtube.com/watch?v=R-o_a6F-SOQ
  - VIDEO: "Build AI Agents from Scratch" -- https://www.youtube.com/watch?v=AxnL5GtWnGE
"""

from __future__ import annotations

from agentexplorr.agents.react_agent import ReActAgent, ReActResult
from agentexplorr.agents.tool_agent import ToolAgent, ToolAgentResult
from agentexplorr.agents.multi_agent import MultiAgentSupervisor, MultiAgentResult

__all__ = [
    # Agent classes
    "ReActAgent",
    "ToolAgent",
    "MultiAgentSupervisor",
    # Result classes
    "ReActResult",
    "ToolAgentResult",
    "MultiAgentResult",
]

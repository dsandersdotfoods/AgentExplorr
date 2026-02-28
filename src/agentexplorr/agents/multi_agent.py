"""
Multi-Agent Supervisor System -- Orchestrating Specialist Agents
=================================================================

WHAT IS A MULTI-AGENT SYSTEM?
  A multi-agent system uses MULTIPLE specialized agents that collaborate
  to solve complex tasks. Instead of one monolithic agent trying to do
  everything, we break the work into roles:

    - **Supervisor** -- The "manager" who reads the user's request and
      decides which specialist to delegate to. It also synthesizes the
      final response from the specialists' outputs.

    - **Researcher** -- Specializes in finding information via web search.
      Good at formulating search queries and extracting relevant facts.

    - **Analyst** -- Specializes in quantitative analysis: calculations,
      data processing, comparisons, and logical reasoning.

    - **Writer** -- Specializes in synthesizing information into clear,
      well-structured responses. Takes raw facts and analysis and turns
      them into polished prose.

WHY MULTI-AGENT?
  Single agents struggle with complex, multi-faceted tasks because:
    1. One prompt can't effectively cover all skills (search, analysis, writing)
    2. Long reasoning chains lose focus and coherence
    3. Debugging is harder when one agent does everything

  Multi-agent systems address this through DIVISION OF LABOR:
    - Each agent has a focused prompt optimized for its specialty
    - The supervisor maintains high-level coordination
    - Specialists can be improved independently
    - The system is more modular and testable

ARCHITECTURE:
  ┌──────────────────────────────────────────────────────┐
  │                    User Query                         │
  └──────────────────────┬───────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────┐
  │                   SUPERVISOR                          │
  │  "Who should handle this? Let me route it."          │
  └───┬──────────────┬──────────────┬────────────────────┘
      │              │              │
      ▼              ▼              ▼
  ┌────────┐   ┌──────────┐   ┌────────┐
  │RESEARCHER│  │ ANALYST  │   │ WRITER │
  │(search) │  │(compute) │   │(draft) │
  └────┬─────┘ └─────┬────┘   └───┬────┘
      │              │              │
      └──────────────┼──────────────┘
                     │
                     ▼
  ┌──────────────────────────────────────────────────────┐
  │                   SUPERVISOR                          │
  │  "Let me combine these results into a final answer." │
  └──────────────────────────────────────────────────────┘

LANGGRAPH IMPLEMENTATION:
  We model this as a state graph where:
    - The supervisor is a node that returns routing decisions
    - Each specialist is a node that processes the query with its tools
    - Conditional edges route from supervisor to the chosen specialist
    - After each specialist finishes, control returns to the supervisor
    - The supervisor decides whether to call another specialist or finish

RELATED PATTERNS:
  - **Crew AI** -- Another multi-agent framework (role-playing agents)
  - **AutoGen** -- Microsoft's multi-agent conversation framework
  - **Mixture of Experts (MoE)** -- Similar idea at the model architecture level
  - **Hierarchical agents** -- Multi-level supervision (manager -> team leads -> workers)

LEARNING RESOURCES:
  - LangGraph Multi-Agent Tutorial: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/
  - LangGraph Supervisor Pattern: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/
  - PAPER: "AutoGen: Enabling Next-Gen LLM Applications" -- https://arxiv.org/abs/2308.08155
  - PAPER: "Communicative Agents for Software Development" (ChatDev) -- https://arxiv.org/abs/2307.07924
  - VIDEO: "Multi-Agent Systems with LangGraph" -- https://www.youtube.com/watch?v=hvAPnpSfSGo
  - VIDEO: "Build a Multi-Agent AI System" -- https://www.youtube.com/watch?v=hKPBBMPb7nk
  - VIDEO: "AI Agent Teams (Supervisor Pattern)" -- https://www.youtube.com/watch?v=DjC6F-ByRHk
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agentexplorr.agents.tools.calculator import calculator
from agentexplorr.agents.tools.search import web_search
from agentexplorr.agents.tools.web_scraper import web_scrape
from agentexplorr.core import Settings, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Specialist agent names (used as node names and routing targets)
# ---------------------------------------------------------------------------
# These constants prevent typos in string-based routing.
# If you add a new specialist, add it here first.

RESEARCHER: str = "researcher"
ANALYST: str = "analyst"
WRITER: str = "writer"
SUPERVISOR: str = "supervisor"
FINISH: str = "FINISH"

SPECIALISTS: list[str] = [RESEARCHER, ANALYST, WRITER]


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class MultiAgentState(TypedDict):
    """State shared across all agents in the multi-agent system.

    WHY A SHARED STATE?
      All agents need to see the same conversation history. When the
      researcher finds information, the analyst needs to see it. When
      the analyst produces numbers, the writer needs them. Shared state
      makes this natural -- each agent reads from and writes to the
      same message list.

    Attributes:
        messages: Full conversation history across all agents.
        next_agent: Which agent the supervisor wants to call next.
        specialist_outputs: Results from each specialist (for aggregation).
        iteration: Loop counter (safety limit).
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_agent: str
    specialist_outputs: dict[str, str]
    iteration: int


# ---------------------------------------------------------------------------
# System prompts for each agent
# ---------------------------------------------------------------------------
# Each specialist has a focused prompt that defines its role, capabilities,
# and output format. The supervisor has a meta-prompt about routing.

SUPERVISOR_PROMPT: str = """You are a supervisor managing a team of specialist AI agents.

Your team members are:
- **researcher**: Expert at finding information using web search. Delegate to researcher when the task requires looking up facts, current events, or external knowledge.
- **analyst**: Expert at quantitative analysis, calculations, comparisons, and logical reasoning. Delegate to analyst when the task requires math, data analysis, or structured reasoning.
- **writer**: Expert at synthesizing information into clear, well-structured responses. Delegate to writer when you have enough raw information and need it crafted into a polished answer.

Your job is to:
1. Read the user's request and any specialist outputs so far
2. Decide which specialist should work next (or if we're done)
3. Provide brief instructions for the chosen specialist

RESPOND WITH EXACTLY ONE OF THESE (and nothing else before it):
- "ROUTE: researcher" -- to send to the researcher
- "ROUTE: analyst" -- to send to the analyst
- "ROUTE: writer" -- to send to the writer
- "ROUTE: FINISH" -- when the task is complete

After your routing decision, you may add a brief instruction for the specialist on a new line.

Example responses:
  ROUTE: researcher
  Search for the latest GDP figures for France and Germany.

  ROUTE: FINISH
  The writer has provided a complete answer.
"""

RESEARCHER_PROMPT: str = """You are a research specialist. Your job is to find accurate, relevant information using web search.

GUIDELINES:
- Formulate clear, specific search queries
- Search for multiple aspects if the topic is complex
- Summarize the KEY FACTS you found (don't dump raw search results)
- Always cite your sources (include URLs when available)
- If the first search doesn't find what you need, try rephrasing the query
- Focus on recent, authoritative sources

Respond with a clear summary of your research findings."""

ANALYST_PROMPT: str = """You are an analysis specialist. Your job is to perform quantitative analysis, calculations, and logical reasoning.

GUIDELINES:
- Use the calculator tool for ALL math operations (don't do mental math)
- Break complex calculations into steps
- Show your work -- explain each calculation
- Compare data points when relevant
- Look for patterns and insights in the data
- Be precise with numbers (include units, significant figures)
- If you need data you don't have, say so clearly

Respond with a structured analysis of the data."""

WRITER_PROMPT: str = """You are a writing specialist. Your job is to synthesize information into clear, well-structured responses.

GUIDELINES:
- Organize information logically (use headers, bullet points, numbered lists)
- Lead with the most important information (inverted pyramid style)
- Use clear, concise language (avoid jargon unless the topic requires it)
- Incorporate data and facts from the research and analysis
- Provide context and explanations where needed
- End with a clear conclusion or summary
- Cite sources when available

Write a polished, comprehensive response to the user's original question."""


# ---------------------------------------------------------------------------
# Tool sets for each specialist
# ---------------------------------------------------------------------------
# Each specialist gets only the tools relevant to their role.
# This is a form of "least privilege" -- the writer doesn't need search,
# the researcher doesn't need the calculator.

RESEARCHER_TOOLS: list[Any] = [web_search, web_scrape]
ANALYST_TOOLS: list[Any] = [calculator]
WRITER_TOOLS: list[Any] = []  # The writer works from information in the state


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def supervisor_node(state: MultiAgentState) -> dict[str, Any]:
    """The supervisor reads the current state and decides who works next.

    The supervisor LLM sees:
      - The original user query
      - Outputs from any specialists who have already worked
      - The full message history

    It then decides which specialist should work next, or whether
    the task is complete (FINISH).

    ROUTING LOGIC:
      The supervisor's response must start with "ROUTE: <target>".
      We parse this to determine the next_agent. If parsing fails,
      we default to the writer (to produce SOME output).

    Args:
        state: Current multi-agent state.

    Returns:
        State update with the routing decision.
    """
    settings = Settings()
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
    )

    # Build context for the supervisor
    specialist_context = ""
    outputs = state.get("specialist_outputs", {})
    if outputs:
        specialist_context = "\n\nSpecialist outputs so far:\n"
        for agent_name, output in outputs.items():
            specialist_context += f"\n--- {agent_name.upper()} ---\n{output}\n"

    messages = [
        SystemMessage(content=SUPERVISOR_PROMPT + specialist_context),
        *list(state["messages"]),
    ]

    logger.info(
        "supervisor_routing",
        iteration=state.get("iteration", 0),
        specialists_completed=list(outputs.keys()),
    )

    response: AIMessage = llm.invoke(messages)
    response_text = response.content or ""

    # Parse the routing decision from the supervisor's response.
    # We look for "ROUTE: <target>" at the start of the response.
    next_agent = _parse_routing(response_text)

    logger.info("supervisor_decision", next_agent=next_agent)

    return {
        "messages": [AIMessage(content=f"[Supervisor] Routing to: {next_agent}. {response_text}")],
        "next_agent": next_agent,
        "iteration": state.get("iteration", 0) + 1,
    }


def _parse_routing(response: str) -> str:
    """Parse the supervisor's routing decision from its response text.

    Expected format: "ROUTE: <agent_name>"

    We're lenient in parsing -- we look for the pattern anywhere in the
    response (not just the start) because LLMs sometimes add preamble.

    Args:
        response: The supervisor's full response text.

    Returns:
        The target agent name, or "FINISH" if we can't parse it.
    """
    response_lower = response.lower().strip()

    # Check each possible routing target
    for target in SPECIALISTS + [FINISH]:
        # Look for "route: <target>" pattern (case-insensitive)
        if f"route: {target.lower()}" in response_lower:
            return target

    # If we can't parse the routing, check for keywords as fallback
    if "research" in response_lower or "search" in response_lower:
        return RESEARCHER
    if "analy" in response_lower or "calcul" in response_lower:
        return ANALYST
    if "writ" in response_lower or "draft" in response_lower:
        return WRITER
    if "finish" in response_lower or "done" in response_lower or "complete" in response_lower:
        return FINISH

    # Last resort: finish to avoid infinite loops
    logger.warning("supervisor_routing_unparseable", response=response[:200])
    return FINISH


def researcher_node(state: MultiAgentState) -> dict[str, Any]:
    """The researcher specialist: searches the web for information.

    This node creates a ChatOllama instance with search tools bound,
    processes the conversation with the research-focused system prompt,
    and stores its output in the specialist_outputs dict.

    Args:
        state: Current multi-agent state.

    Returns:
        State update with research findings.
    """
    return _run_specialist(
        state=state,
        name=RESEARCHER,
        system_prompt=RESEARCHER_PROMPT,
        tools=RESEARCHER_TOOLS,
    )


def analyst_node(state: MultiAgentState) -> dict[str, Any]:
    """The analyst specialist: performs calculations and analysis.

    Args:
        state: Current multi-agent state.

    Returns:
        State update with analysis results.
    """
    return _run_specialist(
        state=state,
        name=ANALYST,
        system_prompt=ANALYST_PROMPT,
        tools=ANALYST_TOOLS,
    )


def writer_node(state: MultiAgentState) -> dict[str, Any]:
    """The writer specialist: synthesizes a polished response.

    The writer sees all previous specialist outputs and crafts them
    into a final, well-structured answer.

    Args:
        state: Current multi-agent state.

    Returns:
        State update with the drafted response.
    """
    return _run_specialist(
        state=state,
        name=WRITER,
        system_prompt=WRITER_PROMPT,
        tools=WRITER_TOOLS,
    )


def _run_specialist(
    state: MultiAgentState,
    name: str,
    system_prompt: str,
    tools: list[Any],
) -> dict[str, Any]:
    """Generic function to run any specialist agent.

    DRY PRINCIPLE:
      All specialists follow the same pattern: take state, invoke LLM
      with tools, store output. Rather than duplicate this logic three
      times, we parameterize it.

    TOOL EXECUTION LOOP:
      Specialists may need to call tools (search, calculator). We run
      a simple loop: invoke LLM -> if tool calls, execute them and
      loop back -> if no tool calls, we're done.

    Args:
        state: Current multi-agent state.
        name: Specialist name (for logging and output storage).
        system_prompt: The specialist's system prompt.
        tools: Tools available to this specialist.

    Returns:
        State update with the specialist's output.
    """
    settings = Settings()
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
    )

    # Build context: include previous specialist outputs so each
    # specialist can build on what came before.
    context = ""
    outputs = state.get("specialist_outputs", {})
    if outputs:
        context = "\n\nPrevious specialist findings:\n"
        for agent_name, output in outputs.items():
            context += f"\n[{agent_name.upper()}]: {output}\n"

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt + context),
        # Include the original user query and relevant history
        *[m for m in state["messages"] if isinstance(m, HumanMessage)],
    ]

    # If this specialist has tools, bind them and run the tool loop
    if tools:
        llm_with_tools = llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        # Tool execution loop (max 5 rounds to prevent infinite loops)
        for _ in range(5):
            response: AIMessage = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                # No more tools to call -- specialist is done
                break

            # Execute each tool call
            for tc in response.tool_calls:
                logger.info(
                    "specialist_tool_call",
                    specialist=name,
                    tool=tc["name"],
                    args=tc["args"],
                )
                if tc["name"] in tool_map:
                    try:
                        result = tool_map[tc["name"]].invoke(tc["args"])
                    except Exception as e:
                        result = f"Tool error: {e}"
                else:
                    result = f"Unknown tool: {tc['name']}"

                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    )
                )
    else:
        # No tools -- just invoke the LLM directly
        response = llm.invoke(messages)

    specialist_output = response.content or f"[{name} produced no output]"

    logger.info(
        "specialist_completed",
        specialist=name,
        output_length=len(specialist_output),
    )

    # Update the specialist outputs dict
    updated_outputs = dict(outputs)
    updated_outputs[name] = specialist_output

    return {
        "messages": [
            AIMessage(content=f"[{name.upper()}] {specialist_output}")
        ],
        "specialist_outputs": updated_outputs,
    }


# ---------------------------------------------------------------------------
# Routing edge function
# ---------------------------------------------------------------------------

def route_from_supervisor(state: MultiAgentState) -> str:
    """Conditional edge: route from supervisor to the chosen specialist.

    This function reads the ``next_agent`` field set by the supervisor
    and returns the corresponding node name. LangGraph uses the return
    value to decide which node to execute next.

    Args:
        state: Current multi-agent state.

    Returns:
        Node name to route to, or END to finish.
    """
    next_agent = state.get("next_agent", FINISH)
    iteration = state.get("iteration", 0)

    # Safety valve: max 8 routing rounds
    if iteration >= 8:
        logger.warning("multi_agent_max_iterations", iteration=iteration)
        return END

    if next_agent == FINISH:
        return END

    if next_agent in SPECIALISTS:
        return next_agent

    # Unknown target -- finish rather than crash
    logger.warning("unknown_routing_target", target=next_agent)
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_multi_agent_graph() -> StateGraph:
    """Build the multi-agent supervisor graph.

    GRAPH STRUCTURE:
        ┌──────────┐
        │  START    │
        └────┬─────┘
             │
             ▼
        ┌────────────┐    researcher    ┌────────────┐
        │ SUPERVISOR  │───────────────→│ RESEARCHER  │──┐
        │             │    analyst      ┌────────────┐  │
        │  (routes to │───────────────→│  ANALYST    │──┤
        │   specialist│    writer       ┌────────────┐  │
        │   or finish)│───────────────→│   WRITER    │──┤
        └──────┬──────┘                                  │
               │                                         │
               │ FINISH                                  │
               ▼                 (all loop back)         │
          ┌─────────┐        ◄───────────────────────────┘
          │   END   │
          └─────────┘

    Returns:
        Compiled LangGraph StateGraph.
    """
    graph = StateGraph(MultiAgentState)

    # Add all nodes
    graph.add_node(SUPERVISOR, supervisor_node)
    graph.add_node(RESEARCHER, researcher_node)
    graph.add_node(ANALYST, analyst_node)
    graph.add_node(WRITER, writer_node)

    # Start with the supervisor
    graph.set_entry_point(SUPERVISOR)

    # Supervisor routes to a specialist or END
    graph.add_conditional_edges(
        SUPERVISOR,
        route_from_supervisor,
        {
            RESEARCHER: RESEARCHER,
            ANALYST: ANALYST,
            WRITER: WRITER,
            END: END,
        },
    )

    # All specialists route back to the supervisor after completing.
    # The supervisor then decides the next step.
    graph.add_edge(RESEARCHER, SUPERVISOR)
    graph.add_edge(ANALYST, SUPERVISOR)
    graph.add_edge(WRITER, SUPERVISOR)

    compiled = graph.compile()

    logger.info(
        "multi_agent_graph_built",
        nodes=[SUPERVISOR, RESEARCHER, ANALYST, WRITER],
    )

    return compiled


# ---------------------------------------------------------------------------
# MultiAgentSupervisor class -- public API
# ---------------------------------------------------------------------------

@dataclass
class MultiAgentSupervisor:
    """A multi-agent system with a supervisor that routes to specialists.

    This is the most sophisticated agent architecture in AgentExplorr.
    The supervisor reads the user's query and delegates to specialist
    agents (researcher, analyst, writer) in whatever order makes sense.

    USAGE:
        >>> from agentexplorr.agents.multi_agent import MultiAgentSupervisor
        >>> system = MultiAgentSupervisor()
        >>> result = system.run(
        ...     "Compare the populations of Tokyo and New York City, "
        ...     "and analyze which is growing faster."
        ... )
        >>> print(result.final_answer)

    HOW IT WORKS:
        1. Supervisor reads the query and decides: "I need the researcher
           to find population data first."
        2. Researcher searches the web and returns population figures.
        3. Supervisor reads the research and decides: "Now I need the
           analyst to compare and analyze the numbers."
        4. Analyst performs calculations and comparison.
        5. Supervisor decides: "Now the writer should draft the final response."
        6. Writer synthesizes everything into a polished answer.
        7. Supervisor decides: "FINISH -- the answer is complete."

    Attributes:
        model: Ollama model name.
        base_url: Ollama server URL.
        max_iterations: Maximum supervisor routing rounds.
        verbose: If True, print detailed execution trace.
    """

    model: str = ""
    base_url: str = ""
    max_iterations: int = 8
    verbose: bool = False
    _graph: Any = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        """Initialize settings and build the multi-agent graph."""
        settings = Settings()
        if not self.model:
            self.model = settings.ollama_model
        if not self.base_url:
            self.base_url = settings.ollama_base_url

        self._graph = build_multi_agent_graph()

        logger.info(
            "multi_agent_system_initialized",
            model=self.model,
            specialists=SPECIALISTS,
        )

    def run(self, query: str) -> MultiAgentResult:
        """Run the multi-agent system on a user query.

        The supervisor coordinates the specialists to produce a
        comprehensive answer. The execution flow depends on the query --
        simple questions might only need the researcher, while complex
        questions might involve all three specialists.

        Args:
            query: The user's question or task.

        Returns:
            A MultiAgentResult with the answer and execution details.
        """
        logger.info("multi_agent_run", query=query[:100])

        initial_state: MultiAgentState = {
            "messages": [HumanMessage(content=query)],
            "next_agent": "",
            "specialist_outputs": {},
            "iteration": 0,
        }

        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as e:
            logger.error("multi_agent_error", error=str(e))
            return MultiAgentResult(
                query=query,
                final_answer=f"Multi-agent system error: {e}",
                specialist_outputs={},
                routing_history=[],
                iterations=0,
                success=False,
            )

        messages = final_state.get("messages", [])
        specialist_outputs = final_state.get("specialist_outputs", {})
        iterations = final_state.get("iteration", 0)

        # The final answer is the last substantive AI message.
        # Prioritize the writer's output if available.
        final_answer = ""
        if WRITER in specialist_outputs:
            final_answer = specialist_outputs[WRITER]
        else:
            # Fall back to the last AI message content
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    content = msg.content
                    # Skip supervisor routing messages
                    if not content.startswith("[Supervisor]"):
                        final_answer = content
                        break

        # Extract routing history from supervisor messages
        routing_history: list[str] = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.content:
                if "[Supervisor] Routing to:" in msg.content:
                    routing_history.append(msg.content)

        if self.verbose:
            self._print_trace(messages, routing_history)

        result = MultiAgentResult(
            query=query,
            final_answer=final_answer,
            specialist_outputs=specialist_outputs,
            routing_history=routing_history,
            iterations=iterations,
            success=True,
        )

        logger.info(
            "multi_agent_completed",
            iterations=iterations,
            specialists_used=list(specialist_outputs.keys()),
        )

        return result

    def _print_trace(
        self,
        messages: Sequence[BaseMessage],
        routing_history: list[str],
    ) -> None:
        """Print a detailed execution trace.

        Args:
            messages: Full message history.
            routing_history: Supervisor routing decisions.
        """
        print("\n" + "=" * 60)
        print("Multi-Agent Execution Trace")
        print("=" * 60)

        print(f"\nRouting history ({len(routing_history)} steps):")
        for i, route in enumerate(routing_history, 1):
            print(f"  {i}. {route}")

        print("\nMessage flow:")
        for msg in messages:
            if isinstance(msg, HumanMessage):
                print(f"\n  [USER] {msg.content[:100]}")
            elif isinstance(msg, AIMessage) and msg.content:
                # Identify the speaker
                content = msg.content
                if content.startswith("["):
                    bracket_end = content.find("]")
                    speaker = content[1:bracket_end] if bracket_end > 0 else "AI"
                    text = content[bracket_end + 2:] if bracket_end > 0 else content
                    print(f"\n  [{speaker}] {text[:150]}...")
                else:
                    print(f"\n  [AI] {content[:150]}...")
            elif isinstance(msg, ToolMessage):
                print(f"\n  [TOOL: {msg.name}] {msg.content[:100]}...")

        print("\n" + "=" * 60)


@dataclass
class MultiAgentResult:
    """Result from a multi-agent system run.

    Attributes:
        query: The original user query.
        final_answer: The synthesized final answer.
        specialist_outputs: Raw output from each specialist that participated.
        routing_history: Record of supervisor routing decisions.
        iterations: Total number of supervisor routing rounds.
        success: Whether the system completed without errors.
    """

    query: str
    final_answer: str
    specialist_outputs: dict[str, str]
    routing_history: list[str]
    iterations: int
    success: bool = True

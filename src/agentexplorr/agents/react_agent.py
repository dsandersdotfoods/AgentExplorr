"""
ReAct Agent -- Reasoning + Acting with LangGraph
==================================================

WHAT IS ReAct?
  ReAct (Reasoning + Acting) is one of the most influential agent architectures.
  Instead of just generating a final answer, the agent alternates between:

    1. **THINK** -- Reason about the current state and what to do next
    2. **ACT**   -- Choose and execute a tool (search, calculate, etc.)
    3. **OBSERVE** -- Read the tool's output and incorporate it

  This loop continues until the agent has enough information to produce a
  final answer. The key insight is that INTERLEAVING reasoning with action
  leads to much better results than either alone.

WHY ReAct WORKS:
  - **Reasoning alone** (Chain of Thought) -- The LLM can make things up
    because it has no way to verify facts. It "hallucinates" confidently.
  - **Acting alone** (tool use without reasoning) -- The agent may call
    tools randomly without a plan, wasting steps and missing the point.
  - **ReAct combines both** -- The reasoning step creates a PLAN, the
    action step EXECUTES it, and the observation step GROUNDS the agent
    in real data. This feedback loop is remarkably effective.

THE ReAct LOOP (visualized):
  ┌─────────────────────────────────────────────────┐
  │                  User Query                      │
  │              "What is the GDP of                 │
  │               France in 2024?"                   │
  └──────────────────┬──────────────────────────────┘
                     │
                     ▼
  ┌──────────────────────────────────────────────────┐
  │  THINK: I need to search for France's GDP in     │
  │  2024. Let me use the web_search tool.           │
  └──────────────────┬───────────────────────────────┘
                     │
                     ▼
  ┌──────────────────────────────────────────────────┐
  │  ACT: web_search("France GDP 2024")              │
  └──────────────────┬───────────────────────────────┘
                     │
                     ▼
  ┌──────────────────────────────────────────────────┐
  │  OBSERVE: "France's GDP in 2024 was              │
  │  approximately $3.1 trillion..."                  │
  └──────────────────┬───────────────────────────────┘
                     │
                     ▼
  ┌──────────────────────────────────────────────────┐
  │  THINK: I now have the answer. France's GDP      │
  │  in 2024 was approximately $3.1 trillion.        │
  │  → Return final answer                           │
  └──────────────────────────────────────────────────┘

IMPLEMENTATION WITH LANGGRAPH:
  LangGraph lets us define the ReAct loop as a **state machine** (graph):
    - **Nodes** are functions that process the current state
    - **Edges** define transitions between nodes
    - **Conditional edges** let us branch (e.g., "if the agent wants to
      use a tool, go to the action node; otherwise, finish")
    - **State** is a TypedDict that flows through the graph

  This is more explicit and debuggable than LangChain's older AgentExecutor,
  which was a black box. With LangGraph, you can see exactly how the agent
  transitions between states.

LEARNING RESOURCES:
  - PAPER: "ReAct: Synergizing Reasoning and Acting" (Yao et al., 2022)
           https://arxiv.org/abs/2210.03629
  - LangGraph docs: https://langchain-ai.github.io/langgraph/
  - LangGraph ReAct tutorial: https://langchain-ai.github.io/langgraph/tutorials/introduction/
  - VIDEO: "ReAct Agents Explained" -- https://www.youtube.com/watch?v=Eug2clsLtFs
  - VIDEO: "Build AI Agents with LangGraph" -- https://www.youtube.com/watch?v=v9fkbTxPzs0
  - VIDEO: "LangGraph Crash Course" -- https://www.youtube.com/watch?v=R-o_a6F-SOQ
  - LangChain-Ollama docs: https://python.langchain.com/docs/integrations/chat/ollama/
  - Ollama model library: https://ollama.com/library
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

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
from agentexplorr.core import Settings, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------
# LangGraph uses TypedDict to define the state that flows through the graph.
# The Annotated[..., add_messages] tells LangGraph to APPEND new messages
# to the list rather than replacing it. This is crucial because each step
# in the ReAct loop adds messages (think, act, observe) to the conversation.

class AgentState(TypedDict):
    """State that flows through the ReAct agent graph.

    Attributes:
        messages: The full conversation history. LangGraph's ``add_messages``
                 reducer appends new messages rather than overwriting.
        iteration: Current iteration count (to enforce max iterations).
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration: int


# ---------------------------------------------------------------------------
# System prompt for ReAct reasoning
# ---------------------------------------------------------------------------
# This prompt is critical -- it teaches the LLM HOW to reason in the
# Think-Act-Observe pattern. Without it, the LLM would just answer directly.

REACT_SYSTEM_PROMPT: str = """You are a helpful AI assistant that follows the ReAct (Reasoning + Acting) framework.

You have access to the following tools:
{tool_descriptions}

For each user question, follow this exact pattern:

THINK: <analyze the question, decide what information you need, plan your approach>
ACT: <call a tool if you need information or computation>
OBSERVE: <read the tool result and incorporate it into your understanding>

Repeat the Think-Act-Observe cycle as needed until you have enough information.

When you have the final answer, respond with:
FINAL ANSWER: <your complete, well-formatted answer>

IMPORTANT RULES:
1. Always THINK before acting -- explain your reasoning
2. Use tools when you need factual information or calculations
3. Don't guess -- if you need data, search for it
4. After observing tool results, THINK about whether you have enough info
5. Keep your thinking concise but thorough
6. If a tool returns an error, THINK about an alternative approach
"""


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
# We collect all available tools in a list so we can:
# 1. Pass them to the LLM (via bind_tools)
# 2. Look them up by name when executing tool calls

AVAILABLE_TOOLS: list[Any] = [web_search, calculator]

# Build a name -> tool mapping for quick lookup during execution
TOOL_MAP: dict[str, Any] = {t.name: t for t in AVAILABLE_TOOLS}


def _get_tool_descriptions() -> str:
    """Generate human-readable tool descriptions for the system prompt.

    Each tool's name, description, and argument schema are formatted
    into a string that the LLM can read. This helps the LLM understand
    WHAT each tool does and WHEN to use it.

    Returns:
        Formatted string of tool descriptions.
    """
    descriptions: list[str] = []
    for t in AVAILABLE_TOOLS:
        # LangChain tools have .name, .description, and .args_schema
        desc = f"- {t.name}: {t.description}"
        descriptions.append(desc)
    return "\n".join(descriptions)


# ---------------------------------------------------------------------------
# Graph node functions
# ---------------------------------------------------------------------------
# Each function below is a "node" in the LangGraph state machine.
# Nodes receive the current state, do some work, and return updates
# that get merged back into the state.


def reasoning_node(state: AgentState) -> dict[str, Any]:
    """The THINK step: invoke the LLM to reason about the current state.

    This node sends the full message history to the LLM. The LLM either:
      a) Decides to call a tool (returns an AIMessage with tool_calls)
      b) Decides it has the answer (returns an AIMessage with final text)

    The LLM has tools bound to it via ``bind_tools()``, so it can express
    tool calls in a structured format that LangGraph can parse.

    HOW bind_tools() WORKS:
      When you call ``llm.bind_tools(tools)``, it modifies the LLM's
      behavior to include tool schemas in the prompt. The LLM can then
      output structured JSON specifying which tool to call and with what
      arguments. This is more reliable than parsing free-text tool calls.

    Args:
        state: Current agent state with message history.

    Returns:
        State update with the LLM's response added to messages.
    """
    settings = Settings()
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,  # Low temperature for more deterministic reasoning
    )

    # Bind tools to the LLM so it can make structured tool calls.
    # This is the LangChain way of enabling "function calling" with
    # any model that supports it (OpenAI, Ollama, Anthropic, etc.)
    llm_with_tools = llm.bind_tools(AVAILABLE_TOOLS)

    # Build the system prompt with tool descriptions
    system_prompt = REACT_SYSTEM_PROMPT.format(
        tool_descriptions=_get_tool_descriptions()
    )

    # Ensure the system prompt is the first message
    messages = list(state["messages"])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=system_prompt))

    logger.info(
        "react_reasoning",
        iteration=state.get("iteration", 0),
        num_messages=len(messages),
    )

    # Invoke the LLM. This returns an AIMessage that may contain:
    # - Just text (the agent's reasoning or final answer)
    # - Tool calls (structured requests to execute tools)
    # - Both text AND tool calls
    response: AIMessage = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "iteration": state.get("iteration", 0) + 1,
    }


def action_node(state: AgentState) -> dict[str, Any]:
    """The ACT + OBSERVE step: execute tool calls and return results.

    This node looks at the last AIMessage for tool_calls. For each tool
    call, it:
      1. Looks up the tool by name in our TOOL_MAP
      2. Executes the tool with the provided arguments
      3. Creates a ToolMessage with the result

    ToolMessage is a special LangChain message type that the LLM understands
    as "the result of a tool I called." This is how the observation gets
    fed back into the reasoning loop.

    Args:
        state: Current agent state. The last message should be an AIMessage
              with tool_calls.

    Returns:
        State update with ToolMessage(s) added to messages.
    """
    messages = state["messages"]
    last_message = messages[-1]

    # Safety check: this node should only be called when there are tool calls
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.warning("action_node_called_without_tool_calls")
        return {"messages": []}

    tool_messages: list[ToolMessage] = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        logger.info(
            "react_executing_tool",
            tool=tool_name,
            args=tool_args,
            iteration=state.get("iteration", 0),
        )

        # Look up the tool and execute it
        if tool_name in TOOL_MAP:
            try:
                result = TOOL_MAP[tool_name].invoke(tool_args)
            except Exception as e:
                result = f"Tool execution error: {e}"
                logger.error(
                    "tool_execution_error",
                    tool=tool_name,
                    error=str(e),
                )
        else:
            result = f"Unknown tool: '{tool_name}'. Available tools: {list(TOOL_MAP.keys())}"
            logger.warning("unknown_tool_called", tool=tool_name)

        # Create a ToolMessage with the result. The tool_call_id links
        # this result back to the specific tool call in the AIMessage.
        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )

    return {"messages": tool_messages}


def should_continue(state: AgentState) -> Literal["action", "end"]:
    """Conditional edge: decide whether to execute tools or finish.

    This function is called after the reasoning_node to determine the
    next step in the graph:
      - If the LLM returned tool calls -> go to action_node
      - If the LLM returned a final answer (no tool calls) -> END
      - If we've exceeded max iterations -> END (safety limit)

    WHY MAX ITERATIONS?
      Without a limit, a confused agent could loop forever. The max
      iteration limit is a safety valve. In practice, most queries
      resolve in 1-3 iterations. If the agent needs more than 10,
      something is probably wrong.

    Args:
        state: Current agent state.

    Returns:
        "action" to execute tools, or "end" to finish.
    """
    messages = state["messages"]
    last_message = messages[-1]
    iteration = state.get("iteration", 0)

    # Safety valve: stop after too many iterations
    max_iterations = 10
    if iteration >= max_iterations:
        logger.warning(
            "react_max_iterations_reached",
            iteration=iteration,
            max_iterations=max_iterations,
        )
        return "end"

    # If the last message has tool calls, execute them
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "action"

    # Otherwise, the agent has finished reasoning
    return "end"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_react_graph() -> StateGraph:
    """Build the ReAct agent as a LangGraph StateGraph.

    LANGGRAPH GRAPH CONSTRUCTION:
      1. Create a StateGraph with our state schema
      2. Add nodes (reasoning, action)
      3. Add edges (including conditional edges for branching)
      4. Set the entry point
      5. Compile the graph into a runnable

    The resulting graph looks like:

        ┌─────────┐
        │  START   │
        └────┬─────┘
             │
             ▼
        ┌─────────────┐    tool_calls    ┌──────────────┐
        │  reasoning   │───────────────→│    action     │
        │   (THINK)    │                 │  (ACT+OBSERVE)│
        └──────┬───────┘                 └───────┬───────┘
               │                                  │
               │ no tool calls                    │ (always loops back)
               │                                  │
               ▼                                  │
          ┌─────────┐                             │
          │   END   │     ◄───────────────────────┘
          └─────────┘

    Returns:
        A compiled LangGraph StateGraph ready to invoke.
    """
    # Step 1: Create the graph with our state schema
    graph = StateGraph(AgentState)

    # Step 2: Add nodes
    # Each node is a function that receives state and returns state updates.
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("action", action_node)

    # Step 3: Set the entry point -- where the graph starts executing
    graph.set_entry_point("reasoning")

    # Step 4: Add conditional edge from reasoning
    # After reasoning, we check should_continue() to decide next step:
    #   "action" -> go to action node
    #   "end" -> go to END (stop the graph)
    graph.add_conditional_edges(
        "reasoning",
        should_continue,
        {
            "action": "action",
            "end": END,
        },
    )

    # Step 5: After action, always go back to reasoning
    # This creates the loop: Think -> Act -> Observe -> Think -> ...
    graph.add_edge("action", "reasoning")

    # Step 6: Compile the graph
    # Compilation validates the graph structure and returns a runnable object.
    compiled = graph.compile()

    logger.info("react_graph_built", nodes=["reasoning", "action"])
    return compiled


# ---------------------------------------------------------------------------
# ReActAgent class -- the main public API
# ---------------------------------------------------------------------------

@dataclass
class ReActAgent:
    """A ReAct (Reasoning + Acting) agent built with LangGraph.

    This agent follows the Think-Act-Observe loop from the original ReAct
    paper (Yao et al., 2022). It uses LangGraph for orchestration and
    ChatOllama for LLM inference.

    USAGE:
        >>> from agentexplorr.agents.react_agent import ReActAgent
        >>> agent = ReActAgent()
        >>> result = agent.run("What is the population of Tokyo?")
        >>> print(result.final_answer)

    HOW IT WORKS:
        1. User query is wrapped in a HumanMessage
        2. The reasoning node (LLM) processes the message and decides:
           - Call a tool for more info -> action node executes the tool
           - Provide final answer -> graph ends
        3. The loop continues until the agent has enough info or hits
           the max iteration limit

    PAPER REFERENCE:
        Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y.
        (2022). ReAct: Synergizing Reasoning and Acting in Language Models.
        https://arxiv.org/abs/2210.03629

    Attributes:
        model: Ollama model name (e.g., "llama3.2", "mistral").
        base_url: Ollama server URL.
        max_iterations: Safety limit on reasoning iterations.
        verbose: If True, print intermediate reasoning steps.
    """

    model: str = ""
    base_url: str = ""
    max_iterations: int = 10
    verbose: bool = False
    _graph: Any = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        """Initialize the agent, loading settings and building the graph."""
        settings = Settings()
        if not self.model:
            self.model = settings.ollama_model
        if not self.base_url:
            self.base_url = settings.ollama_base_url

        self._graph = build_react_graph()
        logger.info(
            "react_agent_initialized",
            model=self.model,
            max_iterations=self.max_iterations,
        )

    def run(self, query: str) -> ReActResult:
        """Run the ReAct agent on a user query.

        This is the main entry point. It:
          1. Creates the initial state with the user's query
          2. Invokes the compiled LangGraph
          3. Extracts and returns the result

        Args:
            query: The user's question or task.

        Returns:
            A ReActResult containing the answer and execution metadata.

        Example:
            >>> agent = ReActAgent(model="llama3.2")
            >>> result = agent.run("What is 25 * 17 + 3?")
            >>> print(result.final_answer)
            428
        """
        logger.info("react_agent_run", query=query[:100])

        # Build the initial state. LangGraph needs the state to match
        # our AgentState TypedDict schema.
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "iteration": 0,
        }

        # Invoke the graph. This runs the full Think-Act-Observe loop
        # until the should_continue edge returns "end".
        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as e:
            logger.error("react_agent_error", query=query[:100], error=str(e))
            return ReActResult(
                query=query,
                final_answer=f"Agent encountered an error: {e}",
                messages=[],
                iterations=0,
                tools_used=[],
                success=False,
            )

        # Extract results from the final state
        messages = final_state.get("messages", [])
        iterations = final_state.get("iteration", 0)

        # The final answer is the content of the last AIMessage
        final_answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_answer = msg.content
                break

        # Collect which tools were used (useful for evaluation/debugging)
        tools_used: list[str] = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_used.append(tc["name"])

        # Print intermediate steps if verbose
        if self.verbose:
            self._print_trace(messages)

        result = ReActResult(
            query=query,
            final_answer=final_answer,
            messages=list(messages),
            iterations=iterations,
            tools_used=tools_used,
            success=True,
        )

        logger.info(
            "react_agent_completed",
            iterations=iterations,
            tools_used=tools_used,
            answer_length=len(final_answer),
        )

        return result

    def _print_trace(self, messages: Sequence[BaseMessage]) -> None:
        """Print a human-readable trace of the agent's reasoning.

        This is invaluable for debugging and understanding HOW the agent
        arrived at its answer. Each message type is color-coded:
          - System: gray
          - Human: blue
          - AI (thinking): green
          - Tool result: yellow

        Args:
            messages: The full message history from the agent run.
        """
        print("\n" + "=" * 60)
        print("ReAct Agent Trace")
        print("=" * 60)

        for msg in messages:
            if isinstance(msg, SystemMessage):
                print(f"\n[SYSTEM] {msg.content[:100]}...")
            elif isinstance(msg, HumanMessage):
                print(f"\n[USER] {msg.content}")
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"\n[THINK+ACT] Calling {tc['name']}({tc['args']})")
                if msg.content:
                    print(f"\n[THINK] {msg.content}")
            elif isinstance(msg, ToolMessage):
                content_preview = msg.content[:200] if msg.content else ""
                print(f"\n[OBSERVE] ({msg.name}) {content_preview}")

        print("\n" + "=" * 60)


@dataclass
class ReActResult:
    """The result of a ReAct agent run.

    This dataclass bundles together everything you need to understand
    what happened during an agent run: the answer, the full message
    history, performance metrics, and success status.

    Attributes:
        query: The original user query.
        final_answer: The agent's final response.
        messages: Full conversation history (for debugging/analysis).
        iterations: How many Think-Act-Observe loops were executed.
        tools_used: List of tool names that were called.
        success: Whether the agent completed without errors.
    """

    query: str
    final_answer: str
    messages: list[BaseMessage]
    iterations: int
    tools_used: list[str]
    success: bool = True

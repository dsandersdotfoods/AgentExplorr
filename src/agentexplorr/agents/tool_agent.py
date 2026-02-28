"""
Tool-Calling Agent -- Simple Tool Dispatch with ChatOllama
============================================================

WHAT IS A TOOL-CALLING AGENT?
  A tool-calling agent is the simplest form of agentic behavior. The LLM
  receives a user query along with a list of available tools, and directly
  decides which tool(s) to call. There is NO explicit "think" step or
  iterative reasoning loop -- the LLM handles it all in one shot.

TOOL-CALLING vs. ReAct:
  ┌─────────────────┬───────────────────┬───────────────────────┐
  │                  │  Tool-Calling      │  ReAct                │
  ├─────────────────┼───────────────────┼───────────────────────┤
  │ Complexity       │  Simple (1-2 steps)│  Multi-step loop      │
  │ Reasoning        │  Implicit (in LLM) │  Explicit (THINK step)│
  │ Best for         │  Direct queries    │  Complex research     │
  │ Latency          │  Low (fewer LLM    │  Higher (multiple LLM │
  │                  │  calls)            │  calls per loop)      │
  │ Debuggability    │  Moderate          │  High (visible trace) │
  │ When to use      │  "What's 2+2?"     │  "Compare GDP of 3    │
  │                  │  "Search for X"    │  countries and analyze"│
  └─────────────────┴───────────────────┴───────────────────────┘

  Rule of thumb: Use tool-calling for single-tool queries, ReAct for
  multi-step research tasks.

HOW TOOL CALLING WORKS (under the hood):
  1. The LLM receives the user query + tool schemas (JSON schema of each tool)
  2. The LLM generates a STRUCTURED response indicating:
     - Which tool to call (by name)
     - What arguments to pass (matching the schema)
  3. The framework parses this structured output
  4. The tool function is executed with the parsed arguments
  5. The tool result is sent back to the LLM for final response generation

  With Ollama, tool calling works via the model's function-calling capability.
  Not all models support it -- llama3.2, mistral, and command-r do.

IMPLEMENTATION:
  We use a simple LangGraph graph with two nodes:
    - ``call_model``: Send the query to the LLM (with bound tools)
    - ``call_tools``: Execute any tool calls the LLM made
  The graph loops between these until the LLM stops calling tools.

LEARNING RESOURCES:
  - LangChain Tool Calling: https://python.langchain.com/docs/concepts/tool_calling/
  - LangChain @tool decorator: https://python.langchain.com/docs/how_to/custom_tools/
  - Ollama function calling: https://ollama.com/blog/tool-support
  - VIDEO: "Function Calling with LangChain" -- https://www.youtube.com/watch?v=p9v2fHLmQCU
  - VIDEO: "AI Agents Tool Use Explained" -- https://www.youtube.com/watch?v=cN9S6CYjmi8
  - LangGraph Tool Calling: https://langchain-ai.github.io/langgraph/how-tos/tool-calling/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agentexplorr.agents.tools.calculator import calculator
from agentexplorr.agents.tools.search import web_search
from agentexplorr.agents.tools.web_scraper import web_scrape
from agentexplorr.core import Settings, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Additional custom tools (demonstrating the @tool decorator)
# ---------------------------------------------------------------------------
# These tools showcase how easy it is to add new capabilities to an agent.
# The @tool decorator converts a plain Python function into a LangChain
# tool with automatic schema generation from the type hints + docstring.

@tool
def get_current_time() -> str:
    """Get the current date and time.

    Use this tool when the user asks about the current time, date,
    or needs a timestamp. Returns the time in a human-readable format.

    Returns:
        Current date and time as a formatted string.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S UTC")


@tool
def text_length(text: str) -> str:
    """Count the number of characters and words in a text.

    Use this tool to analyze the length of a piece of text.
    Useful for word count checks, character limits, etc.

    Args:
        text: The text to analyze.

    Returns:
        A summary of text length statistics.
    """
    char_count = len(text)
    word_count = len(text.split())
    line_count = len(text.splitlines())

    return (
        f"Characters: {char_count}\n"
        f"Words: {word_count}\n"
        f"Lines: {line_count}"
    )


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between common units.

    Supports temperature (C/F/K), distance (km/mi/m/ft), and weight (kg/lb/g/oz).

    Args:
        value: The numeric value to convert.
        from_unit: The source unit (e.g., "celsius", "km", "kg").
        to_unit: The target unit (e.g., "fahrenheit", "mi", "lb").

    Returns:
        The converted value as a formatted string, or an error message.
    """
    # Normalize unit names to lowercase for matching
    from_u = from_unit.lower().strip()
    to_u = to_unit.lower().strip()

    # Temperature conversions
    temp_aliases = {
        "c": "celsius", "celsius": "celsius",
        "f": "fahrenheit", "fahrenheit": "fahrenheit",
        "k": "kelvin", "kelvin": "kelvin",
    }

    from_temp = temp_aliases.get(from_u)
    to_temp = temp_aliases.get(to_u)

    if from_temp and to_temp:
        result = _convert_temperature(value, from_temp, to_temp)
        if result is not None:
            return f"{value} {from_unit} = {result:.2f} {to_unit}"

    # Distance conversions (convert to meters as intermediate)
    distance_to_meters: dict[str, float] = {
        "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
        "m": 1.0, "meter": 1.0, "meters": 1.0,
        "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
        "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
        "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
        "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
        "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
        "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
    }

    if from_u in distance_to_meters and to_u in distance_to_meters:
        meters = value * distance_to_meters[from_u]
        result = meters / distance_to_meters[to_u]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"

    # Weight conversions (convert to grams as intermediate)
    weight_to_grams: dict[str, float] = {
        "kg": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0,
        "g": 1.0, "gram": 1.0, "grams": 1.0,
        "mg": 0.001, "milligram": 0.001, "milligrams": 0.001,
        "lb": 453.592, "pound": 453.592, "pounds": 453.592,
        "oz": 28.3495, "ounce": 28.3495, "ounces": 28.3495,
    }

    if from_u in weight_to_grams and to_u in weight_to_grams:
        grams = value * weight_to_grams[from_u]
        result = grams / weight_to_grams[to_u]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"

    return (
        f"Cannot convert from '{from_unit}' to '{to_unit}'. "
        "Supported categories: temperature (C/F/K), "
        "distance (km/mi/m/ft/in/yd/cm/mm), "
        "weight (kg/lb/g/oz/mg)."
    )


def _convert_temperature(
    value: float, from_unit: str, to_unit: str
) -> float | None:
    """Convert between Celsius, Fahrenheit, and Kelvin.

    Uses the standard conversion formulas:
      - C to F: (C * 9/5) + 32
      - F to C: (F - 32) * 5/9
      - C to K: C + 273.15
      - K to C: K - 273.15

    Args:
        value: Temperature value.
        from_unit: Source unit ("celsius", "fahrenheit", "kelvin").
        to_unit: Target unit ("celsius", "fahrenheit", "kelvin").

    Returns:
        Converted value, or None if conversion is not supported.
    """
    if from_unit == to_unit:
        return value

    # First convert to Celsius as intermediate
    if from_unit == "celsius":
        celsius = value
    elif from_unit == "fahrenheit":
        celsius = (value - 32) * 5 / 9
    elif from_unit == "kelvin":
        celsius = value - 273.15
    else:
        return None

    # Then convert from Celsius to target
    if to_unit == "celsius":
        return celsius
    elif to_unit == "fahrenheit":
        return (celsius * 9 / 5) + 32
    elif to_unit == "kelvin":
        return celsius + 273.15
    else:
        return None


# ---------------------------------------------------------------------------
# Agent state and graph
# ---------------------------------------------------------------------------

class ToolAgentState(TypedDict):
    """State for the tool-calling agent.

    Simpler than ReAct -- we just track messages and a loop counter.
    The LLM decides when to stop calling tools implicitly (by not
    including tool_calls in its response).
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    loop_count: int


# All tools available to this agent
TOOL_AGENT_TOOLS: list[Any] = [
    web_search,
    calculator,
    web_scrape,
    get_current_time,
    text_length,
    unit_converter,
]

# Quick lookup: tool name -> tool object
_TOOL_MAP: dict[str, Any] = {t.name: t for t in TOOL_AGENT_TOOLS}


TOOL_AGENT_SYSTEM_PROMPT: str = """You are a helpful assistant with access to various tools.

When the user asks a question, decide which tool (if any) would help answer it.
If no tool is needed, just answer directly from your knowledge.

Available tools will be provided in your tool definitions. Use them wisely:
- Use web_search for current events, facts, or information you're unsure about
- Use calculator for any math computation (always use the tool, don't do math in your head)
- Use web_scrape to read the full content of a specific URL
- Use get_current_time for date/time questions
- Use text_length for text analysis
- Use unit_converter for unit conversions

Always provide clear, well-formatted final answers."""


def _call_model(state: ToolAgentState) -> dict[str, Any]:
    """Invoke the LLM with the current messages and bound tools.

    This node sends everything to the LLM and lets it decide:
      - Call a tool (returns AIMessage with tool_calls)
      - Answer directly (returns AIMessage with text content)

    The LLM sees the full conversation history, including previous
    tool results, so it can build on earlier information.

    Args:
        state: Current agent state.

    Returns:
        State update with the LLM's response.
    """
    settings = Settings()
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
    )

    # bind_tools() modifies the LLM to include tool schemas.
    # The LLM can then produce structured tool-call outputs.
    llm_with_tools = llm.bind_tools(TOOL_AGENT_TOOLS)

    messages = list(state["messages"])

    # Ensure system prompt is present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=TOOL_AGENT_SYSTEM_PROMPT))

    response: AIMessage = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "loop_count": state.get("loop_count", 0) + 1,
    }


def _call_tools(state: ToolAgentState) -> dict[str, Any]:
    """Execute all tool calls from the last AI message.

    When the LLM decides to call one or more tools, this node:
      1. Extracts tool calls from the AIMessage
      2. Executes each tool
      3. Returns ToolMessages with results

    Multiple tools can be called in a single turn (parallel tool calling).
    This is more efficient than calling one tool at a time.

    Args:
        state: Current agent state.

    Returns:
        State update with tool result messages.
    """
    messages = state["messages"]
    last_message = messages[-1]

    tool_results: list[ToolMessage] = []

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            call_id = tool_call["id"]

            logger.info("tool_agent_calling", tool=tool_name, args=tool_args)

            if tool_name in _TOOL_MAP:
                try:
                    result = _TOOL_MAP[tool_name].invoke(tool_args)
                except Exception as e:
                    result = f"Error calling {tool_name}: {e}"
                    logger.error("tool_call_error", tool=tool_name, error=str(e))
            else:
                result = f"Unknown tool: {tool_name}"

            tool_results.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call_id,
                    name=tool_name,
                )
            )

    return {"messages": tool_results}


def _should_continue(state: ToolAgentState) -> Literal["tools", "end"]:
    """Decide whether to execute tools or finish.

    Simple logic:
      - If the LLM wants to call tools AND we haven't hit the loop limit -> "tools"
      - Otherwise -> "end"

    Args:
        state: Current agent state.

    Returns:
        "tools" to execute tool calls, "end" to finish.
    """
    messages = state["messages"]
    last_message = messages[-1]
    loop_count = state.get("loop_count", 0)

    # Safety limit: max 5 tool-calling rounds
    if loop_count >= 5:
        logger.warning("tool_agent_loop_limit", loop_count=loop_count)
        return "end"

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return "end"


def build_tool_agent_graph() -> StateGraph:
    """Build the tool-calling agent graph.

    This is a simpler graph than ReAct:

        ┌─────────┐
        │  START   │
        └────┬─────┘
             │
             ▼
        ┌────────────┐   tool_calls   ┌────────────┐
        │ call_model  │──────────────→│ call_tools  │
        └─────┬──────┘                └──────┬──────┘
              │                               │
              │ no tool_calls                 │ (loop back)
              ▼                               │
         ┌─────────┐                          │
         │   END   │  ◄───────────────────────┘
         └─────────┘

    Returns:
        Compiled LangGraph StateGraph.
    """
    graph = StateGraph(ToolAgentState)

    graph.add_node("call_model", _call_model)
    graph.add_node("call_tools", _call_tools)

    graph.set_entry_point("call_model")

    graph.add_conditional_edges(
        "call_model",
        _should_continue,
        {"tools": "call_tools", "end": END},
    )

    # After executing tools, always go back to the model so it can
    # process the results and decide next steps.
    graph.add_edge("call_tools", "call_model")

    return graph.compile()


# ---------------------------------------------------------------------------
# ToolAgent class -- public API
# ---------------------------------------------------------------------------

@dataclass
class ToolAgent:
    """A simple tool-calling agent using LangChain + LangGraph.

    This agent is best for straightforward queries that need one or two
    tool calls. For complex multi-step research, use ReActAgent instead.

    USAGE:
        >>> from agentexplorr.agents.tool_agent import ToolAgent
        >>> agent = ToolAgent()
        >>> result = agent.run("What time is it?")
        >>> print(result.answer)

        >>> result = agent.run("Convert 72 fahrenheit to celsius")
        >>> print(result.answer)

        >>> result = agent.run("What is sqrt(144) + 10?")
        >>> print(result.answer)

    ADDING CUSTOM TOOLS:
        You can extend this agent with additional tools by passing them
        to the constructor:

        >>> from langchain_core.tools import tool
        >>> @tool
        ... def my_tool(x: str) -> str:
        ...     '''My custom tool description.'''
        ...     return f"Processed: {x}"
        >>> agent = ToolAgent(extra_tools=[my_tool])

    Attributes:
        model: Ollama model name.
        base_url: Ollama server URL.
        extra_tools: Additional tools to make available to the agent.
        verbose: If True, print tool call details.
    """

    model: str = ""
    base_url: str = ""
    extra_tools: list[Any] = field(default_factory=list)
    verbose: bool = False
    _graph: Any = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        """Initialize settings and build the agent graph."""
        settings = Settings()
        if not self.model:
            self.model = settings.ollama_model
        if not self.base_url:
            self.base_url = settings.ollama_base_url

        # If extra tools were provided, rebuild the graph with them included.
        # For simplicity, we use the default graph; extra tools would require
        # rebuilding the tool list and map. This is left as-is for the common
        # case and can be extended.
        if self.extra_tools:
            # Register extra tools in the module-level maps so the graph
            # nodes can find them.
            for t in self.extra_tools:
                TOOL_AGENT_TOOLS.append(t)
                _TOOL_MAP[t.name] = t

        self._graph = build_tool_agent_graph()

        logger.info(
            "tool_agent_initialized",
            model=self.model,
            num_tools=len(TOOL_AGENT_TOOLS),
        )

    def run(self, query: str) -> ToolAgentResult:
        """Run the tool-calling agent on a query.

        Args:
            query: The user's question or request.

        Returns:
            A ToolAgentResult with the answer and metadata.

        Example:
            >>> agent = ToolAgent()
            >>> result = agent.run("Search for LangGraph tutorials")
            >>> print(result.answer)
        """
        logger.info("tool_agent_run", query=query[:100])

        initial_state: ToolAgentState = {
            "messages": [HumanMessage(content=query)],
            "loop_count": 0,
        }

        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as e:
            logger.error("tool_agent_error", error=str(e))
            return ToolAgentResult(
                query=query,
                answer=f"Agent error: {e}",
                tools_called=[],
                success=False,
            )

        messages = final_state.get("messages", [])

        # Extract final answer from the last AI message
        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                answer = msg.content
                break

        # Collect tool usage info
        tools_called: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_called.append({
                        "name": tc["name"],
                        "args": tc["args"],
                    })

        if self.verbose:
            for tc in tools_called:
                print(f"  [TOOL] {tc['name']}({tc['args']})")

        return ToolAgentResult(
            query=query,
            answer=answer,
            tools_called=tools_called,
            success=True,
        )

    def list_tools(self) -> list[dict[str, str]]:
        """List all available tools with their descriptions.

        Useful for understanding what this agent can do and for
        displaying tool information in a UI.

        Returns:
            List of dicts with "name" and "description" keys.
        """
        return [
            {"name": t.name, "description": t.description}
            for t in TOOL_AGENT_TOOLS
        ]


@dataclass
class ToolAgentResult:
    """Result from a tool-calling agent run.

    Attributes:
        query: The original user query.
        answer: The agent's final answer.
        tools_called: List of tools that were called, with their arguments.
        success: Whether the agent completed without errors.
    """

    query: str
    answer: str
    tools_called: list[dict[str, Any]]
    success: bool = True

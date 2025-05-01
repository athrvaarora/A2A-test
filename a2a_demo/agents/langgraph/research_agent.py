import os
from typing import Any, Dict, AsyncIterable, Literal, List, Tuple, Union
from pydantic import BaseModel, Field
from uuid import uuid4

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Initialize memory for LangGraph sessions
memory = MemorySaver()


@tool
def search_web(query: str) -> str:
    """
    Search the web for information on a topic.
    
    Args:
        query: The search query
    
    Returns:
        Search results as text
    """
    # Mock implementation - in a real system, this would connect to a search API
    if "weather" in query.lower():
        return "Current weather: Mostly sunny with temperatures around 72°F. Light winds from the west at 5-10 mph."
    elif "news" in query.lower():
        return "Latest headlines: New AI breakthrough announced. Global markets show mixed results. International peace talks continue in Geneva."
    elif "technology" in query.lower() or "tech" in query.lower():
        return "Latest in tech: New quantum computing research shows promise. Major tech companies investing in AI. Smartphone sales increased by 5% globally this quarter."
    elif "science" in query.lower():
        return "Science updates: CERN reports new particle discovery. NASA Mars rover finds evidence of ancient water flows. New cancer treatment shows promising results in clinical trials."
    else:
        return f"Search results for '{query}': Found several relevant articles. Topics include general information, recent developments, and expert opinions on this subject."


class ResponseFormat(BaseModel):
    """Format for agent responses."""
    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class AgentState(BaseModel):
    """State for the research agent."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]] = Field(default_factory=list)
    next: str = "agent"


class ResearchAgent:
    """Agent that can perform research and provide information."""
    
    SYSTEM_INSTRUCTION = """
    You are a helpful research assistant. Your purpose is to find information and 
    provide comprehensive, accurate summaries on any topic. Use the search_web tool 
    to look up information, then synthesize it into a clear, well-organized response.
    
    Follow these guidelines:
    1. If a query is unclear, ask clarifying questions before searching
    2. When you have enough information, provide a complete response
    3. Always cite your sources at the end of your response
    4. Be objective and present different perspectives when appropriate
    5. If you cannot find information on a topic, be honest about it
    
    When you have a final answer, make sure to include "FINAL ANSWER:" followed by your response.
    """
    
    def __init__(self, api_key=None):
        """Initialize the research agent."""
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        self.tools = [search_web]
        
        # Create the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the agent graph."""
        # Define the state graph
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("agent", self._agent_node)
        graph.add_node("action", self._action_node)
        
        # Add conditional routing
        def router(state):
            return state.next
        
        # Add edges with conditional routing
        graph.add_conditional_edges("agent", router, {
            "action": "action",
            END: END
        })
        graph.add_edge("action", "agent")
        
        # Set the entry point
        graph.set_entry_point("agent")
        
        # Compile the graph
        return graph.compile()
    
    def _agent_node(self, state: AgentState) -> AgentState:
        """Process the agent's thinking."""
        # If this is the first run, add system message
        if not state.messages or not any(isinstance(msg, SystemMessage) for msg in state.messages):
            state.messages.append(SystemMessage(content=self.SYSTEM_INSTRUCTION))
            
        # Get response from the model
        response = self.model.invoke(state.messages)
        state.messages.append(response)
        
        # Check if we're done or need to use a tool
        if "FINAL ANSWER:" in response.content:
            state.next = END
        elif hasattr(response, "tool_calls") and response.tool_calls:
            state.next = "action"
        else:
            # If the model didn't use a tool or give a final answer, assume we need another agent turn
            state.next = "agent"
            
        return state
    
    def _action_node(self, state: AgentState) -> AgentState:
        """Process tool actions."""
        # Get the last message (should be from the model)
        last_message = state.messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                for tool in self.tools:
                    if tool.name == tool_call.name:
                        try:
                            args = tool_call.args if isinstance(tool_call.args, dict) else {}
                            tool_response = tool.invoke(args or tool_call.args)
                            tool_msg = ToolMessage(
                                content=str(tool_response),
                                tool_call_id=tool_call.id
                            )
                            state.messages.append(tool_msg)
                        except Exception as e:
                            tool_msg = ToolMessage(
                                content=f"Error: {str(e)}",
                                tool_call_id=tool_call.id
                            )
                            state.messages.append(tool_msg)
        
        state.next = "agent"
        return state
    
    def invoke(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Process a query and return a response."""
        if not session_id:
            session_id = uuid4().hex
            
        config = {"configurable": {"session_id": session_id}}
        inputs = AgentState(messages=[HumanMessage(content=query)])
        result = self.graph.invoke(inputs, config)
        
        return self.extract_response(result, config)
    
    async def stream(self, query: str, session_id: str = None) -> AsyncIterable[Dict[str, Any]]:
        """Process a query and stream the response."""
        if not session_id:
            session_id = uuid4().hex
            
        inputs = AgentState(messages=[HumanMessage(content=query)])
        config = {"configurable": {"session_id": session_id}}
        
        # Stream intermediate thinking steps
        async for chunk in self.graph.astream(inputs, config):
            if chunk.next == "action":
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Searching for information...",
                }
            elif chunk.next == "agent" and len(chunk.messages) > 1 and isinstance(chunk.messages[-1], ToolMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing search results...",
                }
        
        # Return final response
        result = self.graph.get_state(config)
        yield self.extract_response(result, config)
    
    def extract_response(self, result: Any, config: dict) -> Dict[str, Any]:
        """Extract the final response from the result."""
        # Get the final state
        if not hasattr(result, "messages"):
            # Try to get state from the checkpointer
            result = self.graph.get_state(config)
            
        # Extract the final message
        messages = result.messages if hasattr(result, "messages") else []
        
        if not messages:
            return {
                "is_task_complete": False,
                "require_user_input": True,
                "content": "I couldn't process your request. Could you try again?"
            }
        
        # Find the final answer in the AI messages
        final_answer = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and "FINAL ANSWER:" in msg.content:
                final_answer = msg.content.split("FINAL ANSWER:", 1)[1].strip()
                break
        
        if final_answer:
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": final_answer
            }
        
        # If no final answer found, return the last AI message
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": msg.content
                }
        
        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "I'm unable to complete this request at the moment. Could you try again or rephrase your question?"
        }
    
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"] 
import os
from typing import Any, Dict, AsyncIterable, Literal, List
from pydantic import BaseModel, Field
from uuid import uuid4

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
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
    
    Set response status to "input_required" if you need more information from the user.
    Set response status to "error" if there is an error processing the request.
    Set response status to "completed" once you have provided a complete answer.
    """
    
    def __init__(self, api_key=None):
        """Initialize the research agent."""
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        self.tools = [search_web]
        
        self.graph = create_react_agent(
            self.model, 
            tools=self.tools, 
            checkpointer=memory, 
            prompt=self.SYSTEM_INSTRUCTION, 
            response_format=ResponseFormat
        )
    
    def invoke(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Process a query and return a response."""
        if not session_id:
            session_id = uuid4().hex
            
        config = {"configurable": {"thread_id": session_id}}
        self.graph.invoke({"messages": [HumanMessage(content=query)]}, config)        
        return self.get_agent_response(config)
    
    async def stream(self, query: str, session_id: str = None) -> AsyncIterable[Dict[str, Any]]:
        """Process a query and stream the response."""
        if not session_id:
            session_id = uuid4().hex
            
        inputs = {"messages": [HumanMessage(content=query)]}
        config = {"configurable": {"thread_id": session_id}}
        
        # Stream intermediate thinking steps
        for item in self.graph.stream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]
            if isinstance(message, AIMessage) and message.tool_calls and len(message.tool_calls) > 0:
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Searching for information...",
                }
            elif isinstance(message, ToolMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing search results...",
                }
        
        # Return final response
        yield self.get_agent_response(config)
    
    def get_agent_response(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the agent's response from the graph state."""
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        
        if structured_response and isinstance(structured_response, ResponseFormat):
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message
                }
            elif structured_response.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message
                }
            elif structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": structured_response.message
                }
        
        # Fallback response if structured format is not available
        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "I'm unable to complete this request at the moment. Could you try again or rephrase your question?",
        }
    
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"] 
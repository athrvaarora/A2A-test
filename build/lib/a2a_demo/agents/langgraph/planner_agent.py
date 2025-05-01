import os
from typing import Any, Dict, AsyncIterable, Literal, List
from pydantic import BaseModel, Field
from uuid import uuid4

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Initialize memory for LangGraph sessions
memory = MemorySaver()


class Plan(BaseModel):
    """A structured plan with goals and steps."""
    title: str
    description: str
    goals: List[str]
    steps: List[Dict[str, Any]]
    estimated_time: str
    resources_needed: List[str] = Field(default_factory=list)


class ResponseFormat(BaseModel):
    """Format for agent responses."""
    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str
    plan: Plan = None


class PlannerAgent:
    """Agent that can create structured plans for various tasks and projects."""
    
    SYSTEM_INSTRUCTION = """
    You are a skilled planning assistant. Your purpose is to help users create 
    detailed, actionable plans for their projects, goals, or tasks. 
    
    When creating a plan:
    1. Analyze the objective carefully to understand what the user wants to achieve
    2. Break down the objective into clear, achievable steps
    3. Establish realistic timelines and resource requirements
    4. Consider potential challenges and provide contingency options
    5. Organize information in a logical, easy-to-follow structure
    
    If you need clarification or more details to create an effective plan, 
    ask follow-up questions.
    
    Set response status to "input_required" if you need more information from the user.
    Set response status to "error" if there is an error processing the request.
    Set response status to "completed" once you have provided a complete plan.
    
    Each plan you create should include:
    - A clear title
    - A brief description
    - The main goals
    - Detailed steps with descriptions
    - Estimated time to complete
    - Required resources
    """
    
    def __init__(self, api_key=None):
        """Initialize the planner agent."""
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        
        self.graph = create_react_agent(
            self.model,
            tools=[],  # No tools needed for planning
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=ResponseFormat
        )
    
    def invoke(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Process a query and return a response with a plan."""
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
            if isinstance(message, AIMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Developing your plan...",
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
                    "content": structured_response.message,
                    "data": structured_response.plan.model_dump() if structured_response.plan else None
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
                    "content": structured_response.message,
                    "data": structured_response.plan.model_dump() if structured_response.plan else None
                }
        
        # Fallback response if structured format is not available
        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "I'm unable to create a plan at the moment. Could you provide more details about what you'd like to plan?",
        }
    
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "data"] 
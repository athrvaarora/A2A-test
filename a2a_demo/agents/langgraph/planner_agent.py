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
def create_plan(goal: str) -> str:
    """
    Create a detailed plan to achieve a goal.
    
    Args:
        goal: The goal to plan for
    
    Returns:
        A structured plan with steps and milestones
    """
    # Mock implementation - in a real system, this would use more sophisticated planning
    return f"""Plan for achieving: {goal}

1. Initial Assessment (Day 1-2)
   - Analyze current situation
   - Identify key stakeholders
   - Define success metrics

2. Research Phase (Day 3-5)
   - Gather relevant information
   - Study best practices
   - Identify potential challenges

3. Strategy Development (Day 6-8)
   - Create detailed action plan
   - Allocate resources
   - Set milestones

4. Implementation (Day 9-20)
   - Execute plan step by step
   - Monitor progress
   - Make adjustments as needed

5. Review and Optimization (Day 21-30)
   - Evaluate results
   - Document learnings
   - Plan for continuous improvement

Key Milestones:
- Day 5: Complete research phase
- Day 8: Finalize strategy
- Day 15: Mid-point review
- Day 30: Final evaluation

Success Metrics:
- Achievement of primary goal
- Stakeholder satisfaction
- Resource efficiency
- Timeline adherence"""


class ResponseFormat(BaseModel):
    """Format for agent responses."""
    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class AgentState(BaseModel):
    """State for the planner agent."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]] = Field(default_factory=list)
    next: str = "agent"


class PlannerAgent:
    """Agent that can create and manage plans."""
    
    SYSTEM_INSTRUCTION = """
    You are a strategic planning assistant. Your purpose is to help users create 
    detailed, actionable plans to achieve their goals. Use the create_plan tool 
    to generate structured plans, then help refine and optimize them.
    
    Follow these guidelines:
    1. If a goal is unclear, ask clarifying questions before planning
    2. Break down complex goals into manageable steps
    3. Consider potential obstacles and include contingency plans
    4. Set realistic timelines and milestones
    5. Provide clear success metrics
    
    When you have a final plan, make sure to include "FINAL PLAN:" followed by your response.
    """
    
    def __init__(self, api_key=None):
        """Initialize the planner agent."""
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        self.tools = [create_plan]
        
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
        if "FINAL PLAN:" in response.content:
            state.next = END
        elif hasattr(response, "tool_calls") and response.tool_calls:
            state.next = "action"
        else:
            # If the model didn't use a tool or give a final plan, assume we need another agent turn
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
                    "content": "Creating plan...",
                }
            elif chunk.next == "agent" and len(chunk.messages) > 1 and isinstance(chunk.messages[-1], ToolMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing plan details...",
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
        
        # Find the final plan in the AI messages
        final_plan = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and "FINAL PLAN:" in msg.content:
                final_plan = msg.content.split("FINAL PLAN:", 1)[1].strip()
                break
        
        if final_plan:
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": final_plan
            }
        
        # If no final plan found, return the last AI message
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
            "content": "I'm unable to create a plan at the moment. Could you try again or provide more details about your goal?"
        }
    
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"] 
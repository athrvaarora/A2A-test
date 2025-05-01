import asyncio
import argparse
import logging
from a2a_demo.common import A2AServer, AgentCard, AgentCapabilities, AgentSkill, AgentProvider
from .task_manager import LangGraphTaskManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_agent_card(agent_type: str, port: int) -> AgentCard:
    """Create an agent card for the LangGraph agent."""
    if agent_type == "research":
        name = "Research Assistant"
        description = "A research assistant that can search for information and provide comprehensive answers."
        skills = [
            AgentSkill(
                id="search",
                name="Web Search",
                description="Search for information on any topic and provide a comprehensive summary.",
                tags=["research", "information", "search"],
                examples=[
                    "What are the latest developments in quantum computing?",
                    "Tell me about climate change initiatives.",
                    "Research the impact of AI on healthcare."
                ]
            )
        ]
    elif agent_type == "planner":
        name = "Task Planner"
        description = "A planning assistant that can create detailed, actionable plans for projects and tasks."
        skills = [
            AgentSkill(
                id="plan",
                name="Create Plan",
                description="Create detailed step-by-step plans for projects, goals, or tasks.",
                tags=["planning", "organization", "project-management"],
                examples=[
                    "Create a plan for launching a new product.",
                    "Help me organize a team retreat.",
                    "Plan a home renovation project."
                ]
            )
        ]
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    return AgentCard(
        name=name,
        description=description,
        url=f"http://localhost:{port}",
        provider=AgentProvider(
            organization="LangGraph Agents",
            url="https://github.com/langchain-ai/langgraph"
        ),
        version="1.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
            stateTransitionHistory=True
        ),
        defaultInputModes=["text"],
        defaultOutputModes=["text", "data"],
        skills=skills
    )


def run_server(agent_type: str, port: int):
    """Run the A2A server for a LangGraph agent."""
    logger.info(f"Starting {agent_type} LangGraph agent server on port {port}")
    
    # Create the task manager for this agent type
    task_manager = LangGraphTaskManager(agent_type)
    
    # Create the agent card
    agent_card = create_agent_card(agent_type, port)
    
    # Create and start the server
    server = A2AServer(
        host="0.0.0.0",
        port=port,
        endpoint="/",
        agent_card=agent_card,
        task_manager=task_manager
    )
    
    server.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a LangGraph agent A2A server")
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=["research", "planner"],
        default="research",
        help="The type of LangGraph agent to run"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on"
    )
    
    args = parser.parse_args()
    run_server(args.agent_type, args.port) 
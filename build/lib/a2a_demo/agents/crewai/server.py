import asyncio
import argparse
import logging
from a2a_demo.common import A2AServer, AgentCard, AgentCapabilities, AgentSkill, AgentProvider
from .task_manager import CrewAITaskManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_agent_card(agent_type: str, port: int) -> AgentCard:
    """Create an agent card for the CrewAI agent."""
    if agent_type == "creative":
        name = "Creative Content Generator"
        description = "A creative assistant that can generate stories, marketing content, and creative ideas."
        skills = [
            AgentSkill(
                id="generate",
                name="Content Generation",
                description="Generate creative content including stories, marketing copy, and ideas.",
                tags=["creative", "writing", "marketing", "brainstorming"],
                examples=[
                    "Write a short story about a time traveler.",
                    "Create marketing copy for a new fitness app.",
                    "Generate 5 unique business ideas for a tech startup.",
                    "Write a social media campaign for a coffee shop."
                ]
            )
        ]
    elif agent_type == "analysis":
        name = "Data Analyst"
        description = "An analytical assistant that can analyze problems, situations, and data to provide insights."
        skills = [
            AgentSkill(
                id="analyze",
                name="Situation Analysis",
                description="Analyze problems, situations, or scenarios to derive meaningful insights and recommendations.",
                tags=["analysis", "problem-solving", "insights", "recommendations"],
                examples=[
                    "Analyze the current trends in remote work and their implications.",
                    "What factors should be considered when expanding a business internationally?",
                    "Analyze the pros and cons of implementing a four-day workweek.",
                    "What are the key considerations for improving customer retention?"
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
            organization="CrewAI Agents",
            url="https://github.com/joaomdmoura/crewAI"
        ),
        version="1.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
            stateTransitionHistory=True
        ),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=skills
    )


def run_server(agent_type: str, port: int):
    """Run the A2A server for a CrewAI agent."""
    logger.info(f"Starting {agent_type} CrewAI agent server on port {port}")
    
    # Create the task manager for this agent type
    task_manager = CrewAITaskManager(agent_type)
    
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
    parser = argparse.ArgumentParser(description="Run a CrewAI agent A2A server")
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=["creative", "analysis"],
        default="creative",
        help="The type of CrewAI agent to run"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5100,
        help="Port to run the server on"
    )
    
    args = parser.parse_args()
    run_server(args.agent_type, args.port) 
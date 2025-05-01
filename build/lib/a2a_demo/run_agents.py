import os
import subprocess
import asyncio
import argparse
import logging
import time
import signal
import sys
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the ports for each agent
AGENT_PORTS = {
    "langgraph_research": 5001,
    "langgraph_planner": 5002,
    "crewai_creative": 5003,
    "crewai_analysis": 5004,
    "chat_ui": 8000
}

# Store running processes
processes: Dict[str, subprocess.Popen] = {}


def signal_handler(sig, frame):
    """Handle CTRL+C to gracefully shut down all processes."""
    logger.info("Shutting down all agents...")
    for name, process in processes.items():
        logger.info(f"Terminating {name}...")
        process.terminate()
    sys.exit(0)


def start_agent(agent_type: str, framework: str, port: int) -> subprocess.Popen:
    """Start an agent process."""
    # Determine the correct command to start the agent
    if framework == "langgraph":
        cmd = [
            sys.executable, "-m", 
            f"a2a_demo.agents.langgraph.server", 
            "--agent-type", agent_type,
            "--port", str(port)
        ]
    elif framework == "crewai":
        cmd = [
            sys.executable, "-m", 
            f"a2a_demo.agents.crewai.server", 
            "--agent-type", agent_type,
            "--port", str(port)
        ]
    else:
        raise ValueError(f"Unknown framework: {framework}")
    
    logger.info(f"Starting {framework} {agent_type} agent on port {port}...")
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )
    
    # Check if process started successfully
    time.sleep(1)
    if process.poll() is not None:
        stderr = process.stderr.read() if process.stderr else "No error output"
        raise RuntimeError(f"Failed to start {framework} {agent_type} agent: {stderr}")
    
    return process


def start_chat_ui(port: int) -> subprocess.Popen:
    """Start the chat UI."""
    cmd = [
        sys.executable, "-m",
        "a2a_demo.chat_ui.app",
        "--host", "localhost",
        "--port", str(port)
    ]
    
    logger.info(f"Starting chat UI on port {port}...")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Check if process started successfully
    time.sleep(1)
    if process.poll() is not None:
        stderr = process.stderr.read() if process.stderr else "No error output"
        raise RuntimeError(f"Failed to start chat UI: {stderr}")
    
    return process


def main():
    """Run all agents and the chat UI."""
    parser = argparse.ArgumentParser(description="Run all A2A agents and the chat UI")
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        choices=["all", "langgraph", "crewai", "ui"],
        default=["all"],
        help="Which agents to run (default: all)"
    )
    
    args = parser.parse_args()
    agents_to_run = args.agents
    
    # Register signal handler for CTRL+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start LangGraph agents
        if "all" in agents_to_run or "langgraph" in agents_to_run:
            processes["langgraph_research"] = start_agent(
                "research", 
                "langgraph", 
                AGENT_PORTS["langgraph_research"]
            )
            processes["langgraph_planner"] = start_agent(
                "planner", 
                "langgraph", 
                AGENT_PORTS["langgraph_planner"]
            )
        
        # Start CrewAI agents
        if "all" in agents_to_run or "crewai" in agents_to_run:
            processes["crewai_creative"] = start_agent(
                "creative", 
                "crewai", 
                AGENT_PORTS["crewai_creative"]
            )
            processes["crewai_analysis"] = start_agent(
                "analysis", 
                "crewai", 
                AGENT_PORTS["crewai_analysis"]
            )
        
        # Start chat UI
        if "all" in agents_to_run or "ui" in agents_to_run:
            processes["chat_ui"] = start_chat_ui(AGENT_PORTS["chat_ui"])
        
        # Print success message with URLs
        logger.info("All agents started successfully!")
        logger.info("\nAgent URLs:")
        if "all" in agents_to_run or "langgraph" in agents_to_run:
            logger.info(f"  Research Agent: http://localhost:{AGENT_PORTS['langgraph_research']}")
            logger.info(f"  Planning Agent: http://localhost:{AGENT_PORTS['langgraph_planner']}")
        if "all" in agents_to_run or "crewai" in agents_to_run:
            logger.info(f"  Creative Agent: http://localhost:{AGENT_PORTS['crewai_creative']}")
            logger.info(f"  Analysis Agent: http://localhost:{AGENT_PORTS['crewai_analysis']}")
        if "all" in agents_to_run or "ui" in agents_to_run:
            logger.info(f"\nChat UI: http://localhost:{AGENT_PORTS['chat_ui']}")
        
        logger.info("\nUse these URLs to add agents to the chat UI.")
        logger.info("Press CTRL+C to stop all agents\n")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
            # Check if any process has terminated
            for name, process in list(processes.items()):
                if process.poll() is not None:
                    stderr = process.stderr.read() if process.stderr else "No error output"
                    logger.error(f"{name} terminated unexpectedly: {stderr}")
                    del processes[name]
            
            # Exit if all processes have terminated
            if not processes:
                logger.error("All processes have terminated. Exiting.")
                break
                
    except Exception as e:
        logger.error(f"Error starting agents: {e}")
        for name, process in processes.items():
            logger.info(f"Terminating {name}...")
            process.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main() 
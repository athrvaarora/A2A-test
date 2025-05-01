# A2A Agent Communication Demo

This project demonstrates Google's Agent2Agent (A2A) protocol by implementing multiple AI agents that can interact with each other through a standardized communication interface.

## Overview

The demo consists of:

1. **Four Specialized Agents**:
   - **Research Agent** (LangGraph): Searches for information and provides answers
   - **Planning Agent** (LangGraph): Creates detailed plans for tasks and projects
   - **Creative Agent** (CrewAI): Generates creative content using a team of sub-agents
   - **Analysis Agent** (CrewAI): Analyzes problems and provides insights

2. **Chat Interface**: A web-based UI that lets you:
   - Add any of the agents (or other A2A-compatible agents) by URL
   - Send messages to specific agents
   - Receive streaming responses
   - View conversation history

## Requirements

- Python 3.10 or later
- The required Python packages (see pyproject.toml)
- Google API key for Gemini models

## Installation

1. Clone the repository
2. Install the package:
   ```
   pip install -e .
   ```
3. Set up your Google API key as an environment variable:
   ```
   export GOOGLE_API_KEY=your_api_key_here
   ```
   or create a `.env` file in the root directory with `GOOGLE_API_KEY=your_api_key_here`

## Running the Demo

The easiest way to run the demo is with the provided script:

```
python -m a2a_demo.run_agents
```

This will start all four agents and the chat UI. You can then access the chat interface at http://localhost:8000.

You can also run specific components:

```
# Run only LangGraph agents
python -m a2a_demo.run_agents --agents langgraph

# Run only CrewAI agents
python -m a2a_demo.run_agents --agents crewai

# Run only the chat UI
python -m a2a_demo.run_agents --agents ui
```

## Using the Chat Interface

1. Go to http://localhost:8000 in your browser
2. Add each agent using the sidebar form:
   - Research Agent: http://localhost:5001
   - Planning Agent: http://localhost:5002
   - Creative Agent: http://localhost:5003
   - Analysis Agent: http://localhost:5004
3. Select an agent from the sidebar
4. Type your message and press Send
5. View the agent's response in the chat

## How Agent Communication Works

The agents communicate using the A2A protocol, which follows these steps:

1. **Discovery**: Agents expose their capabilities via an Agent Card at `/.well-known/agent.json`
2. **Task Sending**: Messages are sent as Tasks with a unique ID and session context
3. **Status Updates**: Agents provide status updates (working, need more input, completed)
4. **Streaming**: Long-running processes can stream updates via Server-Sent Events (SSE)
5. **Content Exchange**: Agents exchange text, structured data, or files using a standardized format

## Project Structure

- **common/**: Core A2A protocol implementation (types, client, server)
- **agents/**: Agent implementations
  - **langgraph/**: LangGraph-based agents (research, planning)
  - **crewai/**: CrewAI-based agents (creative, analysis)
- **chat_ui/**: Web-based chat interface
- **run_agents.py**: Script to start all components

## Extending the Demo

You can extend this demo by:

1. Creating new agent types using different frameworks
2. Enhancing the existing agents with more capabilities
3. Implementing push notifications for asynchronous updates
4. Adding support for file exchange between agents
5. Creating an orchestration layer that routes tasks between specialized agents

## A2A Protocol Documentation

For more information on the A2A protocol, see the official repository: https://github.com/google/agent2agent 
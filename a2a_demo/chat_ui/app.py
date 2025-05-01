import os
import asyncio
import logging
import json
import argparse
import datetime
from typing import Dict, List, Any, Optional
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from .client import A2AAgentClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="A2A Agent Chat")

# Store for connected agents
agents: Dict[str, A2AAgentClient] = {}

# Store for active chat sessions
chat_sessions: Dict[str, Dict[str, Any]] = {}


class MessageRequest(BaseModel):
    """Request body for sending a message."""
    agent_id: str
    message: str
    session_id: Optional[str] = None
    

class AgentRequest(BaseModel):
    """Request body for adding an agent."""
    name: str
    url: str


class ChatMessage(BaseModel):
    """A message in a chat session."""
    id: str
    role: str  # "user" or "agent"
    content: str
    agent_id: Optional[str] = None
    timestamp: str


# HTML UI for chat interface (simplified)
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>A2A Agent Chat</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            height: 100vh;
        }
        
        #sidebar {
            width: 250px;
            background-color: #f5f5f5;
            padding: 20px;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
            overflow-y: auto;
        }
        
        #main {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 20px;
        }
        
        #chat-container {
            flex: 1;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        #input-container {
            display: flex;
            gap: 10px;
        }
        
        #message-input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        
        button {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        
        button:hover {
            background-color: #45a049;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
        }
        
        .user {
            background-color: #e6f7ff;
            align-self: flex-end;
        }
        
        .agent {
            background-color: #f5f5f5;
            align-self: flex-start;
        }
        
        .agent-item {
            display: flex;
            align-items: center;
            padding: 10px;
            margin-bottom: 10px;
            background-color: #fff;
            border-radius: 5px;
            cursor: pointer;
        }
        
        .agent-item.selected {
            background-color: #e6f7ff;
            border-left: 4px solid #1890ff;
        }
        
        .add-agent-form {
            margin-top: 20px;
            border-top: 1px solid #ddd;
            padding-top: 20px;
        }
        
        .add-agent-form input {
            width: 100%;
            padding: 8px;
            margin-bottom: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div id="sidebar">
        <h2>Agents</h2>
        <div id="agents-list"></div>
        
        <div class="add-agent-form">
            <h3>Add Agent</h3>
            <input id="agent-name" placeholder="Agent Name" />
            <input id="agent-url" placeholder="Agent URL" />
            <button id="add-agent-btn">Add Agent</button>
        </div>
    </div>
    
    <div id="main">
        <h1>A2A Agent Chat</h1>
        <div id="chat-container"></div>
        <div id="input-container">
            <input id="message-input" placeholder="Type your message..." />
            <button id="send-btn">Send</button>
        </div>
    </div>

    <script>
        let selectedAgentId = null;
        let chatWebSocket = null;
        
        // Connect to the WebSocket
        function connectWebSocket() {
            const ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'message') {
                    appendMessage(data.message);
                } else if (data.type === 'agent_list') {
                    updateAgentsList(data.agents);
                } else if (data.type === 'chat_history') {
                    loadChatHistory(data.messages);
                }
            };
            
            ws.onclose = function() {
                // Attempt to reconnect
                setTimeout(connectWebSocket, 1000);
            };
            
            chatWebSocket = ws;
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            connectWebSocket();
            
            // Load agents
            fetch('/agents')
                .then(response => response.json())
                .then(data => {
                    updateAgentsList(data);
                });
            
            // Send message button
            document.getElementById('send-btn').addEventListener('click', sendMessage);
            
            // Add agent button
            document.getElementById('add-agent-btn').addEventListener('click', addAgent);
            
            // Enter key to send
            document.getElementById('message-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        });
        
        function selectAgent(agentId) {
            selectedAgentId = agentId;
            
            // Update UI
            document.querySelectorAll('.agent-item').forEach(el => {
                el.classList.remove('selected');
            });
            
            const selectedEl = document.querySelector(`.agent-item[data-id="${agentId}"]`);
            if (selectedEl) {
                selectedEl.classList.add('selected');
            }
            
            // Fetch chat history
            fetch(`/chats/${agentId}`)
                .then(response => response.json())
                .then(data => {
                    loadChatHistory(data.messages);
                });
        }
        
        function updateAgentsList(agents) {
            const agentsList = document.getElementById('agents-list');
            agentsList.innerHTML = '';
            
            agents.forEach(agent => {
                const agentEl = document.createElement('div');
                agentEl.className = 'agent-item';
                agentEl.setAttribute('data-id', agent.id);
                agentEl.innerHTML = `
                    <div>
                        <strong>${agent.name}</strong><br>
                        <small>${agent.description || 'No description'}</small>
                    </div>
                `;
                agentEl.addEventListener('click', () => selectAgent(agent.id));
                agentsList.appendChild(agentEl);
                
                // Select first agent by default
                if (!selectedAgentId && agents.length > 0) {
                    selectAgent(agents[0].id);
                }
            });
        }
        
        function loadChatHistory(messages) {
            const chatContainer = document.getElementById('chat-container');
            chatContainer.innerHTML = '';
            
            messages.forEach(message => {
                appendMessage(message, false);
            });
            
            // Scroll to bottom
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function appendMessage(message, shouldScroll = true) {
            const chatContainer = document.getElementById('chat-container');
            const messageEl = document.createElement('div');
            messageEl.className = `message ${message.role}`;
            messageEl.innerHTML = `
                <strong>${message.role === 'user' ? 'You' : 'Agent'}</strong>
                <p>${message.content}</p>
            `;
            chatContainer.appendChild(messageEl);
            
            if (shouldScroll) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
        
        function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value.trim();
            
            if (!message) return;
            if (!selectedAgentId) {
                alert('Please select an agent first');
                return;
            }
            
            // Clear input
            input.value = '';
            
            // Send message via WebSocket
            chatWebSocket.send(JSON.stringify({
                type: 'send_message',
                agent_id: selectedAgentId,
                message: message
            }));
            
            // Optimistically add message to UI
            appendMessage({
                role: 'user',
                content: message
            });
        }
        
        function addAgent() {
            const nameInput = document.getElementById('agent-name');
            const urlInput = document.getElementById('agent-url');
            
            const name = nameInput.value.trim();
            const url = urlInput.value.trim();
            
            if (!name || !url) {
                alert('Please provide both name and URL');
                return;
            }
            
            fetch('/agents', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ name, url })
            })
            .then(response => response.json())
            .then(data => {
                nameInput.value = '';
                urlInput.value = '';
                
                // Update agents list
                fetch('/agents')
                    .then(response => response.json())
                    .then(data => {
                        updateAgentsList(data);
                    });
            })
            .catch(error => {
                alert('Error adding agent: ' + error.message);
            });
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def get_html():
    """Return the HTML for the chat UI."""
    return HTMLResponse(HTML)


@app.get("/agents")
async def get_agents():
    """Get the list of available agents."""
    result = []
    for agent_id, client in agents.items():
        try:
            agent_card = await client.get_agent_info()
            result.append({
                "id": agent_id,
                "name": agent_card.name,
                "description": agent_card.description,
                "url": agent_card.url
            })
        except Exception as e:
            logger.error(f"Error fetching agent info for {agent_id}: {e}")
    
    return result


@app.post("/agents")
async def add_agent(request: AgentRequest):
    """Add a new agent."""
    agent_id = str(uuid.uuid4())
    try:
        client = A2AAgentClient(request.url)
        agent_card = await client.get_agent_info()
        
        agents[agent_id] = client
        
        # Create a new chat session for this agent
        chat_sessions[agent_id] = {
            "session_id": None,  # Will be set when first message is sent
            "messages": []
        }
        
        return {
            "id": agent_id,
            "name": agent_card.name,
            "url": agent_card.url
        }
    except Exception as e:
        logger.error(f"Error adding agent {request.url}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to connect to agent: {str(e)}")


@app.get("/chats/{agent_id}")
async def get_chat(agent_id: str):
    """Get the chat history for an agent."""
    if agent_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    return {"messages": chat_sessions[agent_id]["messages"]}


@app.post("/send")
async def send_message(request: MessageRequest):
    """Send a message to an agent."""
    if request.agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    client = agents[request.agent_id]
    
    # Get or create session ID
    session_id = request.session_id
    if session_id is None:
        if chat_sessions[request.agent_id]["session_id"] is None:
            # First message for this agent
            session_id = str(uuid.uuid4())
            chat_sessions[request.agent_id]["session_id"] = session_id
        else:
            # Use existing session
            session_id = chat_sessions[request.agent_id]["session_id"]
    
    # Send message
    try:
        result = await client.send_message(request.message, session_id)
        
        # Record message history
        chat_sessions[request.agent_id]["messages"].append({
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": request.message,
            "agent_id": request.agent_id,
            "timestamp": str(datetime.datetime.now())
        })
        
        if result["success"] and "response" in result:
            chat_sessions[request.agent_id]["messages"].append({
                "id": str(uuid.uuid4()),
                "role": "agent",
                "content": result["response"],
                "agent_id": request.agent_id,
                "timestamp": str(datetime.datetime.now())
            })
        
        return result
    except Exception as e:
        logger.error(f"Error sending message to agent {request.agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error sending message: {str(e)}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for real-time chat."""
    await websocket.accept()
    
    # Send initial agent list
    agent_list = await get_agents()
    await websocket.send_json({"type": "agent_list", "agents": agent_list})
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "send_message":
                agent_id = message_data.get("agent_id")
                message_text = message_data.get("message")
                
                if not agent_id or not message_text or agent_id not in agents:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid message or agent not found"
                    })
                    continue
                
                # Get the client for this agent
                client = agents[agent_id]
                
                # Get or create session ID
                session_id = None
                if chat_sessions[agent_id]["session_id"] is None:
                    # First message for this agent
                    session_id = str(uuid.uuid4())
                    chat_sessions[agent_id]["session_id"] = session_id
                else:
                    # Use existing session
                    session_id = chat_sessions[agent_id]["session_id"]
                
                # Add user message to chat history
                user_message = {
                    "id": str(uuid.uuid4()),
                    "role": "user",
                    "content": message_text,
                    "agent_id": agent_id,
                    "timestamp": str(datetime.datetime.now())
                }
                chat_sessions[agent_id]["messages"].append(user_message)
                
                # Process with streaming if agent supports it
                try:
                    async for event in client.get_stream(message_text, session_id):
                        if event.get("type") == "error":
                            await websocket.send_json({
                                "type": "error",
                                "message": event.get("error", "Unknown error")
                            })
                            continue
                        
                        if event.get("type") == "artifact" and "content" in event:
                            # Final response received
                            agent_message = {
                                "id": str(uuid.uuid4()),
                                "role": "agent",
                                "content": event.get("content", ""),
                                "agent_id": agent_id,
                                "timestamp": str(datetime.datetime.now())
                            }
                            
                            chat_sessions[agent_id]["messages"].append(agent_message)
                            
                            # Send to client
                            await websocket.send_json({
                                "type": "message",
                                "message": agent_message
                            })
                            
                            if event.get("final"):
                                break
                        
                        # For interim updates (status type), we can send them too
                        if event.get("type") == "status" and event.get("content"):
                            await websocket.send_json({
                                "type": "status_update",
                                "content": event.get("content")
                            })
                            
                except Exception as e:
                    logger.error(f"Error streaming response from agent {agent_id}: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error from agent: {str(e)}"
                    })
                    
                    # Fallback to non-streaming if needed
                    try:
                        result = await client.send_message(message_text, session_id)
                        
                        if result["success"] and "response" in result:
                            agent_message = {
                                "id": str(uuid.uuid4()),
                                "role": "agent",
                                "content": result["response"],
                                "agent_id": agent_id,
                                "timestamp": str(datetime.datetime.now())
                            }
                            
                            chat_sessions[agent_id]["messages"].append(agent_message)
                            
                            await websocket.send_json({
                                "type": "message",
                                "message": agent_message
                            })
                    except Exception as e2:
                        logger.error(f"Fallback also failed for agent {agent_id}: {e2}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Communication with agent failed: {str(e2)}"
                        })
            
            elif message_data.get("type") == "get_agents":
                # Send updated agent list
                agent_list = await get_agents()
                await websocket.send_json({"type": "agent_list", "agents": agent_list})
                
            elif message_data.get("type") == "get_chat_history":
                agent_id = message_data.get("agent_id")
                if agent_id in chat_sessions:
                    await websocket.send_json({
                        "type": "chat_history",
                        "messages": chat_sessions[agent_id]["messages"]
                    })
                    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


def run_app(host="localhost", port=8000):
    """Run the FastAPI app."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the A2A agent chat UI")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on"
    )
    
    args = parser.parse_args()
    run_app(args.host, args.port) 
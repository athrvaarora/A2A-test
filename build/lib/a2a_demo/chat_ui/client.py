import asyncio
import json
import uuid
from typing import Dict, List, Any
import httpx
from a2a_demo.common import (
    A2AClient, 
    AgentCard,
    Message, 
    TextPart, 
    TaskSendParams
)

class A2AAgentClient:
    """Client for interacting with A2A agents."""
    
    def __init__(self, agent_url: str):
        """Initialize the client with an agent URL."""
        self.client = A2AClient(url=agent_url)
        self.agent_card = None
        
    async def get_agent_info(self) -> AgentCard:
        """Fetch the agent card from the agent."""
        if not self.agent_card:
            self.agent_card = await self.client.get_agent_card()
        return self.agent_card
    
    async def send_message(self, message_text: str, session_id: str = None) -> Dict[str, Any]:
        """Send a message to the agent and get the response."""
        if not session_id:
            session_id = uuid.uuid4().hex
            
        # Create the message
        message = Message(
            role="user",
            parts=[TextPart(text=message_text)]
        )
        
        # Create the task params
        task_id = uuid.uuid4().hex
        params = TaskSendParams(
            id=task_id,
            sessionId=session_id,
            message=message
        )
        
        # Send the message
        response = await self.client.send_task(params.model_dump())
        
        if response.error:
            return {
                "success": False,
                "error": f"Error {response.error.code}: {response.error.message}",
                "task_id": task_id,
                "session_id": session_id
            }
            
        task = response.result
        
        # Process the response
        result = {
            "success": True,
            "task_id": task_id,
            "session_id": session_id,
            "state": task.status.state,
            "response": ""
        }
        
        # Extract the response content
        if task.artifacts and len(task.artifacts) > 0:
            artifact = task.artifacts[0]
            for part in artifact.parts:
                if part.type == "text":
                    result["response"] = part.text
                elif part.type == "data":
                    result["data"] = part.data
        elif task.status.message:
            for part in task.status.message.parts:
                if part.type == "text":
                    result["response"] = part.text
                elif part.type == "data":
                    result["data"] = part.data
        
        return result
    
    async def get_stream(self, message_text: str, session_id: str = None):
        """Send a message to the agent and get a streaming response."""
        if not session_id:
            session_id = uuid.uuid4().hex
            
        # Create the message
        message = Message(
            role="user",
            parts=[TextPart(text=message_text)]
        )
        
        # Create the task params
        task_id = uuid.uuid4().hex
        params = TaskSendParams(
            id=task_id,
            sessionId=session_id,
            message=message
        )
        
        # Send the task with streaming
        client = httpx.Client(timeout=None)
        try:
            from httpx_sse import connect_sse
            
            with connect_sse(
                client, "POST", self.client.url, 
                json={
                    "jsonrpc": "2.0",
                    "id": uuid.uuid4().hex,
                    "method": "tasks/sendSubscribe",
                    "params": params.model_dump()
                }
            ) as event_source:
                for sse in event_source.iter_sse():
                    try:
                        data = json.loads(sse.data)
                        result = data.get("result", {})
                        
                        # Event could be a status update or an artifact
                        if "status" in result:
                            status = result["status"]
                            state = status.get("state")
                            message = status.get("message")
                            
                            response_text = ""
                            if message and "parts" in message:
                                for part in message["parts"]:
                                    if part.get("type") == "text":
                                        response_text = part.get("text", "")
                            
                            yield {
                                "type": "status",
                                "state": state,
                                "content": response_text,
                                "task_id": task_id,
                                "session_id": session_id,
                                "final": result.get("final", False)
                            }
                            
                        elif "artifact" in result:
                            artifact = result["artifact"]
                            content = ""
                            data = None
                            
                            if "parts" in artifact:
                                for part in artifact["parts"]:
                                    if part.get("type") == "text":
                                        content = part.get("text", "")
                                    elif part.get("type") == "data":
                                        data = part.get("data")
                            
                            yield {
                                "type": "artifact",
                                "content": content,
                                "data": data,
                                "task_id": task_id,
                                "session_id": session_id,
                                "final": result.get("final", False)
                            }
                            
                    except json.JSONDecodeError:
                        yield {
                            "type": "error", 
                            "error": "Failed to parse event data",
                            "task_id": task_id,
                            "session_id": session_id
                        }
                        
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e),
                "task_id": task_id,
                "session_id": session_id
            }
        finally:
            client.close() 
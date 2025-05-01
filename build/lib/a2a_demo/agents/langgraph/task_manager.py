import asyncio
import logging
import json
from typing import AsyncIterable, Dict, List, Union, Any
from uuid import uuid4

from a2a_demo.common import (
    InMemoryTaskManager, 
    Message, 
    TextPart, 
    TaskStatus, 
    Artifact, 
    Task, 
    TaskState,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    JSONRPCResponse,
    TaskSendParams,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    InternalError
)

from .research_agent import ResearchAgent
from .planner_agent import PlannerAgent

logger = logging.getLogger(__name__)


class LangGraphTaskManager(InMemoryTaskManager):
    """Task manager for LangGraph agents."""
    
    def __init__(self, agent_type: str = "research"):
        """Initialize the task manager with a specific agent type."""
        super().__init__()
        if agent_type == "research":
            self.agent = ResearchAgent()
        elif agent_type == "planner":
            self.agent = PlannerAgent()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        self.agent_type = agent_type
        
    def _get_user_query(self, task_send_params: TaskSendParams) -> str:
        """Extract the user query from the message."""
        for part in task_send_params.message.parts:
            if part.type == "text":
                return part.text
                
        raise ValueError("No text part found in message")
        
    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """Handle a request to send a task."""
        task_send_params = request.params
        query = self._get_user_query(task_send_params)
        
        try:
            # Create or update the task
            await self.upsert_task(task_send_params)
            
            # Set status to working
            task = await self.update_store(
                task_send_params.id, 
                TaskStatus(state=TaskState.WORKING),
                None
            )
            
            # Process with the agent
            agent_response = self.agent.invoke(query, task_send_params.sessionId)
            
            # Update task with agent's response
            parts = [TextPart(text=agent_response["content"])]
            
            if agent_response["require_user_input"]:
                task_status = TaskStatus(
                    state=TaskState.INPUT_REQUIRED,
                    message=Message(role="agent", parts=parts)
                )
                artifact = None
            else:
                task_status = TaskStatus(state=TaskState.COMPLETED)
                artifact = Artifact(parts=parts)
                
            task = await self.update_store(
                task_send_params.id, 
                task_status,
                None if artifact is None else [artifact]
            )
            
            # Return the task with limited history
            task_result = self.append_task_history(task, task_send_params.historyLength)
            return SendTaskResponse(id=request.id, result=task_result)
            
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            return SendTaskResponse(
                id=request.id,
                error=InternalError(message=f"Error processing task: {e}")
            )
    
    async def _run_streaming_agent(self, request: SendTaskStreamingRequest):
        """Run the agent with streaming and update subscribers."""
        task_send_params = request.params
        query = self._get_user_query(task_send_params)
        
        try:
            async for item in self.agent.stream(query, task_send_params.sessionId):
                is_task_complete = item["is_task_complete"]
                require_user_input = item["require_user_input"]
                content = item["content"]
                data = item.get("data")
                
                parts = [TextPart(text=content)]
                if data:
                    from a2a_demo.common import DataPart
                    parts.append(DataPart(data=data))
                
                artifact = None
                message = None
                end_stream = False
                
                if not is_task_complete and not require_user_input:
                    # Still working
                    task_state = TaskState.WORKING
                    message = Message(role="agent", parts=parts)
                elif require_user_input:
                    # Need more input
                    task_state = TaskState.INPUT_REQUIRED
                    message = Message(role="agent", parts=parts)
                    end_stream = True
                else:
                    # Complete
                    task_state = TaskState.COMPLETED
                    artifact = Artifact(parts=parts, index=0, append=False)
                    end_stream = True
                
                # Update task status
                task_status = TaskStatus(state=task_state, message=message)
                task = await self.update_store(
                    task_send_params.id,
                    task_status,
                    None if artifact is None else [artifact],
                )
                
                # Send updates to subscribers
                if artifact:
                    artifact_event = TaskArtifactUpdateEvent(
                        id=task_send_params.id, 
                        artifact=artifact,
                        final=end_stream
                    )
                    await self.enqueue_events_for_sse(task_send_params.id, artifact_event)
                
                status_event = TaskStatusUpdateEvent(
                    id=task_send_params.id, 
                    status=task_status, 
                    final=end_stream
                )
                await self.enqueue_events_for_sse(task_send_params.id, status_event)
                
        except Exception as e:
            logger.error(f"Error in streaming agent: {e}")
            error = InternalError(message=f"Error in streaming agent: {e}")
            await self.enqueue_events_for_sse(task_send_params.id, error)
    
    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> Union[AsyncIterable[SendTaskStreamingResponse], JSONRPCResponse]:
        """Handle a request to send a task with streaming."""
        try:
            task_send_params = request.params
            
            # Create or update the task
            await self.upsert_task(task_send_params)
            
            # Set up SSE consumer
            sse_event_queue = await self.setup_sse_consumer(task_send_params.id, False)
            
            # Run the agent in the background
            asyncio.create_task(self._run_streaming_agent(request))
            
            # Return SSE stream
            return self.dequeue_events_for_sse(
                request.id, task_send_params.id, sse_event_queue
            )
        except Exception as e:
            logger.error(f"Error setting up streaming: {e}")
            return JSONRPCResponse(
                id=request.id,
                error=InternalError(message=f"Error setting up streaming: {e}")
            ) 
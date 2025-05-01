import asyncio
import logging
from typing import AsyncIterable, Dict, List, Union
from uuid import uuid4

from .server import TaskManager
from .types import (
    Artifact,
    CancelTaskRequest,
    CancelTaskResponse,
    GetTaskPushNotificationRequest,
    GetTaskPushNotificationResponse,
    GetTaskRequest,
    GetTaskResponse,
    JSONRPCResponse,
    Message,
    PushNotificationConfig,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    SetTaskPushNotificationRequest,
    SetTaskPushNotificationResponse,
    Task,
    TaskIdParams,
    TaskNotCancelableError,
    TaskNotFoundError,
    TaskPushNotificationConfig,
    TaskQueryParams,
    TaskResubscriptionRequest,
    TaskSendParams,
    TaskStatus,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    UnsupportedOperationError,
    InternalError,
)

logger = logging.getLogger(__name__)


class InMemoryTaskManager(TaskManager):
    """Base task manager that stores tasks in memory."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.push_notification_infos: Dict[str, PushNotificationConfig] = {}
        self.lock = asyncio.Lock()
        self.task_sse_subscribers: Dict[str, List[asyncio.Queue]] = {}
        self.subscriber_lock = asyncio.Lock()

    async def on_get_task(self, request: GetTaskRequest) -> GetTaskResponse:
        """Handle a request to get a task's status."""
        task_query_params: TaskQueryParams = request.params

        async with self.lock:
            task = self.tasks.get(task_query_params.id)
            if task is None:
                return GetTaskResponse(id=request.id, error=TaskNotFoundError())

            task_result = self.append_task_history(
                task, task_query_params.historyLength
            )

        return GetTaskResponse(id=request.id, result=task_result)

    async def on_cancel_task(self, request: CancelTaskRequest) -> CancelTaskResponse:
        """Handle a request to cancel a task."""
        task_id_params: TaskIdParams = request.params

        async with self.lock:
            task = self.tasks.get(task_id_params.id)
            if task is None:
                return CancelTaskResponse(id=request.id, error=TaskNotFoundError())

        # Default implementation doesn't support cancellation
        return CancelTaskResponse(id=request.id, error=TaskNotCancelableError())
    
    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """Handle a request to send a task. Must be implemented by subclasses."""
        return SendTaskResponse(
            id=request.id, 
            error=UnsupportedOperationError(message="This method must be implemented by subclasses")
        )
    
    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> Union[AsyncIterable[SendTaskStreamingResponse], JSONRPCResponse]:
        """Handle a request to send a task with streaming. Must be implemented by subclasses."""
        return JSONRPCResponse(
            id=request.id,
            error=UnsupportedOperationError(message="This method must be implemented by subclasses")
        )

    async def on_set_task_push_notification(
        self, request: SetTaskPushNotificationRequest
    ) -> SetTaskPushNotificationResponse:
        """Handle a request to set push notification settings."""
        task_notification_params: TaskPushNotificationConfig = request.params

        try:
            await self.set_push_notification_info(
                task_notification_params.id, 
                task_notification_params.pushNotificationConfig
            )
        except Exception as e:
            logger.error(f"Error while setting push notification info: {e}")
            return JSONRPCResponse(
                id=request.id,
                error=InternalError(
                    message="An error occurred while setting push notification info"
                ),
            )
            
        return SetTaskPushNotificationResponse(id=request.id, result=task_notification_params)

    async def on_get_task_push_notification(
        self, request: GetTaskPushNotificationRequest
    ) -> GetTaskPushNotificationResponse:
        """Handle a request to get push notification settings."""
        task_params: TaskIdParams = request.params

        try:
            notification_info = await self.get_push_notification_info(task_params.id)
        except Exception as e:
            logger.error(f"Error while getting push notification info: {e}")
            return GetTaskPushNotificationResponse(
                id=request.id,
                error=InternalError(
                    message="An error occurred while getting push notification info"
                ),
            )
        
        return GetTaskPushNotificationResponse(
            id=request.id, 
            result=TaskPushNotificationConfig(
                id=task_params.id, 
                pushNotificationConfig=notification_info
            )
        )

    async def on_resubscribe_to_task(
        self, request: TaskResubscriptionRequest
    ) -> Union[AsyncIterable[JSONRPCResponse], JSONRPCResponse]:
        """Handle a request to resubscribe to task updates."""
        return JSONRPCResponse(
            id=request.id,
            error=UnsupportedOperationError(message="Resubscription is not supported")
        )

    async def set_push_notification_info(self, task_id: str, notification_config: PushNotificationConfig):
        """Set push notification information for a task."""
        async with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                raise ValueError(f"Task not found for {task_id}")

            self.push_notification_infos[task_id] = notification_config

    async def get_push_notification_info(self, task_id: str) -> PushNotificationConfig:
        """Get push notification information for a task."""
        async with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                raise ValueError(f"Task not found for {task_id}")

            return self.push_notification_infos.get(task_id)

    async def has_push_notification_info(self, task_id: str) -> bool:
        """Check if push notification information exists for a task."""
        async with self.lock:
            return task_id in self.push_notification_infos

    async def upsert_task(self, task_send_params: TaskSendParams) -> Task:
        """Create or update a task."""
        async with self.lock:
            task = self.tasks.get(task_send_params.id)
            if task is None:
                task = Task(
                    id=task_send_params.id,
                    sessionId=task_send_params.sessionId,
                    status=TaskStatus(state="submitted"),
                    history=[task_send_params.message],
                )
                self.tasks[task_send_params.id] = task
            else:
                if task.history is None:
                    task.history = []
                task.history.append(task_send_params.message)

            return task

    async def update_store(
        self, task_id: str, status: TaskStatus, artifacts: List[Artifact] = None
    ) -> Task:
        """Update a task's status and artifacts."""
        async with self.lock:
            try:
                task = self.tasks[task_id]
            except KeyError:
                logger.error(f"Task {task_id} not found for updating the task")
                raise ValueError(f"Task {task_id} not found")

            task.status = status

            if status.message is not None and task.history is not None:
                task.history.append(status.message)

            if artifacts:
                if task.artifacts is None:
                    task.artifacts = []
                task.artifacts.extend(artifacts)

            return task

    def append_task_history(self, task: Task, history_length: int = None) -> Task:
        """Create a copy of a task with limited history."""
        new_task = task.model_copy()
        if history_length is not None and history_length > 0 and new_task.history:
            new_task.history = new_task.history[-history_length:]
        else:
            new_task.history = []

        return new_task        

    async def setup_sse_consumer(self, task_id: str, is_resubscribe: bool = False) -> asyncio.Queue:
        """Set up an SSE consumer for a task."""
        async with self.subscriber_lock:
            if task_id not in self.task_sse_subscribers:
                if is_resubscribe:
                    raise ValueError("Task not found for resubscription")
                else:
                    self.task_sse_subscribers[task_id] = []

            sse_event_queue = asyncio.Queue(maxsize=0)  # No limit
            self.task_sse_subscribers[task_id].append(sse_event_queue)
            return sse_event_queue

    async def enqueue_events_for_sse(self, task_id: str, event):
        """Enqueue events for SSE subscribers."""
        async with self.subscriber_lock:
            if task_id not in self.task_sse_subscribers:
                return
            
            for queue in self.task_sse_subscribers[task_id]:
                await queue.put(event)

    async def dequeue_events_for_sse(
        self, request_id, task_id, sse_event_queue: asyncio.Queue
    ) -> AsyncIterable[SendTaskStreamingResponse]:
        """Dequeue and yield events for SSE."""
        try:
            while True:
                event = await sse_event_queue.get()
                if isinstance(event, (TaskStatusUpdateEvent, TaskArtifactUpdateEvent)):
                    yield SendTaskStreamingResponse(id=request_id, result=event)
                    if event.final:
                        break
                elif isinstance(event, JSONRPCResponse):
                    yield event
                    break
                else:
                    # Unknown event type
                    yield SendTaskStreamingResponse(
                        id=request_id,
                        error=InternalError(message=f"Unknown event type: {type(event)}"),
                    )
                    break
        finally:
            # Clean up the subscriber
            async with self.subscriber_lock:
                if task_id in self.task_sse_subscribers:
                    if sse_event_queue in self.task_sse_subscribers[task_id]:
                        self.task_sse_subscribers[task_id].remove(sse_event_queue)
                    if not self.task_sse_subscribers[task_id]:
                        del self.task_sse_subscribers[task_id] 
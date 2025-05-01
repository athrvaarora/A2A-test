import httpx
from httpx_sse import connect_sse
from typing import Any, AsyncIterable, Dict
import json
import asyncio
from .types import (
    AgentCard,
    GetTaskRequest,
    SendTaskRequest,
    SendTaskResponse,
    JSONRPCRequest,
    GetTaskResponse,
    CancelTaskResponse,
    CancelTaskRequest,
    SetTaskPushNotificationRequest,
    SetTaskPushNotificationResponse,
    GetTaskPushNotificationRequest,
    GetTaskPushNotificationResponse,
    A2AClientHTTPError,
    A2AClientJSONError,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
)


class A2AClient:
    def __init__(self, agent_card: AgentCard = None, url: str = None):
        if agent_card:
            self.url = agent_card.url
        elif url:
            self.url = url
        else:
            raise ValueError("Must provide either agent_card or url")

    async def get_agent_card(self) -> AgentCard:
        """Fetch the agent card from the well-known location."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.url}/.well-known/agent.json")
                response.raise_for_status()
                return AgentCard(**response.json())
            except httpx.HTTPStatusError as e:
                raise A2AClientHTTPError(e.response.status_code, str(e))
            except json.JSONDecodeError as e:
                raise A2AClientJSONError(str(e))

    async def send_task(self, payload: Dict[str, Any]) -> SendTaskResponse:
        """Send a task to the agent."""
        request = SendTaskRequest(params=payload)
        return SendTaskResponse(**await self._send_request(request))

    async def send_task_streaming(
        self, payload: Dict[str, Any]
    ) -> AsyncIterable[SendTaskStreamingResponse]:
        """Send a task and get streaming updates."""
        request = SendTaskStreamingRequest(params=payload)
        client = httpx.Client(timeout=None)
        try:
            with connect_sse(
                client, "POST", self.url, json=request.model_dump()
            ) as event_source:
                for sse in event_source.iter_sse():
                    yield SendTaskStreamingResponse(**json.loads(sse.data))
        except json.JSONDecodeError as e:
            raise A2AClientJSONError(str(e))
        except httpx.RequestError as e:
            raise A2AClientHTTPError(400, str(e))
        finally:
            client.close()

    async def _send_request(self, request: JSONRPCRequest) -> Dict[str, Any]:
        """Send a JSON-RPC request to the agent."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.url, json=request.model_dump(), timeout=30
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                raise A2AClientHTTPError(e.response.status_code, str(e))
            except json.JSONDecodeError as e:
                raise A2AClientJSONError(str(e))

    async def get_task(self, payload: Dict[str, Any]) -> GetTaskResponse:
        """Get the status of a task."""
        request = GetTaskRequest(params=payload)
        return GetTaskResponse(**await self._send_request(request))

    async def cancel_task(self, payload: Dict[str, Any]) -> CancelTaskResponse:
        """Cancel a task."""
        request = CancelTaskRequest(params=payload)
        return CancelTaskResponse(**await self._send_request(request))

    async def set_task_callback(
        self, payload: Dict[str, Any]
    ) -> SetTaskPushNotificationResponse:
        """Set a push notification callback for a task."""
        request = SetTaskPushNotificationRequest(params=payload)
        return SetTaskPushNotificationResponse(**await self._send_request(request))

    async def get_task_callback(
        self, payload: Dict[str, Any]
    ) -> GetTaskPushNotificationResponse:
        """Get the push notification configuration for a task."""
        request = GetTaskPushNotificationRequest(params=payload)
        return GetTaskPushNotificationResponse(**await self._send_request(request)) 
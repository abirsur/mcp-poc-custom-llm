from contextlib import AsyncExitStack
from collections.abc import Sequence
import json
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


class MCPClient:
    def __init__(
        self,
        server_url: str,
        *,
        server_name: str = "default",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.server_url = server_url
        self.server_name = server_name
        self.headers = headers or {}
        self._client: MultiServerMCPClient | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools_map: dict[str, Any] | None = None

    def _build_client(self) -> MultiServerMCPClient:
        config: dict[str, Any] = {
            self.server_name: {
                "transport": "streamable_http",
                "url": self.server_url,
            }
        }
        if self.headers:
            config[self.server_name]["headers"] = self.headers
        return MultiServerMCPClient(config)

    async def __aenter__(self) -> "MCPClient":
        self._client = self._build_client()
        self._exit_stack = AsyncExitStack()
        session = await self._exit_stack.enter_async_context(self._client.session(self.server_name))
        tools = await load_mcp_tools(session)
        self._tools_map = {tool.name: tool for tool in tools}
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._client = None
        self._exit_stack = None
        self._tools_map = None

    async def _get_tools_map(self) -> dict[str, Any]:
        if self._tools_map is not None:
            return self._tools_map

        client = self._build_client()
        tools = await client.get_tools(server_name=self.server_name)
        return {tool.name: tool for tool in tools}

    def _normalize_result(self, result: Any) -> Any:
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return result
        return result

    async def list_tools(self) -> list[dict[str, Any]]:
        tools_map = await self._get_tools_map()
        tools_metadata: list[dict[str, Any]] = []
        for tool in tools_map.values():
            args_schema = getattr(tool, "args_schema", None)
            input_schema = None
            if args_schema is not None and hasattr(args_schema, "model_json_schema"):
                input_schema = args_schema.model_json_schema()

            tools_metadata.append(
                {
                    "name": tool.name,
                    "description": getattr(tool, "description", None),
                    "input_schema": input_schema,
                }
            )
        return tools_metadata

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        tools_map = await self._get_tools_map()
        tool = tools_map.get(tool_name)
        if tool is None:
            available_tools = ", ".join(sorted(tools_map))
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {available_tools}")

        result = self._normalize_result(await tool.ainvoke(arguments or {}))
        return {
            "tool_name": tool_name,
            "arguments": arguments or {},
            "result": result,
        }

    async def call_tools(self, operations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        tools_map = await self._get_tools_map()
        results: list[dict[str, Any]] = []
        for operation in operations:
            tool_name = operation["tool_name"]
            arguments = operation.get("arguments", {})
            tool = tools_map.get(tool_name)
            if tool is None:
                available_tools = ", ".join(sorted(tools_map))
                raise ValueError(f"Tool '{tool_name}' not found. Available tools: {available_tools}")

            result = self._normalize_result(await tool.ainvoke(arguments))
            results.append(
                {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result,
                }
            )
        return results

from collections.abc import Sequence
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client


class MCPClient:
    def __init__(self, server_url: str) -> None:
        self.server_url = server_url

    async def get_server_info(self) -> dict[str, str]:
        async with streamable_http_client(self.server_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                init_result = await session.initialize()
                return {
                    "name": init_result.serverInfo.name,
                    "version": init_result.serverInfo.version,
                }

    async def list_tools(self) -> list[dict[str, Any]]:
        async with streamable_http_client(self.server_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()
                return [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    for tool in result.tools
                ]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        async with streamable_http_client(self.server_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments or {})
                return {
                    "tool_name": tool_name,
                    "arguments": arguments or {},
                    "is_error": result.isError,
                    "content": [item.model_dump(mode="json") for item in result.content],
                    "structured_content": result.structuredContent,
                }

    async def call_tools(self, operations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        async with streamable_http_client(self.server_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                results: list[dict[str, Any]] = []
                for operation in operations:
                    tool_name = operation["tool_name"]
                    arguments = operation.get("arguments", {})
                    result = await session.call_tool(tool_name, arguments)
                    results.append(
                        {
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "is_error": result.isError,
                            "content": [item.model_dump(mode="json") for item in result.content],
                            "structured_content": result.structuredContent,
                        }
                    )
                return results

import asyncio
import sys
import traceback
from urllib.parse import urlparse
from typing import Optional

from mcp import ClientSession
from mcp.client.sse import sse_client


def print_items(name: str, result: any) -> None:
    """Print items with formatting.

    Args:
        name: Category name (tools/resources/prompts)
        result: Result object containing items list
    """
    print(f"\nAvailable {name}:")
    items = getattr(result, name)
    if items:
        for item in items:
            print(" *", item)
    else:
        print("No items available")


async def main(server_url: str, operation: str = None, *args):
    """Connect to the MCP server and perform story operations.

    Args:
        server_url: Full URL to SSE endpoint (e.g. http://localhost:8000/sse)
        operation: The operation to perform (generate/list/summarize/update)
        args: Additional arguments for the operation
    """
    if urlparse(server_url).scheme not in ("http", "https"):
        print("Error: Server URL must start with http:// or https://")
        sys.exit(1)

    try:
        async with sse_client(server_url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                print("Connected to MCP server at", server_url)
                print_items("tools", await session.list_tools())
                print_items("resources", await session.list_resources())
                print_items("prompts", await session.list_prompts())

                if operation:
                    try:
                        if operation == "generate":
                            if len(args) < 2:
                                print("Error: generate operation requires prompt and output file")
                                return
                            prompt, output_file = args[0], args[1]
                            response = await session.call_tool(
                                "generate_and_save_story",
                                arguments={
                                    "prompt": prompt,
                                    "file_path": output_file
                                }
                            )
                            print("\n=== Story Generation Result ===\n")
                            print(response)

                        elif operation == "list":
                            directory = args[0] if args else "stories"
                            response = await session.call_tool(
                                "list_stories",
                                arguments={"directory": directory}
                            )
                            print("\n=== Available Stories ===\n")
                            for story in response:
                                print(f" * {story}")

                        elif operation == "summarize":
                            if not args:
                                print("Error: summarize operation requires story file path")
                                return
                            response = await session.call_tool(
                                "summarize_story",
                                arguments={"file_path": args[0]}
                            )
                            print("\n=== Story Summary ===\n")
                            print(response)

                        elif operation == "update":
                            if len(args) < 2:
                                print("Error: update operation requires file path and update prompt")
                                return
                            file_path, update_prompt = args[0], args[1]
                            response = await session.call_tool(
                                "update_story",
                                arguments={
                                    "file_path": file_path,
                                    "update_prompt": update_prompt
                                }
                            )
                            print("\n=== Story Update Result ===\n")
                            print(response)

                        else:
                            print(f"Error: Unknown operation '{operation}'")

                    except Exception as tool_exc:
                        print(f"Error performing {operation} operation:")
                        traceback.print_exception(
                            type(tool_exc), tool_exc, tool_exc.__traceback__
                        )
    except Exception as e:
        print(f"Error connecting to server: {e}")
        traceback.print_exception(type(e), e, e.__traceback__)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python client.py <server_url> [<operation> <args...>]\n"
            "Operations:\n"
            "  generate <prompt> <output_file>  - Generate a new story\n"
            "  list [directory]                 - List all stories in directory\n"
            "  summarize <file_path>            - Summarize a story\n"
            "  update <file_path> <prompt>      - Update an existing story\n"
            "\nExamples:\n"
            "  python client.py http://localhost:8000/sse generate \"Write a story about a magical forest\" stories/forest.txt\n"
            "  python client.py http://localhost:8000/sse list stories\n"
            "  python client.py http://localhost:8000/sse summarize stories/forest.txt\n"
            "  python client.py http://localhost:8000/sse update stories/forest.txt \"Make the story more dramatic\""
        )
        sys.exit(1)

    server_url = sys.argv[1]
    operation = sys.argv[2] if len(sys.argv) > 2 else None
    args = sys.argv[3:] if len(sys.argv) > 3 else []
    asyncio.run(main(server_url, operation, *args))

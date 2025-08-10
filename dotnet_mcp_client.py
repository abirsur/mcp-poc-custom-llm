import os
import asyncio
import json
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, ToolMessage

load_dotenv()
async def main():
    # Make sure to set your GOOGLE_API_KEY environment variable
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    client = MultiServerMCPClient({
        "dotnet_converter": {
            "url": "http://localhost:8000/sse",
            "transport": "sse"
        }
    }) 

    tools = await client.get_tools()
    agent = create_react_agent(model, tools)

    # Test C# to VB.NET conversion
    csharp_code = r"C:\Users\abirs\OneDrive\office\projects\AspNet-WebApi-Sample-master\AspNet-WebApi-Sample-master"
    csharp_code_converted = r"C:\Users\abirs\OneDrive\office\projects\AspNet-WebApi-Sample-master\AspNet-WebApi-Sample-master\converted"
    prompt = (
        f"""
Convert the code present in this path {csharp_code}. Create the project in this path {csharp_code_converted} to .NET 8 Web API applications and generate a log and README file.
The project should be created in the path {csharp_code_converted} and the code should be converted to .NET 8 Web API applications.
The log should be generated in the path {csharp_code_converted} and the README file should be generated in the path {csharp_code_converted}.
"""
    )

    print("Starting conversion with ReAct agent...\n")
    inputs = {"messages": [{"role": "user", "content": prompt}]}

    processed_index = 0
    final_answer = None

    async for state in agent.astream(inputs, stream_mode="values"):
        messages = state.get("messages", [])

        # Print only newly added messages
        while processed_index < len(messages):
            message = messages[processed_index]
            processed_index += 1

            if isinstance(message, AIMessage):  
                tool_calls = getattr(message, "tool_calls", None)
                if tool_calls:
                    # Thought is intentionally summarized to avoid exposing hidden reasoning
                    print("Thought: Decided to use a tool based on the current context.")
                    for call in tool_calls:
                        tool_name = call.get("name", "unknown_tool")
                        tool_args = call.get("args", {})
                        print(f"Action: {tool_name}")
                        try:
                            print("Action Input:")
                            print(json.dumps(tool_args, indent=2))
                        except Exception:
                            print(str(tool_args))
                        print("")
                else:
                    # No tool calls => likely final assistant response
                    if message.content:
                        final_answer = message.content
            elif isinstance(message, ToolMessage):
                # Observation from a tool
                print("Observation:")
                try:
                    if isinstance(message.content, str):
                        print(message.content)
                    else:
                        print(json.dumps(message.content, indent=2))
                except Exception:
                    print(str(message.content))
                print("")

    if final_answer:
        print("Final Answer:")
        if isinstance(final_answer, str):
            print(final_answer)
        else:
            print(json.dumps(final_answer, indent=2))

if __name__ == "__main__":
   asyncio.run(main())

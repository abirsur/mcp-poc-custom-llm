import streamlit as st
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
import os
from pathlib import Path

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'tools' not in st.session_state:
    st.session_state.tools = []

async def initialize_session():
    """Initialize the MCP session and get available tools."""
    try:
        async with sse_client("http://localhost:8000/sse") as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                tools = await session.list_tools()
                return tools.tools
    except Exception as e:
        st.error(f"Error connecting to server: {str(e)}")
        return []

async def call_tool(tool_name: str, **kwargs):
    """Call a specific tool with the given arguments."""
    try:
        async with sse_client("http://localhost:8000/sse") as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                response = await session.call_tool(tool_name, arguments=kwargs)
                return response
    except Exception as e:
        st.error(f"Error calling tool: {str(e)}")
        return None

def main():
    st.title("Story Teller Assistant")
    
    # Initialize tools if not already done
    if not st.session_state.tools:
        st.session_state.tools = asyncio.run(initialize_session())
    
    # Display available tools
    st.sidebar.title("Available Tools")
    for tool in st.session_state.tools:
        st.sidebar.write(f"🔧 {tool}")
    
    # Main chat interface
    st.write("### Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to do?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process the prompt
        try:
            # Check if the prompt is a tool command
            if prompt.startswith("generate"):
                _, story_prompt, file_path = prompt.split("|")
                response = asyncio.run(call_tool(
                    "generate_and_save_story",
                    prompt=story_prompt.strip(),
                    file_path=file_path.strip()
                ))
            elif prompt.startswith("list"):
                directory = prompt.split("|")[1].strip() if "|" in prompt else "stories"
                response = asyncio.run(call_tool(
                    "list_stories",
                    directory=directory
                ))
            elif prompt.startswith("summarize"):
                file_path = prompt.split("|")[1].strip()
                response = asyncio.run(call_tool(
                    "summarize_story",
                    file_path=file_path
                ))
            elif prompt.startswith("update"):
                _, file_path, update_prompt = prompt.split("|")
                response = asyncio.run(call_tool(
                    "update_story",
                    file_path=file_path.strip(),
                    update_prompt=update_prompt.strip()
                ))
            else:
                response = "Please use one of the following commands:\n" + \
                          "- generate | <story prompt> | <output file>\n" + \
                          "- list | [directory]\n" + \
                          "- summarize | <file path>\n" + \
                          "- update | <file path> | <update prompt>"
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)

if __name__ == "__main__":
    main() 
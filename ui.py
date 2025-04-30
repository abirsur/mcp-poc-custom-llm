import streamlit as st
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
import os
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

def get_llm():
    """Initialize the Azure OpenAI LLM."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

def create_tool_selection_prompt(tools):
    """Create a prompt for the LLM to select the appropriate tool."""
    tool_descriptions = "\n".join([f"- {tool}" for tool in tools])
    return f"""You are an AI assistant that helps users with story-related tasks. Based on the user's request, select the most appropriate tool to use.

Available tools:
{tool_descriptions}

The user's request is: {{user_input}}

Respond with the tool name and its arguments in JSON format. For example:
{{
    "tool": "generate_and_save_story",
    "arguments": {{
        "prompt": "A story about a brave knight",
        "file_path": "stories/knight.txt"
    }}
}}

If no tool matches the request, respond with:
{{
    "tool": null,
    "message": "I couldn't find a suitable tool for your request. Please try rephrasing or ask for help with available tools."
}}
"""

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
        
        # Process the prompt using LLM
        try:
            llm = get_llm()
            prompt_template = create_tool_selection_prompt(st.session_state.tools)
            response = llm.invoke(prompt_template.format(user_input=prompt))
            
            # Parse the LLM response
            import json
            try:
                tool_selection = json.loads(response.content)
                
                if tool_selection["tool"] is None:
                    # No suitable tool found
                    st.session_state.messages.append({"role": "assistant", "content": tool_selection["message"]})
                    with st.chat_message("assistant"):
                        st.write(tool_selection["message"])
                else:
                    # Call the selected tool
                    tool_response = asyncio.run(call_tool(
                        tool_selection["tool"],
                        **tool_selection["arguments"]
                    ))
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": tool_response})
                    with st.chat_message("assistant"):
                        st.write(tool_response)
                        
            except json.JSONDecodeError:
                error_msg = "I had trouble understanding your request. Please try rephrasing it."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)

if __name__ == "__main__":
    main() 
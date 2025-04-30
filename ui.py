import streamlit as st
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
import os
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import json
from typing import Dict, Any, List

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
                tools_response = await session.list_tools()
                # Ensure we have a list of tools
                if hasattr(tools_response, 'tools'):
                    return tools_response.tools
                elif isinstance(tools_response, list):
                    return tools_response
                else:
                    st.error("Unexpected tools response format")
                    return []
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

def create_tool_identification_prompt(tools: List[Any]) -> str:
    """Create a prompt for the LLM to identify the appropriate tool."""
    tool_descriptions = []
    for tool in tools:
        # Handle different tool object structures
        tool_name = getattr(tool, 'name', str(tool))
        tool_desc = getattr(tool, 'description', 'No description available')
        
        tool_info = f"- {tool_name}: {tool_desc}\n"
        if hasattr(tool, 'parameters'):
            tool_info += "  Parameters:\n"
            for param in tool.parameters:
                param_name = getattr(param, 'name', str(param))
                param_desc = getattr(param, 'description', 'No description available')
                tool_info += f"    - {param_name}: {param_desc}\n"
        tool_descriptions.append(tool_info)
    
    tool_descriptions_str = "\n".join(tool_descriptions)
    
    return f"""You are an AI assistant that helps users with story-related tasks. Based on the user's request, identify the most appropriate tool to use.

Available tools:
{tool_descriptions_str}

The user's request is: {{user_input}}

Analyze the request and identify the most appropriate tool. Consider:
1. The tool's purpose
2. The required parameters
3. Whether the user's request matches the tool's capabilities

Respond with the tool name in JSON format. For example:
{{
    "tool": "generate_and_save_story",
    "confidence": 0.9,
    "reason": "The user wants to create a new story, which matches this tool's purpose"
}}

If no tool matches the request, respond with:
{{
    "tool": null,
    "confidence": 0.0,
    "reason": "No suitable tool found for this request"
}}
"""

def create_parameter_formatting_prompt(tool: Any, user_input: str) -> str:
    """Create a prompt for the LLM to format parameters for the selected tool."""
    tool_name = getattr(tool, 'name', str(tool))
    tool_desc = getattr(tool, 'description', 'No description available')
    
    tool_info = f"Tool: {tool_name}\nDescription: {tool_desc}\n"
    if hasattr(tool, 'parameters'):
        tool_info += "Parameters:\n"
        for param in tool.parameters:
            param_name = getattr(param, 'name', str(param))
            param_desc = getattr(param, 'description', 'No description available')
            tool_info += f"- {param_name}: {param_desc}\n"
    
    return f"""You are an AI assistant that helps format parameters for tool calls. Based on the user's request and the selected tool, format the parameters appropriately.

Tool Information:
{tool_info}

User's request: {user_input}

Format the parameters in JSON format. For example, if the tool is generate_and_save_story and the user says "write a story about a dragon", you might respond with:
{{
    "prompt": "A story about a dragon",
    "file_path": "stories/dragon.txt"
}}

Respond with only the JSON object containing the formatted parameters.
"""

def main():
    st.title("Story Teller Assistant")
    
    # Initialize tools if not already done
    if not st.session_state.tools:
        st.session_state.tools = asyncio.run(initialize_session())
        if not st.session_state.tools:
            st.error("Failed to initialize tools. Please check the server connection.")
            return
    
    # Display available tools
    st.sidebar.title("Available Tools")
    for tool in st.session_state.tools:
        tool_name = getattr(tool, 'name', str(tool))
        tool_desc = getattr(tool, 'description', 'No description available')
        st.sidebar.write(f"🔧 {tool_name}")
        st.sidebar.write(f"   {tool_desc}")
    
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
            
            # Step 1: Identify the appropriate tool
            tool_identification_prompt = create_tool_identification_prompt(st.session_state.tools)
            tool_selection = json.loads(llm.invoke(tool_identification_prompt.format(user_input=prompt)).content)
            
            if tool_selection["tool"] is None:
                # No suitable tool found
                st.session_state.messages.append({"role": "assistant", "content": tool_selection["reason"]})
                with st.chat_message("assistant"):
                    st.write(tool_selection["reason"])
            else:
                # Step 2: Format parameters for the selected tool
                selected_tool = next((tool for tool in st.session_state.tools 
                                   if getattr(tool, 'name', str(tool)) == tool_selection["tool"]), None)
                
                if selected_tool is None:
                    error_msg = f"Could not find tool: {tool_selection['tool']}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    with st.chat_message("assistant"):
                        st.error(error_msg)
                    return
                
                parameter_prompt = create_parameter_formatting_prompt(selected_tool, prompt)
                formatted_parameters = json.loads(llm.invoke(parameter_prompt).content)
                
                # Step 3: Call the tool with formatted parameters
                tool_response = asyncio.run(call_tool(
                    tool_selection["tool"],
                    **formatted_parameters
                ))
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": tool_response})
                with st.chat_message("assistant"):
                    st.write(tool_response)
                
        except json.JSONDecodeError as e:
            error_msg = f"I had trouble understanding the response: {str(e)}"
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
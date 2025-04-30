import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
from html2text import html2text
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from datetime import datetime

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route, Mount

from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INTERNAL_ERROR, INVALID_PARAMS
from mcp.server.sse import SseServerTransport

# Load environment variables
load_dotenv()

# Define Pydantic models
class Story(BaseModel):
    title: str = Field(description="The title of the story")
    content: str = Field(description="The main content of the story")
    genre: str = Field(description="The genre of the story")
    moral: Optional[str] = Field(description="The moral or lesson of the story", default=None)

class StorySummary(BaseModel):
    title: str
    genre: str
    summary: str
    moral: Optional[str]
    created_at: datetime

# Initialize Azure OpenAI with LangChain
llm = AzureChatOpenAI(
    openai_api_version="2024-02-15-preview",
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.7
)

# Create prompt templates
story_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative storyteller. Generate a story based on the given prompt."),
    ("user", "{prompt}")
])

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a story summarizer. Create a concise summary of the given story."),
    ("user", "{story}")
])

# Create output parsers
story_parser = PydanticOutputParser(pydantic_object=Story)
summary_parser = PydanticOutputParser(pydantic_object=StorySummary)

mcp = FastMCP("asur_story_teller")

@mcp.tool()
def read_wikipedia_article(url: str) -> str:
    """
    Fetch a Wikipedia article at the provided URL, parse its main content,
    convert it to Markdown, and return the resulting text.

    Usage:
        read_wikipedia_article("https://en.wikipedia.org/wiki/Python_(programming_language)")
    """
    try:
        if not url.startswith("http"):
            raise ValueError("URL must start with http or https.")

        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to retrieve the article. HTTP status code: {response.status_code}"
                )
            )

        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.find("div", {"id": "mw-content-text"})
        if not content_div:
            raise McpError(
                ErrorData(
                    INVALID_PARAMS,
                    "Could not find the main content on the provided Wikipedia URL."
                )
            )

        markdown_text = html2text(str(content_div))
        return markdown_text

    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def generate_and_save_story(prompt: str, file_path: str) -> str:
    """
    Generate a story using Azure OpenAI and LangChain, then save it to the specified file path.
    
    Args:
        prompt: The prompt to generate the story
        file_path: The path where the story should be saved
        
    Returns:
        str: The generated story content
    """
    try:
        # Create the chain
        chain = story_prompt | llm | story_parser
        
        # Generate the story
        story = chain.invoke({"prompt": prompt})
        
        # Format the story content
        story_content = f"Title: {story.title}\n\n"
        story_content += f"Genre: {story.genre}\n\n"
        story_content += f"Story:\n{story.content}\n\n"
        if story.moral:
            story_content += f"Moral: {story.moral}\n"
        
        # Ensure the directory exists
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the story to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(story_content)
            
        return f"Story has been generated and saved to {file_path}"
        
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Error generating or saving story: {str(e)}")) from e

@mcp.tool()
def list_stories(directory: str) -> List[str]:
    """
    List all story files in the specified directory.
    
    Args:
        directory: The directory to search for stories
        
    Returns:
        List[str]: List of story file paths
    """
    try:
        story_dir = Path(directory)
        if not story_dir.exists():
            return []
        
        story_files = [str(f) for f in story_dir.glob("*.txt")]
        return story_files
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Error listing stories: {str(e)}")) from e

@mcp.tool()
def summarize_story(file_path: str) -> str:
    """
    Generate a summary of a story from a file.
    
    Args:
        file_path: Path to the story file
        
    Returns:
        str: Summary of the story
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            story_content = f.read()
        
        # Create the summary chain
        chain = summary_prompt | llm | summary_parser
        
        # Generate the summary
        summary = chain.invoke({"story": story_content})
        
        # Format the summary
        summary_content = f"Title: {summary.title}\n"
        summary_content += f"Genre: {summary.genre}\n"
        summary_content += f"Summary: {summary.summary}\n"
        if summary.moral:
            summary_content += f"Moral: {summary.moral}\n"
        summary_content += f"Created: {summary.created_at}\n"
        
        return summary_content
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Error summarizing story: {str(e)}")) from e

@mcp.tool()
def update_story(file_path: str, update_prompt: str) -> str:
    """
    Update an existing story based on the update prompt.
    
    Args:
        file_path: Path to the story file
        update_prompt: Instructions for updating the story
        
    Returns:
        str: Confirmation message
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            story_content = f.read()
        
        # Create the update chain
        update_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a story editor. Update the following story based on the given instructions."),
            ("user", f"Original story:\n{story_content}\n\nUpdate instructions:\n{update_prompt}")
        ])
        
        chain = update_prompt_template | llm | story_parser
        
        # Generate the updated story
        updated_story = chain.invoke({})
        
        # Format the updated story content
        updated_content = f"Title: {updated_story.title}\n\n"
        updated_content += f"Genre: {updated_story.genre}\n\n"
        updated_content += f"Story:\n{updated_story.content}\n\n"
        if updated_story.moral:
            updated_content += f"Moral: {updated_story.moral}\n"
        
        # Save the updated story
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
            
        return f"Story has been updated and saved to {file_path}"
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Error updating story: {str(e)}")) from e

# Set up the SSE transport for MCP communication.
sse = SseServerTransport("/messages/")

async def handle_sse(request: Request) -> None:
    _server = mcp._mcp_server
    async with sse.connect_sse(
        request.scope,
        request.receive,
        request._send,
    ) as (reader, writer):
        await _server.run(reader, writer, _server.create_initialization_options())

# Create the Starlette app with two endpoints:
# - "/sse": for SSE connections from clients.
# - "/messages/": for handling incoming POST messages.
app = Starlette(
    debug=True,
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ],
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

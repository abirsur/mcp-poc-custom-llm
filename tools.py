"""
MCP Tool Generator
------------------
Generates MCP-compliant tools using:
- Structure templates (prompt scaffolds)
- API knowledge configuration (knowledge.json)
- Azure OpenAI for code generation

This file ONLY contains MCP tools.
"""

import os
import json
from fastmcp import FastMCP
from openai import AzureOpenAI

# -------------------------------------------------------------------
# MCP SERVER
# -------------------------------------------------------------------

mcp = FastMCP(
    name="MCP-Tool-Generator",
    description="Generates MCP tools from structure + knowledge using Azure OpenAI"
)

# -------------------------------------------------------------------
# TOOL 1: Load MCP Tool Structure (Prompt Template)
# -------------------------------------------------------------------

@mcp.tool()
def load_tool_structure(tool_type: str) -> str:
    """
    Loads the MCP tool structure and examples for a given tool type.
    Example tool_type: merge_api, nested_api, single_api
    """
    path = f"templates/{tool_type}.txt"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Tool structure not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# -------------------------------------------------------------------
# TOOL 2: Select Relevant API Knowledge
# -------------------------------------------------------------------

@mcp.tool()
def select_relevant_knowledge(tool_type: str, user_purpose: str) -> dict:
    """
    Reads knowledge.json and extracts API definitions relevant
    to the requested MCP tool type.
    """
    with open("knowledge.json", "r", encoding="utf-8") as f:
        knowledge = json.load(f)

    api_names = knowledge.get("tool_capabilities", {}).get(tool_type, [])

    if not api_names:
        raise ValueError(f"No APIs mapped for tool_type: {tool_type}")

    return {
        "purpose": user_purpose,
        "apis": {
            api: knowledge["apis"][api]
            for api in api_names
        }
    }

# -------------------------------------------------------------------
# TOOL 3: Build LLM Prompt
# -------------------------------------------------------------------

@mcp.tool()
def build_llm_prompt(
    structure: str,
    knowledge: dict,
    tool_type: str
) -> str:
    """
    Assembles the final prompt for the LLM.
    """
    return f"""
{structure}

--- USER INTENT ---
Tool Type: {tool_type}
Purpose: {knowledge['purpose']}

--- API KNOWLEDGE ---
{json.dumps(knowledge['apis'], indent=2)}

--- INSTRUCTIONS ---
Generate a COMPLETE, VALID, MCP-compliant Python tool.
Do NOT include explanations.
Return ONLY code.
"""

# -------------------------------------------------------------------
# TOOL 4: Generate MCP Tool Code (Azure OpenAI)
# -------------------------------------------------------------------

@mcp.tool()
def generate_mcp_tool_code(prompt: str) -> str:
    """
    Uses Azure OpenAI to generate MCP tool code.
    All configuration is read from environment variables.
    """

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_MODEL"),
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are an expert MCP tool generator."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# -------------------------------------------------------------------
# TOOL 5: Basic MCP Compliance Validation (Optional but Recommended)
# -------------------------------------------------------------------

@mcp.tool()
def validate_mcp_tool_code(code: str) -> dict:
    """
    Performs lightweight validation to ensure MCP compliance.
    """
    checks = {
        "has_mcp_decorator": "@mcp.tool" in code,
        "has_async_def": "async def" in code,
        "uses_fastmcp": "FastMCP" in code,
        "uses_httpx": "httpx" in code
    }

    return {
        "valid": all(checks.values()),
        "checks": checks
    }

# -------------------------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()

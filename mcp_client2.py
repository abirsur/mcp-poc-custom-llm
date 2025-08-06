#!/usr/bin/env python3
"""
MCP Client with ReActChain for Data Post-Processing
Integrates with Azure OpenAI GPT-4o for intelligent data processing
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
from datetime import datetime

# Azure OpenAI imports
from openai import AsyncAzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# MCP client imports
from mcp import ClientSession, HttpPostTransport, ServerId
from mcp.client.http import http_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReActStep:
    """Represents a step in ReAct reasoning chain"""
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    step_number: int

class ReActChain:
    """
    ReAct (Reasoning and Acting) Chain for data processing pipeline
    """
    
    def __init__(self, azure_client: AsyncAzureOpenAI, mcp_session: ClientSession):
        self.azure_client = azure_client
        self.mcp_session = mcp_session
        self.steps: List[ReActStep] = []
        self.available_tools = []
        
    async def initialize(self):
        """Initialize the chain by fetching available tools"""
        try:
            tools_response = await self.mcp_session.list_tools()
            self.available_tools = tools_response.tools
            logger.info(f"Initialized with {len(self.available_tools)} tools")
        except Exception as e:
            logger.error(f"Failed to initialize ReActChain: {e}")
            raise
    
    def create_system_prompt(self) -> str:
        """Create system prompt for ReAct reasoning with detailed tool schemas"""
        tools_description = []
        for tool in self.available_tools:
            schema = tool.inputSchema
            required_fields = schema.get('required', [])
            properties = schema.get('properties', {})
            
            tool_desc = f"- {tool.name}: {tool.description}\n"
            tool_desc += f"  Required parameters: {required_fields}\n"
            tool_desc += "  Parameters:\n"
            
            for prop, details in properties.items():
                prop_type = details.get('type', 'unknown')
                default_val = details.get('default', 'N/A')
                tool_desc += f"    - {prop} ({prop_type}): {details.get('description', 'No description')}"
                if default_val != 'N/A':
                    tool_desc += f" [default: {default_val}]"
                tool_desc += "\n"
            
            tools_description.append(tool_desc)
        
        tools_text = "\n".join(tools_description)
        
        return f"""You are an expert data processing agent using ReAct (Reasoning and Acting) methodology.

Available Tools:
{tools_text}

Your task is to process data from multiple sources using a structured approach:

1. FUZZY MATCHING: Match data across sources
2. COMPREHENSIVE VALUE IDENTIFICATION: Find the most complete values
3. LLM ENRICHMENT: Enhance data using language models
4. PRIORITY PROCESSING: Handle data based on priority lists
5. DEPENDENCY MANAGEMENT: Handle field dependencies
6. RAG OPERATIONS: Use retrieval-augmented generation when needed

Use this ReAct format:

Thought: [Your reasoning about what to do next]
Action: [The tool to use]
Action Input: [JSON object with ALL required parameters for the tool]
Observation: [The result from the tool]

Continue this process until you have a final answer.

CRITICAL REQUIREMENTS:
- ALWAYS provide ALL required parameters for each tool as specified in the tool schema
- Use proper JSON format for Action Input
- For tools requiring "sources" parameter, convert data_sources to the proper format
- For tools requiring "field_name", specify which field you're processing
- Always track data sources throughout the process
- Handle dependencies by processing prerequisite fields first
- Use priority lists to select the best data when multiple sources are available
- For complex queries, use RAG operations with raw data context

Begin by analyzing the user's request and planning your approach."""

    async def execute(self, 
                     user_prompt: str, 
                     data_sources: List[Dict], 
                     raw_data: Dict = None,
                     priority_list: List[str] = None,
                     max_iterations: int = 10) -> Dict[str, Any]:
        """
        Execute the ReAct chain for data processing
        """
        # Prepare the initial context
        context = {
            "user_request": user_prompt,
            "data_sources": data_sources,
            "raw_data": raw_data or {},
            "priority_list": priority_list or [],
            "processing_history": []
        }
        
        # Create the initial conversation
        messages = [
            {"role": "system", "content": self.create_system_prompt()},
            {"role": "user", "content": self._create_user_message(context)}
        ]
        
        final_result = {}
        
        for iteration in range(max_iterations):
            try:
                # Get LLM response
                response = await self.azure_client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME"),
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2000
                )
                
                assistant_message = response.choices[0].message.content
                messages.append({"role": "assistant", "content": assistant_message})
                
                # Parse ReAct format
                thought, action, action_input = self._parse_react_response(assistant_message)
                
                if not action:
                    # No action means we're done
                    final_result = self._extract_final_answer(assistant_message)
                    break
                
                # Validate and fix action input based on tool schema
                validated_input = self._validate_and_fix_action_input(action, action_input, context)
                
                # Execute the action
                observation = await self._execute_action(action, validated_input)
                
                # Record the step
                step = ReActStep(
                    thought=thought,
                    action=action,
                    action_input=validated_input,
                    observation=observation,
                    step_number=iteration + 1
                )
                self.steps.append(step)
                
                # Add observation to conversation
                observation_message = f"Observation: {observation}"
                messages.append({"role": "user", "content": observation_message})
                
                # Check if we have a final answer
                if "Final Answer:" in assistant_message:
                    final_result = self._extract_final_answer(assistant_message)
                    break
                    
            except Exception as e:
                logger.error(f"Error in iteration {iteration + 1}: {e}")
                final_result = {"error": str(e), "completed_steps": len(self.steps)}
                break
        
        return {
            "final_result": final_result,
            "steps": [self._step_to_dict(step) for step in self.steps],
            "total_iterations": len(self.steps),
            "context": context
        }
    
    def _create_user_message(self, context: Dict) -> str:
        """Create the initial user message with context"""
        return f"""
User Request: {context['user_request']}

Data Sources ({len(context['data_sources'])} sources):
{json.dumps(context['data_sources'], indent=2)}

Priority List: {context['priority_list']}

Raw Data Available: {'Yes' if context['raw_data'] else 'No'}

Please process this data according to the requirements. Start by analyzing the request and determining the best approach.

IMPORTANT: When using tools, ensure you provide ALL required parameters as specified in the tool schemas.
"""
    
    def _parse_react_response(self, response: str) -> Tuple[str, str, Dict]:
        """Parse ReAct format response"""
        thought_match = re.search(r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer):|$)", response, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*?)(?=\n|$)", response)
        action_input_match = re.search(r"Action Input:\s*(.*?)(?=\n(?:Observation|Thought|Action|Final Answer):|$)", response, re.DOTALL)
        
        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else ""
        
        action_input = {}
        if action_input_match:
            try:
                action_input_str = action_input_match.group(1).strip()
                # Try to parse as JSON
                action_input = json.loads(action_input_str)
            except json.JSONDecodeError:
                # If not JSON, treat as string and try to structure it
                action_input = {"query": action_input_str}
        
        return thought, action, action_input
    
    def _validate_and_fix_action_input(self, action: str, action_input: Dict, context: Dict) -> Dict:
        """Validate and fix action input based on tool schema"""
        # Find the tool schema
        tool_schema = None
        for tool in self.available_tools:
            if tool.name == action:
                tool_schema = tool.inputSchema
                break
        
        if not tool_schema:
            logger.warning(f"No schema found for tool: {action}")
            return action_input
        
        required_fields = tool_schema.get('required', [])
        properties = tool_schema.get('properties', {})
        validated_input = action_input.copy()
        
        # Check and provide missing required fields
        for field in required_fields:
            if field not in validated_input or validated_input[field] is None:
                # Try to infer the missing field
                if field == "sources":
                    validated_input["sources"] = context["data_sources"]
                elif field == "field_name":
                    # Try to extract field name from the context or action input
                    if "field" in action_input:
                        validated_input["field_name"] = action_input["field"]
                    elif "fields" in action_input and isinstance(action_input["fields"], list) and action_input["fields"]:
                        validated_input["field_name"] = action_input["fields"][0]
                    else:
                        # Default to a common field name
                        validated_input["field_name"] = "client_name"
                elif field == "target_value":
                    validated_input["target_value"] = ""  # Empty string for auto-detection
                elif field == "matches":
                    validated_input["matches"] = []
                elif field == "priority_list":
                    validated_input["priority_list"] = context.get("priority_list", [])
                elif field == "fields":
                    # Extract fields from config or use common fields
                    config = context.get("raw_data", {}).get("config", {})
                    if config and "fields" in config:
                        validated_input["fields"] = list(config["fields"].keys())
                    else:
                        validated_input["fields"] = ["client_name", "policy_number", "claim_amount"]
                elif field == "data":
                    validated_input["data"] = context["data_sources"]
                elif field == "enrichment_prompt":
                    validated_input["enrichment_prompt"] = "Please enrich this data with additional context and details."
                elif field == "query":
                    validated_input["query"] = action_input.get("query", "Process and analyze the provided data")
                elif field == "raw_data":
                    validated_input["raw_data"] = context.get("raw_data", {})
                elif field == "field_name" and action == "set_field_dependencies":
                    validated_input["field_name"] = action_input.get("field_name", "client_name")
                elif field == "dependencies":
                    validated_input["dependencies"] = action_input.get("dependencies", [])
                else:
                    logger.warning(f"Could not infer value for required field '{field}' in tool '{action}'")
                    # Set a default value based on the property type
                    prop_type = properties.get(field, {}).get('type', 'string')
                    if prop_type == 'string':
                        validated_input[field] = ""
                    elif prop_type == 'array':
                        validated_input[field] = []
                    elif prop_type == 'object':
                        validated_input[field] = {}
                    elif prop_type == 'number':
                        validated_input[field] = 0
                    elif prop_type == 'boolean':
                        validated_input[field] = False
        
        # Apply default values for optional fields
        for field, prop_details in properties.items():
            if field not in validated_input and 'default' in prop_details:
                validated_input[field] = prop_details['default']
        
        logger.info(f"Validated input for {action}: {validated_input}")
        return validated_input
    
    async def _execute_action(self, action: str, action_input: Dict) -> str:
        """Execute an action using MCP tools"""
        try:
            result = await self.mcp_session.call_tool(action, action_input)
            
            # Extract text content from result
            if result.content:
                observations = []
                for content in result.content:
                    if hasattr(content, 'text'):
                        observations.append(content.text)
                return "\n".join(observations)
            
            return "No result returned from tool"
            
        except Exception as e:
            logger.error(f"Error executing action {action} with input {action_input}: {e}")
            return f"Error: {str(e)}"
    
    def _extract_final_answer(self, response: str) -> Dict:
        """Extract final answer from response"""
        final_answer_match = re.search(r"Final Answer:\s*(.*?)$", response, re.DOTALL)
        
        if final_answer_match:
            answer_text = final_answer_match.group(1).strip()
            try:
                # Try to parse as JSON
                return json.loads(answer_text)
            except json.JSONDecodeError:
                return {"answer": answer_text}
        
        return {"answer": "Processing completed", "status": "success"}
    
    def _step_to_dict(self, step: ReActStep) -> Dict:
        """Convert ReActStep to dictionary"""
        return {
            "step_number": step.step_number,
            "thought": step.thought,
            "action": step.action,
            "action_input": step.action_input,
            "observation": step.observation
        }

class DataProcessingClient:
    """
    Main client for data processing pipeline
    """
    
    def __init__(self, azure_config: Dict[str, str]):
        self.azure_config = azure_config
        self.azure_client = None
        self.mcp_session = None
        self.react_chain = None
    
    async def initialize(self, server_url: str = "http://127.0.0.1:8080"):
        """Initialize the client with Azure OpenAI and MCP server"""
        # Initialize Azure OpenAI client
        self.azure_client = AsyncAzureOpenAI(
            api_key=self.azure_config["api_key"],
            api_version=self.azure_config["api_version"],
            azure_endpoint=self.azure_config["endpoint"]
        )
        
        # Initialize MCP session
        server_id = ServerId(
            url=server_url,
            name="data-postprocessor"
        )
        transport = HttpPostTransport(server_id)
        self.mcp_session = await http_client(transport).__aenter__()
        
        # Initialize ReAct chain
        self.react_chain = ReActChain(self.azure_client, self.mcp_session)
        await self.react_chain.initialize()
        
        logger.info("Data Processing Client initialized successfully")
    
    async def process_data(self, 
                          prompt: str,
                          data_sources: List[Dict],
                          raw_data: Dict = None,
                          priority_list: List[str] = None,
                          processing_options: Dict = None) -> Dict:
        """
        Main method to process data using ReAct chain
        """
        if not self.react_chain:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        
        processing_options = processing_options or {}
        max_iterations = processing_options.get("max_iterations", 10)
        
        logger.info(f"Starting data processing with {len(data_sources)} sources")
        
        result = await self.react_chain.execute(
            user_prompt=prompt,
            data_sources=data_sources,
            raw_data=raw_data,
            priority_list=priority_list,
            max_iterations=max_iterations
        )
        
        return result
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.mcp_session:
            await self.mcp_session.__aexit__(None, None, None)

# Example usage and testing
async def main(config: Dict, extracted_data: List[Dict]):
    """
    Main function to run the data processing client based on config.
    """
    # Azure OpenAI configuration
    azure_config = {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": "2024-05-01-preview"
    }
    
    # Check if Azure configuration is available
    if not all(azure_config.values()):
        logger.error("Azure OpenAI environment variables not set.")
        logger.error("Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT.")
        return

    client = DataProcessingClient(azure_config)
    
    try:
        # Initialize client
        await client.initialize()
        
        # Transform data sources to the required format
        data_sources = []
        for i, source in enumerate(extracted_data):
            data_sources.append({
                "id": source["source_id"],
                "name": source["doc_type"],
                "priority": i + 1, # Default priority
                "data": source["data"],
                "timestamp": source.get("metadata", {}).get("timestamp", ""),
                "confidence": 0.0  # Will be calculated during processing
            })

        # Create a comprehensive prompt for the ReAct agent
        prompt = f"""
Process the data from the provided data sources to determine the best value for each field specified in the configuration.

Configuration:
{json.dumps(config, indent=2)}

Follow the operations and priorities defined in the configuration for each field.
Handle any dependencies between fields. For example, process 'client_name' before 'policy_number'.
Your final answer should be a single JSON object containing the processed data for all fields listed in the configuration.

PROCESSING STEPS:
1. First, set field dependencies as defined in the config
2. Use fuzzy matching to find similar values across sources
3. Apply priority-based processing according to the priority list
4. For each field, identify the most comprehensive value
5. Use LLM enrichment when needed for data enhancement
6. Generate a final consolidated result

Start by setting the dependencies for the fields as defined in the config.
"""

        logger.info("--- Starting Comprehensive Data Processing ---")
        
        # Extract priority list from config if available
        priority_list = []
        if "global_settings" in config and "priority_list" in config["global_settings"]:
            priority_list = config["global_settings"]["priority_list"]
        
        # Process data using the comprehensive prompt
        result = await client.process_data(
            prompt=prompt,
            data_sources=data_sources,
            priority_list=priority_list,
            # Pass raw data if needed for RAG
            raw_data={"extracted": extracted_data, "config": config}
        )
        
        final_processed_data = result.get('final_result', {})
        
        # Print the result
        logger.info("\n--- FINAL PROCESSED DATA ---")
        logger.info(json.dumps(final_processed_data, indent=2))
        
        logger.info("\n--- ReAct Steps ---")
        logger.info(json.dumps(result.get('steps', []), indent=2))
        
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
    
    finally:
        # Cleanup resources
        await client.cleanup()

if __name__ == "__main__":
    # To run this file directly, you can use:
    # with open("config.json", "r") as f:
    #     config_data = json.load(f)
    # with open("extracted_data.json", "r") as f:
    #     extracted_data_json = json.load(f)
    # asyncio.run(main(config_data, extracted_data_json))
    pass
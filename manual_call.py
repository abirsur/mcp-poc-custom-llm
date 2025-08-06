#!/usr/bin/env python3
"""
Simplified MCP Client for Data Post-Processing based on configuration.
This client reads a configuration file and calls MCP tools accordingly
to process data fields.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from mcp import ClientSession, HttpPostTransport, ServerId
from mcp.client.http import http_client

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_mcp_result(result: Any, default_message: str) -> str:
    """Helper function to parse MCP tool call results."""
    if result and result.content:
        texts = [content.text for content in result.content if hasattr(content, 'text') and content.text]
        if texts:
            return "\n".join(texts)
    return default_message


def _safe_parse_json(text: str, default: Dict = None) -> Dict:
    """Safely parse JSON from a string that might contain other content."""
    if default is None:
        default = {}
    try:
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


class SimpleDataProcessingClient:
    """
    A simplified client that processes data by calling MCP tools as defined in a configuration file.
    It follows the instructions in the config to call the matched MCP tools and gets the field-wise output.
    """

    def __init__(self):
        self.mcp_session = None
        self.mcp_client = None
        self._session_initialized = False

        self.operation_to_tool_map = {
            "fuzzy_match": "fuzzy_match_sources",
            "comprehensivematch": "identify_comprehensive_values",
            "llmenrichment": "enrich_data_llm",
            "regex_match": "extract_patterns",
            "validation": "validate_field_data",
            "normalization": "normalize_data",
        }

    async def initialize(self, server_url: str = "http://127.0.0.1:8080"):
        """Initialize the client with the MCP server."""
        try:
            server_id = ServerId(url=server_url, name="data-postprocessor")
            transport = HttpPostTransport(server_id)
            self.mcp_client = http_client(transport)
            self.mcp_session = await self.mcp_client.__aenter__()
            self._session_initialized = True
            logger.info("Simple Data Processing Client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize client: {e}")
            await self.cleanup()
            raise

    async def cleanup(self):
        """Clean up resources properly."""
        try:
            if self.mcp_client and self._session_initialized:
                await self.mcp_client.__aexit__(None, None, None)
                self._session_initialized = False
                logger.info("MCP session cleaned up successfully.")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            self.mcp_session = None
            self.mcp_client = None

    async def call_mcp_tool(self, tool_name: str, parameters: Dict) -> str:
        """Wrapper for MCP tool calls."""
        if not self._session_initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        try:
            logger.info(f"Calling tool '{tool_name}' with params: {json.dumps(parameters, indent=2)}")
            result = await self.mcp_session.call_tool(tool_name, parameters)
            return _parse_mcp_result(result, f"No result from {tool_name}")
        except Exception as e:
            logger.error(f"Error calling {tool_name}: {e}")
            return json.dumps({"error": f"Error in {tool_name}: {str(e)}"})

    def _prepare_tool_params(self, tool_name: str, field_name: str, data_sources: List[Dict], operation: Dict, field_config: Dict) -> Dict:
        """Prepare parameters for the MCP tool call based on the operation."""
        params = {
            "field_name": field_name,
            "sources": data_sources,
            "priority_list": field_config.get("priority", [])
        }

        # Add operation-specific parameters
        params.update(operation)

        # Rename 'value' from config to 'threshold' for fuzzy_match
        if tool_name == "fuzzy_match_sources" and "value" in params:
            params["threshold"] = params.pop("value")

        if tool_name == "enrich_data_llm":
            params["enrichment_prompt"] = params.pop("query", f"Enrich data for field {field_name}")
            params["context"] = {"support_field": params.pop("support_field", None)}

        return params

    async def process_data_from_config(self, config: Dict, data_sources: List[Dict]) -> Dict:
        """
        Processes data based on a configuration file by calling corresponding MCP tools.
        """
        if not self._session_initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        logger.info("Starting simplified data processing from config.")
        final_results = {}

        for group in config.get("groups", []):
            group_name = group.get("group_name")
            final_results[group_name] = {}
            for field_config in group.get("field_list", []):
                field_name = field_config.get("field_name")
                logger.info(f"Processing field: '{field_name}'")
                
                operation_results = []
                for operation in field_config.get("operation", []):
                    op_type = operation.get("type", "").lower()
                    tool_name = self.operation_to_tool_map.get(op_type)

                    if not tool_name:
                        logger.warning(f"No MCP tool mapping for operation type: '{op_type}' in field '{field_name}'")
                        continue

                    params = self._prepare_tool_params(tool_name, field_name, data_sources, operation, field_config)
                    
                    result_str = await self.call_mcp_tool(tool_name, params)
                    parsed_result = _safe_parse_json(result_str)
                    
                    operation_results.append({
                        "operation": op_type,
                        "tool_name": tool_name,
                        "result": parsed_result
                    })

                final_results[group_name][field_name] = operation_results
                logger.info(f"Finished processing for field '{field_name}'.")

        return final_results


class SimpleDataProcessingClientManager:
    """Context manager wrapper for proper resource management."""
    
    def __init__(self, server_url: str = "http://127.0.0.1:8080"):
        self.server_url = server_url
        self.client = None
    
    async def __aenter__(self):
        self.client = SimpleDataProcessingClient()
        await self.client.initialize(self.server_url)
        return self.client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.cleanup()
        self.client = None


async def main(config: Dict, extracted_data: List[Dict]):
    """
    Main function to run the simplified data processing client.
    """
    async with SimpleDataProcessingClientManager() as client:
        try:
            data_sources = []
            for i, source in enumerate(extracted_data):
                data_sources.append({
                    "id": source.get("source_id", f"source_{i}"),
                    "name": source.get("doc_type", f"document_{i}"),
                    "priority": i + 1,
                    "data": source.get("data", {}),
                    "timestamp": source.get("metadata", {}).get("timestamp", ""),
                    "confidence": source.get("confidence", 0.7)
                })

            logger.info(f"Processing {len(data_sources)} data sources with simplified config-driven processing.")
            
            result = await client.process_data_from_config(
                config=config,
                data_sources=data_sources
            )
            
            logger.info("--- SIMPLIFIED PROCESSING COMPLETED ---")
            logger.info(json.dumps(result, indent=2))
            
            return result
            
        except Exception as e:
            logger.error(f"An error occurred during simplified processing: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}


if __name__ == "__main__":
    async def run_example():
        try:
            with open("config.json", "r") as f:
                config_data = json.load(f)
            with open("extracted_data.json", "r") as f:
                extracted_data_json = json.load(f)
            
            result = await main(config_data, extracted_data_json)
            print("\n--- Final Processing Result ---")
            print(json.dumps(result, indent=2))

        except FileNotFoundError:
            logger.error("Could not find 'config.json' or 'extracted_data.json'. Please create them for the example.")
        except Exception as e:
            logger.error(f"An error occurred during the example run: {e}")

    # To run the example, ensure you have a running MCP server and required JSON files,
    # then uncomment the following line:
    # asyncio.run(run_example())
    pass

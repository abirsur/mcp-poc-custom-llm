
#!/usr/bin/env python3
"""
MCP Server for Data Post-Processing Pipeline
Handles fuzzy matching, data enrichment, priority processing, and RAG operations
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import difflib
from fuzzywuzzy import fuzz, process
import numpy as np
from datetime import datetime

# MCP imports
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from openai import AsyncAzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Represents a data source with its metadata"""
    id: str
    name: str
    priority: int
    data: Dict[str, Any]
    confidence: float = 0.0
    timestamp: str = ""

@dataclass
class ProcessedResult:
    """Represents processed data with source tracking"""
    value: Any
    source: DataSource
    confidence: float
    processing_method: str
    dependencies: List[str] = None

class ProcessingStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class DataPostProcessor:
    """Core data processing engine"""
    
    def __init__(self):
        self.processed_cache = {}
        self.dependency_graph = {}
        
    def fuzzy_match(self,
                    target_value: str,
                    sources: List[DataSource],
                    field_name: str,
                    threshold: float = 0.8) -> List[Tuple[DataSource, float]]:
        """
        Perform fuzzy matching across multiple data sources.
        If target_value is empty, it finds the best candidate among sources.
        """
        
        source_values = [str(s.data[field_name]) for s in sources if field_name in s.data]
        if not source_values:
            return []

        # If no target value is provided, find the most common or representative value to act as the target.
        if not target_value:
            # Find the most central value by comparing all values against each other
            if len(source_values) > 1:
                scores = {val: 0 for val in source_values}
                for i in range(len(source_values)):
                    for j in range(i + 1, len(source_values)):
                        score = fuzz.token_set_ratio(source_values[i], source_values[j])
                        scores[source_values[i]] += score
                        scores[source_values[j]] += score
                target_value = max(scores, key=scores.get)
            else:
                target_value = source_values[0]

        matches = []
        for source in sources:
            if field_name in source.data:
                source_value = str(source.data[field_name])
                
                # Calculate multiple similarity scores
                ratio = fuzz.ratio(target_value.lower(), source_value.lower()) / 100
                partial_ratio = fuzz.partial_ratio(target_value.lower(), source_value.lower()) / 100
                token_sort = fuzz.token_sort_ratio(target_value.lower(), source_value.lower()) / 100
                token_set = fuzz.token_set_ratio(target_value.lower(), source_value.lower()) / 100
                
                # Weighted average of similarity scores
                similarity = (ratio * 0.3 + partial_ratio * 0.2 + token_sort * 0.25 + token_set * 0.25)
                
                if similarity >= threshold:
                    source.confidence = similarity
                    matches.append((source, similarity))
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def select_best_match(self, 
                         matches: List[Tuple[DataSource, float]], 
                         priority_list: List[str] = None) -> Optional[DataSource]:
        """
        Select the best match based on similarity and priority
        """
        if not matches:
            return None
        
        if priority_list:
            # Apply priority weighting
            weighted_matches = []
            for source, similarity in matches:
                priority_weight = 1.0
                if source.name in priority_list:
                    priority_index = priority_list.index(source.name)
                    priority_weight = 1.0 + (len(priority_list) - priority_index) * 0.1
                
                weighted_score = similarity * priority_weight
                weighted_matches.append((source, weighted_score))
            
            weighted_matches.sort(key=lambda x: x[1], reverse=True)
            return weighted_matches[0][0]
        
        return matches[0][0]
    
    def identify_comprehensive_values(self, 
                                    sources: List[DataSource], 
                                    field_name: str) -> List[ProcessedResult]:
        """
        Identify comprehensive values from multiple sources
        """
        comprehensive_results = []
        
        for source in sources:
            if field_name in source.data:
                value = source.data[field_name]
                
                # Calculate comprehensiveness score based on data completeness
                comprehensiveness = self._calculate_comprehensiveness(value, source.data)
                
                result = ProcessedResult(
                    value=value,
                    source=source,
                    confidence=comprehensiveness,
                    processing_method="comprehensive_analysis"
                )
                comprehensive_results.append(result)
        
        # Sort by comprehensiveness score
        comprehensive_results.sort(key=lambda x: x.confidence, reverse=True)
        return comprehensive_results
    
    def _calculate_comprehensiveness(self, value: Any, full_data: Dict) -> float:
        """
        Calculate comprehensiveness score based on data completeness
        """
        if isinstance(value, str):
            # For strings, consider length and richness
            base_score = min(len(value) / 100, 1.0)  # Normalize to 0-1
            
            # Bonus for structured information
            if any(char in value for char in [',', ';', '|', '\n']):
                base_score += 0.1
            
        elif isinstance(value, (list, dict)):
            # For collections, consider size and depth
            if isinstance(value, list):
                base_score = min(len(value) / 10, 1.0)
            else:
                base_score = min(len(value.keys()) / 10, 1.0)
        else:
            base_score = 0.5  # Default for other types
        
        # Consider overall data source completeness
        completeness_bonus = len([k for k, v in full_data.items() if v]) / max(len(full_data), 1) * 0.2
        
        return min(base_score + completeness_bonus, 1.0)
    
    def check_dependencies(self, 
                          field_name: str, 
                          processed_data: Dict[str, ProcessedResult]) -> List[str]:
        """
        Check field dependencies and return missing dependencies
        """
        if field_name not in self.dependency_graph:
            return []
        
        dependencies = self.dependency_graph[field_name]
        missing_deps = []
        
        for dep in dependencies:
            if dep not in processed_data:
                missing_deps.append(dep)
        
        return missing_deps
    
    def set_dependency(self, field_name: str, dependencies: List[str]):
        """
        Set dependencies for a field
        """
        self.dependency_graph[field_name] = dependencies

# Initialize the MCP server
server = Server("data-postprocessor")
processor = DataPostProcessor()

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """
    List available tools for data post-processing
    """
    return [
        Tool(
            name="fuzzy_match_sources",
            description="Perform fuzzy matching across multiple data sources",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_value": {"type": "string"},
                    "sources": {"type": "array"},
                    "field_name": {"type": "string"},
                    "threshold": {"type": "number"},
                    "priority_list": {"type": "array"}
                },
                "required": ["sources", "field_name"]
            }
        ),
        Tool(
            name="select_best_match",
            description="Select the best matching value based on similarity and priority",
            inputSchema={
                "type": "object",
                "properties": {
                    "matches": {"type": "array"},
                    "priority_list": {"type": "array"}
                },
                "required": ["matches"]
            }
        ),
        Tool(
            name="identify_comprehensive_values",
            description="Identify comprehensive values with source tracking",
            inputSchema={
                "type": "object",
                "properties": {
                    "sources": {"type": "array"},
                    "field_name": {"type": "string"},
                    "priority_list": {"type": "array"}
                },
                "required": ["sources", "field_name"]
            }
        ),
        Tool(
            name="enrich_data_llm",
            description="Enrich data using LLM with source preservation",
            inputSchema={
                "type": "object",
                "properties": {
                    "field_name": {"type": "string"},
                    "sources": {"type": "array"},
                    "priority_list": {"type": "array"},
                    "enrichment_prompt": {"type": "string"},
                    "context": {"type": "object"}
                },
                "required": ["field_name", "sources", "enrichment_prompt"]
            }
        ),
        Tool(
            name="process_with_priority",
            description="Process data according to priority list",
            inputSchema={
                "type": "object",
                "properties": {
                    "sources": {"type": "array"},
                    "priority_list": {"type": "array"},
                    "fields": {"type": "array"}
                },
                "required": ["sources", "priority_list", "fields"]
            }
        ),
        Tool(
            name="set_field_dependencies",
            description="Set dependencies between fields",
            inputSchema={
                "type": "object",
                "properties": {
                    "field_name": {"type": "string"},
                    "dependencies": {"type": "array"}
                },
                "required": ["field_name", "dependencies"]
            }
        ),
        Tool(
            name="rag_query",
            description="Perform RAG-based query for data enrichment",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "raw_data": {"type": "object"},
                    "context_sources": {"type": "array"}
                },
                "required": ["query", "raw_data"]
            }
        ),
        Tool(
            name="generate_summary",
            description="Generate comprehensive summary using RAG and LLM",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {"type": "object"},
                    "summary_type": {"type": "string"},
                    "include_sources": {"type": "boolean", "default": True}
                },
                "required": ["data"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    Handle tool calls for data processing operations
    """
    try:
        if name == "fuzzy_match_sources":
            target_value = arguments.get("target_value", "")
            sources_data = arguments["sources"]
            field_name = arguments["field_name"]
            threshold = arguments.get("threshold", 0.8)
            
            # Convert dict sources to DataSource objects
            sources = [DataSource(**src) for src in sources_data]
            
            matches = processor.fuzzy_match(target_value, sources, field_name, threshold)
            
            # Convert results to serializable format
            result = []
            for source, similarity in matches:
                result.append({
                    "source": asdict(source),
                    "similarity": similarity
                })
            
            return [TextContent(
                type="text",
                text=json.dumps({"matches": result}, indent=2)
            )]
        
        elif name == "select_best_match":
            matches_data = arguments["matches"]
            priority_list = arguments.get("priority_list")
            
            # Reconstruct matches from serialized data
            matches = []
            for match_data in matches_data:
                source = DataSource(**match_data["source"])
                similarity = match_data["similarity"]
                matches.append((source, similarity))
            
            best_match = processor.select_best_match(matches, priority_list)
            
            result = asdict(best_match) if best_match else None
            
            return [TextContent(
                type="text",
                text=json.dumps({"best_match": result}, indent=2)
            )]
        
        elif name == "identify_comprehensive_values":
            sources_data = arguments["sources"]
            field_name = arguments["field_name"]
            
            sources = [DataSource(**src) for src in sources_data]
            comprehensive_values = processor.identify_comprehensive_values(sources, field_name)
            
            result = [asdict(cv) for cv in comprehensive_values]
            
            return [TextContent(
                type="text",
                text=json.dumps({"comprehensive_values": result}, indent=2)
            )]
        
        elif name == "enrich_data_llm":
            field_name = arguments["field_name"]
            sources_data = arguments["sources"]
            enrichment_prompt = arguments["enrichment_prompt"]
            context_data = arguments.get("context", {})
            priority_list = arguments.get("priority_list", [])

            sources = [DataSource(**src) for src in sources_data]

            # Find the best current value for the field to be enriched.
            matches = processor.fuzzy_match("", sources, field_name, threshold=0.1) # low threshold to get any value
            best_source = processor.select_best_match(matches, priority_list)
            
            data_to_enrich = {}
            if best_source and field_name in best_source.data:
                data_to_enrich[field_name] = best_source.data[field_name]
            else:
                data_to_enrich[field_name] = None

            # Add support field from context if available
            support_field = context_data.get("support_field")
            if support_field and best_source and support_field in best_source.data:
                data_to_enrich[support_field] = best_source.data[support_field]
            
            # Integrate with LLM for enrichment
            try:
                azure_client = AsyncAzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version="2024-05-01-preview",
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                )

                llm_context_info = f"Current data for enrichment: {json.dumps(data_to_enrich)}"

                response = await azure_client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME"),
                    messages=[
                        {"role": "system", "content": "You are a data enrichment assistant."},
                        {"role": "user", "content": f"Enrich this data: {json.dumps(data_to_enrich)}. Prompt: {enrichment_prompt}. Context: {llm_context_info}"}
                    ],
                    temperature=0.2,
                    max_tokens=500
                )
                enriched_content = response.choices[0].message.content
                
                enriched_data = {
                    "original_data": data_to_enrich,
                    "enrichment_prompt": enrichment_prompt,
                    "enriched_content": enriched_content,
                    "enriched": True,
                    "timestamp": datetime.now().isoformat(),
                    "status": "enriched_via_llm"
                }

            except Exception as llm_error:
                logger.error(f"LLM enrichment failed: {llm_error}")
                enriched_data = {"error": str(llm_error)}
            
            return [TextContent(
                type="text",
                text=json.dumps({"enriched_data": enriched_data}, indent=2)
            )]
        
        elif name == "process_with_priority":
            sources_data = arguments["sources"]
            priority_list = arguments["priority_list"]
            fields = arguments["fields"]
            
            sources = [DataSource(**src) for src in sources_data]
            processed_results = {}
            
            # Sort sources by priority
            priority_map = {name: idx for idx, name in enumerate(priority_list)}
            sources.sort(key=lambda s: priority_map.get(s.name, float('inf')))
            
            for field in fields:
                # Check dependencies
                missing_deps = processor.check_dependencies(field, processed_results)
                
                if missing_deps:
                    # Process dependencies first
                    for dep in missing_deps:
                        if dep in fields:
                            dep_matches = processor.fuzzy_match("", sources, dep)
                            if dep_matches:
                                best_match = processor.select_best_match(dep_matches, priority_list)
                                if best_match:
                                    processed_results[dep] = ProcessedResult(
                                        value=best_match.data.get(dep),
                                        source=best_match,
                                        confidence=best_match.confidence,
                                        processing_method="dependency_resolution"
                                    )
                
                # Process the main field
                matches = processor.fuzzy_match("", sources, field)
                if matches:
                    best_match = processor.select_best_match(matches, priority_list)
                    if best_match:
                        processed_results[field] = ProcessedResult(
                            value=best_match.data.get(field),
                            source=best_match,
                            confidence=best_match.confidence,
                            processing_method="priority_based"
                        )
            
            result = {k: asdict(v) for k, v in processed_results.items()}
            
            return [TextContent(
                type="text",
                text=json.dumps({"processed_results": result}, indent=2)
            )]
        
        elif name == "set_field_dependencies":
            field_name = arguments["field_name"]
            dependencies = arguments["dependencies"]
            
            processor.set_dependency(field_name, dependencies)
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "message": f"Dependencies set for field '{field_name}'",
                    "dependencies": dependencies
                }, indent=2)
            )]
        
        elif name == "rag_query":
            query = arguments["query"]
            raw_data = arguments["raw_data"]
            context_sources = arguments.get("context_sources", [])
            
            # Placeholder for RAG implementation
            rag_result = {
                "query": query,
                "raw_data_summary": f"Processed {len(raw_data)} data points",
                "context_sources": context_sources,
                "rag_response": f"RAG-based response for: {query}",
                "confidence": 0.85,
                "timestamp": datetime.now().isoformat()
            }
            
            return [TextContent(
                type="text",
                text=json.dumps({"rag_result": rag_result}, indent=2)
            )]
        
        elif name == "generate_summary":
            data = arguments["data"]
            summary_type = arguments.get("summary_type", "comprehensive")
            include_sources = arguments.get("include_sources", True)
            
            # Generate summary based on processed data
            summary = {
                "summary_type": summary_type,
                "data_points": len(data),
                "summary": f"Comprehensive {summary_type} summary of processed data",
                "include_sources": include_sources,
                "generated_at": datetime.now().isoformat()
            }
            
            if include_sources:
                summary["sources"] = list(set([
                    item.get("source", {}).get("name", "unknown")
                    for item in data.values()
                    if isinstance(item, dict) and "source" in item
                ]))
            
            return [TextContent(
                type="text",
                text=json.dumps({"summary": summary}, indent=2)
            )]
        
        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"})
            )]
    
    except Exception as e:
        logger.error(f"Error in tool {name}: {str(e)}")
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]

async def get_app():
    # Import here to avoid issues with event loops
    from mcp.server.stdio import stdio_server
    
    app = stdio_server(server, InitializationOptions(
        server_name="data-postprocessor",
        server_version="1.0.0",
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
    ))
    return app

async def main():
    # This part is for development and testing.
    # In production, you would use a proper ASGI server like gunicorn or uvicorn.
    import uvicorn
    app = await get_app()
    
    config = uvicorn.Config(app, host="localhost", port=8080, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())

----
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

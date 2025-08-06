#!/usr/bin/env python3
"""
MCP Client for Data Post-Processing
Direct MCP tool calls based on configuration logic
Covers all available tools with fallback scenarios
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Set
import os
from dotenv import load_dotenv

# Azure OpenAI imports
from openai import AsyncAzureOpenAI

# MCP client imports
from mcp import ClientSession, HttpPostTransport, ServerId
from mcp.client.http import http_client

load_dotenv()

# Configure logging
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
    """Safely parse JSON from text response."""
    if default is None:
        default = {}
    try:
        # Try to extract JSON from text that might contain other content
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(text)
    except:
        return default


class DataProcessingClient:
    """
    Main client for data processing pipeline using direct MCP tool calls
    """
    
    def __init__(self, azure_config: Dict[str, str]):
        self.azure_config = azure_config
        self.mcp_session = None
        self.mcp_client = None
        self._session_initialized = False
        
        # Available MCP tools
        self.available_tools = {
            "fuzzy_match_sources",
            "select_best_match", 
            "identify_comprehensive_values",
            "enrich_data_llm",
            "process_with_priority",
            "set_field_dependencies",
            "rag_query",
            "generate_summary"
        }
        
        # Track field processing state
        self.processed_fields = {}
        self.field_dependencies = {}
    
    async def initialize(self, server_url: str = "http://127.0.0.1:8080"):
        """Initialize the client with MCP server"""
        try:
            # Initialize MCP session with proper context management
            server_id = ServerId(
                url=server_url,
                name="data-postprocessor"
            )
            transport = HttpPostTransport(server_id)
            
            # Create the client context manager
            self.mcp_client = http_client(transport)
            # Enter the context properly
            self.mcp_session = await self.mcp_client.__aenter__()
            self._session_initialized = True
            
            logger.info("Data Processing Client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize client: {e}")
            await self.cleanup()
            raise

    async def _call_mcp_tool(self, tool_name: str, parameters: Dict) -> str:
        """Safe wrapper for MCP tool calls"""
        try:
            if tool_name not in self.available_tools:
                return f"Tool {tool_name} not available"
            
            result = await self.mcp_session.call_tool(tool_name, parameters)
            return _parse_mcp_result(result, f"No result from {tool_name}")
            
        except Exception as e:
            logger.error(f"Error calling {tool_name}: {e}")
            return f"Error in {tool_name}: {str(e)}"

    async def _set_field_dependencies(self, config: Dict):
        """Set up field dependencies from config"""
        logger.info("Setting up field dependencies...")
        
        # Extract dependencies from config
        if "fields" in config:
            for field_name, field_config in config["fields"].items():
                if "dependencies" in field_config:
                    dependencies = field_config["dependencies"]
                    if dependencies:
                        result = await self._call_mcp_tool("set_field_dependencies", {
                            "field_name": field_name,
                            "dependencies": dependencies
                        })
                        self.field_dependencies[field_name] = dependencies
                        logger.info(f"Set dependencies for {field_name}: {dependencies}")

    async def _fuzzy_match_field(self, field_name: str, data_sources: List[Dict], 
                                field_config: Dict) -> Dict:
        """Perform fuzzy matching for a specific field"""
        logger.info(f"Performing fuzzy matching for field: {field_name}")
        
        # Get target value from config or detect automatically
        target_value = field_config.get("target_value", "")
        threshold = field_config.get("fuzzy_threshold", 0.8)
        
        result = await self._call_mcp_tool("fuzzy_match_sources", {
            "target_value": target_value,
            "sources": data_sources,
            "field_name": field_name,
            "threshold": threshold
        })
        
        # Parse result to extract matches
        matches_data = _safe_parse_json(result, {"matches": []})
        return matches_data

    async def _select_best_match(self, matches: List[Dict], priority_list: List[str] = None) -> Dict:
        """Select the best match from fuzzy matching results"""
        if not matches:
            return {}
        
        logger.info("Selecting best match from fuzzy results")
        
        params = {"matches": matches}
        if priority_list:
            params["priority_list"] = priority_list
            
        result = await self._call_mcp_tool("select_best_match", params)
        
        return _safe_parse_json(result, {"selected_match": {}})

    async def _identify_comprehensive_values(self, field_name: str, data_sources: List[Dict]) -> Dict:
        """Identify comprehensive values for a field"""
        logger.info(f"Identifying comprehensive values for: {field_name}")
        
        result = await self._call_mcp_tool("identify_comprehensive_values", {
            "sources": data_sources,
            "field_name": field_name
        })
        
        return _safe_parse_json(result, {"comprehensive_values": {}})

    async def _enrich_data(self, data: Dict, field_name: str, config: Dict) -> Dict:
        """Enrich data using LLM"""
        logger.info(f"Enriching data for field: {field_name}")
        
        # Create enrichment prompt based on field type and config
        field_config = config.get("fields", {}).get(field_name, {})
        enrichment_type = field_config.get("enrichment_type", "standard")
        
        enrichment_prompts = {
            "standard": f"Enhance and standardize the {field_name} field data",
            "normalize": f"Normalize and clean the {field_name} field data",
            "extract": f"Extract relevant information for {field_name} from the provided data",
            "validate": f"Validate and correct the {field_name} field data",
            "complete": f"Complete missing information for {field_name} field"
        }
        
        prompt = enrichment_prompts.get(enrichment_type, enrichment_prompts["standard"])
        context = field_config.get("context", "")
        
        params = {
            "data": data,
            "enrichment_prompt": prompt
        }
        if context:
            params["context"] = context
            
        result = await self._call_mcp_tool("enrich_data_llm", params)
        
        return _safe_parse_json(result, {"enriched_data": data})

    async def _process_with_priority(self, data_sources: List[Dict], priority_list: List[str], 
                                   fields: List[str]) -> Dict:
        """Process data with priority ordering"""
        logger.info(f"Processing with priority for fields: {fields}")
        
        result = await self._call_mcp_tool("process_with_priority", {
            "sources": data_sources,
            "priority_list": priority_list,
            "fields": fields
        })
        
        return _safe_parse_json(result, {"priority_results": {}})

    async def _rag_query(self, query: str, raw_data: Dict, context_sources: List[str] = None) -> Dict:
        """Perform RAG-based query"""
        logger.info(f"Performing RAG query: {query[:50]}...")
        
        params = {
            "query": query,
            "raw_data": raw_data
        }
        if context_sources:
            params["context_sources"] = context_sources
            
        result = await self._call_mcp_tool("rag_query", params)
        
        return _safe_parse_json(result, {"rag_results": {}})

    async def _generate_summary(self, data: Dict, summary_type: str = "comprehensive", 
                              include_sources: bool = True) -> Dict:
        """Generate summary of processed data"""
        logger.info(f"Generating {summary_type} summary")
        
        params = {"data": data}
        if summary_type != "comprehensive":
            params["summary_type"] = summary_type
        if include_sources != True:
            params["include_sources"] = include_sources
            
        result = await self._call_mcp_tool("generate_summary", params)
        
        return _safe_parse_json(result, {"summary": {}})

    async def _process_field_by_strategy(self, field_name: str, field_config: Dict, 
                                       data_sources: List[Dict], priority_list: List[str],
                                       raw_data: Dict) -> Dict:
        """Process a single field based on its configured strategy"""
        strategy = field_config.get("processing_strategy", "comprehensive")
        field_result = {}
        
        logger.info(f"Processing field '{field_name}' with strategy '{strategy}'")
        
        try:
            if strategy == "fuzzy_match":
                # Fuzzy matching strategy
                matches_data = await self._fuzzy_match_field(field_name, data_sources, field_config)
                if matches_data.get("matches"):
                    best_match = await self._select_best_match(matches_data["matches"], priority_list)
                    field_result = best_match.get("selected_match", {})
                
            elif strategy == "comprehensive":
                # Comprehensive value identification
                comp_values = await self._identify_comprehensive_values(field_name, data_sources)
                field_result = comp_values.get("comprehensive_values", {})
                
            elif strategy == "priority":
                # Priority-based processing
                priority_result = await self._process_with_priority(data_sources, priority_list, [field_name])
                field_result = priority_result.get("priority_results", {}).get(field_name, {})
                
            elif strategy == "rag_enhanced":
                # RAG-enhanced processing
                query = f"Find and extract the best value for {field_name}"
                rag_result = await self._rag_query(query, raw_data)
                field_result = rag_result.get("rag_results", {})
                
            else:
                # Default comprehensive processing
                comp_values = await self._identify_comprehensive_values(field_name, data_sources)
                field_result = comp_values.get("comprehensive_values", {})
            
            # Apply enrichment if specified
            if field_config.get("enrich", False):
                enriched = await self._enrich_data(field_result, field_name, {"fields": {field_name: field_config}})
                field_result = enriched.get("enriched_data", field_result)
            
            self.processed_fields[field_name] = field_result
            return field_result
            
        except Exception as e:
            logger.error(f"Error processing field {field_name}: {e}")
            return {"error": str(e), "field": field_name}

    async def _handle_missing_config_scenarios(self, data_sources: List[Dict], 
                                             priority_list: List[str], raw_data: Dict) -> Dict:
        """Handle scenarios not covered in current configuration"""
        additional_results = {}
        
        logger.info("Handling additional processing scenarios...")
        
        # Scenario 1: Auto-detect important fields using RAG
        try:
            auto_detect_query = "Identify the most important fields and their values from the provided data sources"
            rag_result = await self._rag_query(auto_detect_query, raw_data)
            additional_results["auto_detected_fields"] = rag_result.get("rag_results", {})
        except Exception as e:
            logger.error(f"Auto-detection failed: {e}")
        
        # Scenario 2: Cross-field fuzzy matching for related data
        try:
            # Find potential field relationships
            all_fields = set()
            for source in data_sources:
                if isinstance(source.get("data"), dict):
                    all_fields.update(source["data"].keys())
            
            # Perform cross-field analysis for common business fields
            common_fields = ["name", "id", "number", "date", "amount", "status", "type"]
            detected_fields = [f for f in all_fields if any(cf in f.lower() for cf in common_fields)]
            
            if detected_fields:
                for field in detected_fields[:3]:  # Limit to 3 fields to avoid overprocessing
                    matches_data = await self._fuzzy_match_field(field, data_sources, {"fuzzy_threshold": 0.7})
                    if matches_data.get("matches"):
                        best_match = await self._select_best_match(matches_data["matches"], priority_list)
                        additional_results[f"cross_matched_{field}"] = best_match.get("selected_match", {})
        except Exception as e:
            logger.error(f"Cross-field matching failed: {e}")
        
        # Scenario 3: Comprehensive analysis of all sources
        try:
            if data_sources:
                # Get comprehensive values for first available field
                sample_fields = []
                for source in data_sources:
                    if isinstance(source.get("data"), dict):
                        sample_fields.extend(list(source["data"].keys())[:2])
                        break
                
                for field in sample_fields:
                    comp_values = await self._identify_comprehensive_values(field, data_sources)
                    additional_results[f"comprehensive_{field}"] = comp_values.get("comprehensive_values", {})
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
        
        # Scenario 4: Data enrichment on merged data
        try:
            # Merge all data for enrichment
            merged_data = {}
            for source in data_sources:
                if isinstance(source.get("data"), dict):
                    merged_data.update(source["data"])
            
            if merged_data:
                enriched = await self._enrich_data(merged_data, "merged_data", {})
                additional_results["enriched_merged_data"] = enriched.get("enriched_data", {})
        except Exception as e:
            logger.error(f"Data enrichment failed: {e}")
        
        return additional_results

    async def process_data(self, 
                          config: Dict,
                          data_sources: List[Dict],
                          raw_data: Dict = None,
                          priority_list: List[str] = None) -> Dict:
        """
        Main method to process data using direct MCP tool calls based on configuration
        """
        if not self._session_initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        
        logger.info("Starting data processing with direct MCP tool calls")
        
        # Extract priority list from config if not provided
        if not priority_list:
            priority_list = config.get("global_settings", {}).get("priority_list", [])
        
        # Ensure we have raw_data
        if not raw_data:
            raw_data = {"sources": data_sources, "config": config}
        
        try:
            # Step 1: Set up field dependencies
            await self._set_field_dependencies(config)
            
            # Step 2: Process fields according to configuration
            field_results = {}
            
            # Get fields from config
            fields_config = config.get("fields", {})
            
            # Sort fields by dependencies (process dependencies first)
            def get_dependency_order(fields):
                ordered = []
                remaining = set(fields.keys())
                
                while remaining:
                    # Find fields with no unprocessed dependencies
                    ready = []
                    for field in remaining:
                        deps = self.field_dependencies.get(field, [])
                        if not deps or all(dep in ordered for dep in deps):
                            ready.append(field)
                    
                    if not ready:
                        # Break circular dependencies
                        ready = [next(iter(remaining))]
                    
                    for field in ready:
                        ordered.append(field)
                        remaining.remove(field)
                
                return ordered
            
            ordered_fields = get_dependency_order(fields_config)
            
            # Process each field
            for field_name in ordered_fields:
                field_config = fields_config[field_name]
                field_result = await self._process_field_by_strategy(
                    field_name, field_config, data_sources, priority_list, raw_data
                )
                field_results[field_name] = field_result
                
                # Add delay between field processing to avoid overwhelming the server
                await asyncio.sleep(0.1)
            
            # Step 3: Handle additional scenarios not in config
            additional_results = await self._handle_missing_config_scenarios(
                data_sources, priority_list, raw_data
            )
            
            # Step 4: Generate final summary
            all_results = {
                "processed_fields": field_results,
                "additional_analysis": additional_results,
                "metadata": {
                    "total_fields_processed": len(field_results),
                    "total_sources": len(data_sources),
                    "priority_list": priority_list,
                    "processing_timestamp": asyncio.get_event_loop().time()
                }
            }
            
            # Generate comprehensive summary
            summary_result = await self._generate_summary(all_results, "comprehensive", True)
            
            final_result = {
                "status": "completed",
                "processed_data": field_results,
                "additional_insights": additional_results,
                "summary": summary_result.get("summary", {}),
                "metadata": all_results["metadata"]
            }
            
            logger.info(f"Data processing completed successfully. Processed {len(field_results)} fields.")
            return final_result
            
        except asyncio.TimeoutError:
            logger.error("Data processing timed out")
            return {
                "status": "timeout",
                "error": "Processing timed out",
                "partial_results": self.processed_fields
            }
        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            return {
                "status": "error",
                "error": str(e),
                "partial_results": self.processed_fields
            }

    async def cleanup(self):
        """Cleanup resources properly"""
        try:
            if self.mcp_client and self._session_initialized:
                await self.mcp_client.__aexit__(None, None, None)
                self._session_initialized = False
                logger.info("MCP session cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            self.mcp_session = None
            self.mcp_client = None


class DataProcessingClientManager:
    """Context manager wrapper for proper resource management"""
    
    def __init__(self, azure_config: Dict[str, str], server_url: str = "http://127.0.0.1:8080"):
        self.azure_config = azure_config
        self.server_url = server_url
        self.client = None
    
    async def __aenter__(self):
        self.client = DataProcessingClient(self.azure_config)
        await self.client.initialize(self.server_url)
        return self.client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.cleanup()
        self.client = None


# Example usage with comprehensive coverage
async def main(config: Dict, extracted_data: List[Dict]):
    """
    Main function with direct MCP tool calls and comprehensive coverage
    """
    # Azure OpenAI configuration
    azure_config = {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": "2024-05-01-preview"
    }
    
    if not all([azure_config["api_key"], azure_config["endpoint"]]):
        logger.error("Azure OpenAI environment variables not set.")
        logger.error("Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT.")
        return {"error": "Azure OpenAI configuration missing"}

    # Use context manager for proper resource management
    async with DataProcessingClientManager(azure_config) as client:
        try:
            # Transform data sources to required format
            data_sources = []
            for i, source in enumerate(extracted_data):
                data_sources.append({
                    "id": source.get("source_id", f"source_{i}"),
                    "name": source.get("doc_type", f"document_{i}"),
                    "priority": i + 1,
                    "data": source.get("data", {}),
                    "timestamp": source.get("metadata", {}).get("timestamp", ""),
                    "confidence": source.get("confidence", 0.0)
                })

            logger.info(f"Processing {len(data_sources)} data sources with direct MCP calls")
            
            # Process data using direct MCP tool calls
            result = await client.process_data(
                config=config,
                data_sources=data_sources,
                raw_data={"extracted": extracted_data, "config": config}
            )
            
            logger.info("--- PROCESSING COMPLETED ---")
            logger.info(f"Status: {result.get('status', 'unknown')}")
            logger.info(f"Fields processed: {len(result.get('processed_data', {}))}")
            logger.info(f"Additional insights: {len(result.get('additional_insights', {}))}")
            
            return result
            
        except Exception as e:
            logger.error(f"An error occurred during processing: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}


async def main_direct(config: Dict, extracted_data: List[Dict]):
    """Alternative direct usage without context manager"""
    azure_config = {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": "2024-05-01-preview"
    }
    
    if not all([azure_config["api_key"], azure_config["endpoint"]]):
        return {"error": "Azure OpenAI configuration missing"}

    client = DataProcessingClient(azure_config)
    
    try:
        await client.initialize()
        
        data_sources = []
        for i, source in enumerate(extracted_data):
            data_sources.append({
                "id": source.get("source_id", f"source_{i}"),
                "name": source.get("doc_type", f"document_{i}"),
                "priority": i + 1,
                "data": source.get("data", {}),
                "timestamp": source.get("metadata", {}).get("timestamp", ""),
                "confidence": source.get("confidence", 0.0)
            })
        
        result = await client.process_data(
            config=config,
            data_sources=data_sources,
            raw_data={"extracted": extracted_data, "config": config}
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return {"error": str(e), "status": "failed"}
    
    finally:
        await client.cleanup()


if __name__ == "__main__":
    # Example usage:
    # with open("config.json", "r") as f:
    #     config_data = json.load(f)
    # with open("extracted_data.json", "r") as f:
    #     extracted_data_json = json.load(f)
    # result = asyncio.run(main(config_data, extracted_data_json))
    pass

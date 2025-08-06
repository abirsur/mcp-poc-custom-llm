#!/usr/bin/env python3
"""
Enhanced MCP Client for Data Post-Processing
Complete coverage of all MCP tools with configuration-driven processing
Handles regex matching, validation, and advanced processing scenarios
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Union
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


class EnhancedDataProcessingClient:
    """
    Enhanced client for data processing pipeline with complete MCP tool coverage
    """
    
    def __init__(self, azure_config: Dict[str, str]):
        self.azure_config = azure_config
        self.mcp_session = None
        self.mcp_client = None
        self._session_initialized = False
        
        # All available MCP tools
        self.available_tools = {
            "fuzzy_match_sources",
            "select_best_match", 
            "identify_comprehensive_values",
            "enrich_data_llm",
            "process_with_priority",
            "set_field_dependencies",
            "rag_query",
            "generate_summary",
            "validate_field_data",
            "normalize_data",
            "cross_reference_fields",
            "extract_patterns",
            "merge_data_sources",
            "calculate_confidence_scores",
            "detect_anomalies"
        }
        
        # Track field processing state
        self.processed_fields = {}
        self.field_dependencies = {}
        self.validation_results = {}
    
    async def initialize(self, server_url: str = "http://127.0.0.1:8080"):
        """Initialize the client with MCP server"""
        try:
            server_id = ServerId(
                url=server_url,
                name="data-postprocessor"
            )
            transport = HttpPostTransport(server_id)
            
            self.mcp_client = http_client(transport)
            self.mcp_session = await self.mcp_client.__aenter__()
            self._session_initialized = True
            
            logger.info("Enhanced Data Processing Client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize client: {e}")
            await self.cleanup()
            raise

    async def _call_mcp_tool(self, tool_name: str, parameters: Dict) -> str:
        """Safe wrapper for MCP tool calls"""
        try:
            if tool_name not in self.available_tools:
                logger.warning(f"Tool {tool_name} not in available tools, attempting call anyway")
            
            result = await self.mcp_session.call_tool(tool_name, parameters)
            return _parse_mcp_result(result, f"No result from {tool_name}")
            
        except Exception as e:
            logger.error(f"Error calling {tool_name}: {e}")
            return f"Error in {tool_name}: {str(e)}"

    # === REGEX AND VALIDATION METHODS ===
    
    async def _regex_match_field(self, field_name: str, data_sources: List[Dict], 
                                pattern: str, extract_mode: bool = False) -> Dict:
        """Perform regex matching for field validation/extraction"""
        logger.info(f"Performing regex matching for field: {field_name} with pattern: {pattern}")
        
        # Local regex processing with MCP tool backup
        results = []
        
        for source in data_sources:
            source_data = source.get("data", {})
            field_value = source_data.get(field_name, "")
            
            if isinstance(field_value, str):
                if extract_mode:
                    # Extract all matches
                    matches = re.findall(pattern, field_value)
                    if matches:
                        results.append({
                            "source_id": source.get("id", "unknown"),
                            "matches": matches,
                            "original_value": field_value,
                            "confidence": 0.9
                        })
                else:
                    # Validate match
                    if re.match(pattern, field_value):
                        results.append({
                            "source_id": source.get("id", "unknown"),
                            "value": field_value,
                            "valid": True,
                            "confidence": 1.0
                        })
        
        # Use MCP tool for pattern extraction if available
        try:
            mcp_result = await self._call_mcp_tool("extract_patterns", {
                "sources": data_sources,
                "field_name": field_name,
                "pattern": pattern,
                "extract_mode": extract_mode
            })
            mcp_data = _safe_parse_json(mcp_result, {})
            if mcp_data.get("pattern_matches"):
                results.extend(mcp_data["pattern_matches"])
        except Exception as e:
            logger.warning(f"MCP pattern extraction failed, using local results: {e}")
        
        return {"regex_results": results, "pattern": pattern}

    async def _validate_field_data(self, field_name: str, field_value: Any, 
                                  validation_rules: Dict) -> Dict:
        """Validate field data using various rules"""
        logger.info(f"Validating field: {field_name}")
        
        validation_result = {
            "field_name": field_name,
            "value": field_value,
            "is_valid": True,
            "validation_errors": [],
            "confidence": 1.0
        }
        
        # Local validation
        try:
            # Required field check
            if validation_rules.get("required", False) and not field_value:
                validation_result["is_valid"] = False
                validation_result["validation_errors"].append("Field is required but empty")
            
            # Type validation
            expected_type = validation_rules.get("type")
            if expected_type and field_value:
                if expected_type == "email":
                    if not re.match(r'^[\w.-]+@[\w.-]+\.\w{2,}$', str(field_value)):
                        validation_result["is_valid"] = False
                        validation_result["validation_errors"].append("Invalid email format")
                elif expected_type == "phone":
                    if not re.match(r'\d{3}-\d{3}-\d{4}', str(field_value)):
                        validation_result["is_valid"] = False
                        validation_result["validation_errors"].append("Invalid phone format")
                elif expected_type == "zip":
                    if not re.match(r'\d{5}(-\d{4})?', str(field_value)):
                        validation_result["is_valid"] = False
                        validation_result["validation_errors"].append("Invalid ZIP code format")
            
            # Length validation
            min_length = validation_rules.get("min_length")
            max_length = validation_rules.get("max_length")
            if field_value and isinstance(field_value, str):
                if min_length and len(field_value) < min_length:
                    validation_result["is_valid"] = False
                    validation_result["validation_errors"].append(f"Too short (min: {min_length})")
                if max_length and len(field_value) > max_length:
                    validation_result["is_valid"] = False
                    validation_result["validation_errors"].append(f"Too long (max: {max_length})")
        
        except Exception as e:
            validation_result["validation_errors"].append(f"Validation error: {str(e)}")
            validation_result["is_valid"] = False
        
        # Use MCP validation tool if available
        try:
            mcp_result = await self._call_mcp_tool("validate_field_data", {
                "field_name": field_name,
                "field_value": field_value,
                "validation_rules": validation_rules
            })
            mcp_data = _safe_parse_json(mcp_result, {})
            if mcp_data.get("validation_result"):
                # Merge MCP validation results
                mcp_validation = mcp_data["validation_result"]
                validation_result["mcp_validation"] = mcp_validation
                if not mcp_validation.get("is_valid", True):
                    validation_result["is_valid"] = False
                    validation_result["validation_errors"].extend(
                        mcp_validation.get("validation_errors", [])
                    )
        except Exception as e:
            logger.warning(f"MCP validation failed: {e}")
        
        # Update confidence based on validation result
        if validation_result["validation_errors"]:
            validation_result["confidence"] = max(0.1, 
                1.0 - (len(validation_result["validation_errors"]) * 0.2))
        
        return validation_result

    # === ENHANCED PROCESSING METHODS ===
    
    async def _normalize_data(self, data: Dict, field_name: str, 
                             normalization_rules: Dict) -> Dict:
        """Normalize data using specified rules"""
        logger.info(f"Normalizing data for field: {field_name}")
        
        normalized_data = data.copy()
        field_value = data.get(field_name)
        
        if field_value:
            try:
                # Apply normalization rules
                if normalization_rules.get("case") == "upper":
                    normalized_data[field_name] = str(field_value).upper()
                elif normalization_rules.get("case") == "lower":
                    normalized_data[field_name] = str(field_value).lower()
                elif normalization_rules.get("case") == "title":
                    normalized_data[field_name] = str(field_value).title()
                
                # Remove extra whitespace
                if normalization_rules.get("trim", True):
                    normalized_data[field_name] = str(normalized_data[field_name]).strip()
                
                # Phone number normalization
                if normalization_rules.get("phone_format"):
                    phone_clean = re.sub(r'[^\d]', '', str(field_value))
                    if len(phone_clean) == 10:
                        normalized_data[field_name] = f"{phone_clean[:3]}-{phone_clean[3:6]}-{phone_clean[6:]}"
                
                # State abbreviation normalization
                if normalization_rules.get("state_abbreviation"):
                    state_map = {
                        "california": "CA", "texas": "TX", "florida": "FL", "new york": "NY",
                        # Add more state mappings as needed
                    }
                    state_key = str(field_value).lower().strip()
                    if state_key in state_map:
                        normalized_data[field_name] = state_map[state_key]
            
            except Exception as e:
                logger.warning(f"Local normalization failed: {e}")
        
        # Use MCP normalization tool
        try:
            mcp_result = await self._call_mcp_tool("normalize_data", {
                "data": data,
                "field_name": field_name,
                "normalization_rules": normalization_rules
            })
            mcp_data = _safe_parse_json(mcp_result, {})
            if mcp_data.get("normalized_data"):
                normalized_data.update(mcp_data["normalized_data"])
        except Exception as e:
            logger.warning(f"MCP normalization failed: {e}")
        
        return {"normalized_data": normalized_data}

    async def _cross_reference_fields(self, data_sources: List[Dict], 
                                     field_relationships: Dict) -> Dict:
        """Cross-reference related fields across sources"""
        logger.info("Performing cross-field reference analysis")
        
        cross_ref_results = {}
        
        # Local cross-referencing logic
        for primary_field, related_fields in field_relationships.items():
            field_correlations = {}
            
            for source in data_sources:
                source_data = source.get("data", {})
                primary_value = source_data.get(primary_field)
                
                if primary_value:
                    correlations = {}
                    for related_field in related_fields:
                        related_value = source_data.get(related_field)
                        if related_value:
                            correlations[related_field] = {
                                "value": related_value,
                                "confidence": self._calculate_field_correlation(
                                    primary_value, related_value
                                )
                            }
                    
                    if correlations:
                        field_correlations[source.get("id", "unknown")] = {
                            "primary_value": primary_value,
                            "correlations": correlations
                        }
            
            cross_ref_results[primary_field] = field_correlations
        
        # Use MCP cross-reference tool
        try:
            mcp_result = await self._call_mcp_tool("cross_reference_fields", {
                "sources": data_sources,
                "field_relationships": field_relationships
            })
            mcp_data = _safe_parse_json(mcp_result, {})
            if mcp_data.get("cross_reference_results"):
                # Merge with local results
                for field, mcp_refs in mcp_data["cross_reference_results"].items():
                    if field in cross_ref_results:
                        cross_ref_results[field]["mcp_analysis"] = mcp_refs
                    else:
                        cross_ref_results[field] = {"mcp_analysis": mcp_refs}
        except Exception as e:
            logger.warning(f"MCP cross-referencing failed: {e}")
        
        return {"cross_reference_results": cross_ref_results}

    def _calculate_field_correlation(self, primary_value: Any, related_value: Any) -> float:
        """Calculate correlation score between fields"""
        if not primary_value or not related_value:
            return 0.0
        
        # Simple correlation based on common patterns
        primary_str = str(primary_value).lower()
        related_str = str(related_value).lower()
        
        # Check for common substrings
        common_words = set(primary_str.split()) & set(related_str.split())
        if common_words:
            return min(1.0, len(common_words) * 0.3)
        
        # Check for similar patterns (addresses, names, etc.)
        if any(char.isdigit() for char in primary_str) and any(char.isdigit() for char in related_str):
            return 0.6  # Both contain numbers
        
        return 0.3  # Default correlation

    async def _merge_data_sources(self, data_sources: List[Dict], 
                                 merge_strategy: str = "priority") -> Dict:
        """Merge multiple data sources using specified strategy"""
        logger.info(f"Merging {len(data_sources)} data sources with {merge_strategy} strategy")
        
        merged_data = {}
        merge_metadata = {
            "total_sources": len(data_sources),
            "merge_strategy": merge_strategy,
            "field_sources": {}
        }
        
        if merge_strategy == "priority":
            # Sort by priority (lower number = higher priority)
            sorted_sources = sorted(data_sources, key=lambda x: x.get("priority", 999))
            
            for source in sorted_sources:
                source_data = source.get("data", {})
                source_id = source.get("id", "unknown")
                
                for field, value in source_data.items():
                    if field not in merged_data and value:  # Only add if not already present
                        merged_data[field] = value
                        merge_metadata["field_sources"][field] = source_id
        
        elif merge_strategy == "confidence":
            # Merge based on confidence scores
            field_candidates = {}
            
            for source in data_sources:
                source_data = source.get("data", {})
                source_confidence = source.get("confidence", 0.5)
                source_id = source.get("id", "unknown")
                
                for field, value in source_data.items():
                    if value:
                        if field not in field_candidates:
                            field_candidates[field] = []
                        field_candidates[field].append({
                            "value": value,
                            "confidence": source_confidence,
                            "source_id": source_id
                        })
            
            # Select best candidate for each field
            for field, candidates in field_candidates.items():
                best_candidate = max(candidates, key=lambda x: x["confidence"])
                merged_data[field] = best_candidate["value"]
                merge_metadata["field_sources"][field] = best_candidate["source_id"]
        
        # Use MCP merge tool
        try:
            mcp_result = await self._call_mcp_tool("merge_data_sources", {
                "sources": data_sources,
                "merge_strategy": merge_strategy
            })
            mcp_data = _safe_parse_json(mcp_result, {})
            if mcp_data.get("merged_data"):
                mcp_merged = mcp_data["merged_data"]
                # Prefer MCP results but keep local as backup
                for field, value in mcp_merged.items():
                    if value and (field not in merged_data or not merged_data[field]):
                        merged_data[field] = value
                        merge_metadata["field_sources"][field] = "mcp_merge"
        except Exception as e:
            logger.warning(f"MCP merge failed: {e}")
        
        return {
            "merged_data": merged_data,
            "merge_metadata": merge_metadata
        }

    # === CONFIGURATION-DRIVEN PROCESSING ===
    
    async def _process_operation(self, operation: Dict, field_name: str, 
                                data_sources: List[Dict], field_value: Any = None) -> Dict:
        """Process a single operation from configuration"""
        operation_type = operation.get("type", "").lower()
        result = {"operation": operation_type, "success": False, "result": None}
        
        try:
            if operation_type == "fuzzy_match":
                threshold = operation.get("value", 0.75)
                matches_data = await self._fuzzy_match_field(field_name, data_sources, {
                    "fuzzy_threshold": threshold,
                    "target_value": field_value or ""
                })
                if matches_data.get("matches"):
                    result["result"] = matches_data["matches"]
                    result["success"] = True
            
            elif operation_type == "comprehensivematch":
                comp_values = await self._identify_comprehensive_values(field_name, data_sources)
                result["result"] = comp_values.get("comprehensive_values", {})
                result["success"] = True
            
            elif operation_type == "regex_match":
                pattern = operation.get("pattern", "")
                if pattern:
                    regex_result = await self._regex_match_field(field_name, data_sources, pattern)
                    result["result"] = regex_result
                    result["success"] = bool(regex_result.get("regex_results"))
            
            elif operation_type == "llmenrichment":
                query = operation.get("query", f"Enrich {field_name}")
                support_field = operation.get("support_field")
                
                # Prepare data for enrichment
                enrich_data = {}
                if field_value:
                    enrich_data[field_name] = field_value
                
                # Add support field data if specified
                if support_field:
                    for source in data_sources:
                        support_value = source.get("data", {}).get(support_field)
                        if support_value:
                            enrich_data[support_field] = support_value
                            break
                
                enriched = await self._enrich_data(enrich_data, field_name, {
                    "fields": {
                        field_name: {
                            "enrichment_type": "complete",
                            "context": query
                        }
                    }
                })
                result["result"] = enriched.get("enriched_data", {})
                result["success"] = True
            
            elif operation_type == "validation":
                validation_rules = operation.get("rules", {})
                validation_result = await self._validate_field_data(field_name, field_value, validation_rules)
                result["result"] = validation_result
                result["success"] = validation_result.get("is_valid", False)
            
            elif operation_type == "normalization":
                normalization_rules = operation.get("rules", {})
                if field_value:
                    norm_result = await self._normalize_data({field_name: field_value}, 
                                                           field_name, normalization_rules)
                    result["result"] = norm_result.get("normalized_data", {})
                    result["success"] = True
            
        except Exception as e:
            logger.error(f"Operation {operation_type} failed for field {field_name}: {e}")
            result["error"] = str(e)
        
        return result

    async def _process_field_with_config(self, field_config: Dict, data_sources: List[Dict], 
                                       priority_list: List[str]) -> Dict:
        """Process field using configuration-driven approach"""
        field_name = field_config.get("field_name")
        priority = field_config.get("priority", priority_list)
        operations = field_config.get("operation", [])
        dependency = field_config.get("dependency")
        
        logger.info(f"Processing field '{field_name}' with {len(operations)} operations")
        
        field_result = {
            "field_name": field_name,
            "operations_results": [],
            "final_value": None,
            "confidence": 0.0,
            "source": None,
            "dependency_satisfied": True
        }
        
        # Check dependency
        if dependency:
            depend_field = dependency.get("depend_on_field")
            if depend_field and depend_field not in self.processed_fields:
                field_result["dependency_satisfied"] = False
                field_result["error"] = f"Dependency field '{depend_field}' not processed yet"
                return field_result
        
        # Filter sources by priority
        prioritized_sources = []
        for priority_source in priority:
            matching_sources = [s for s in data_sources if s.get("name", "").lower() == priority_source.lower()]
            prioritized_sources.extend(matching_sources)
        
        # Add remaining sources
        used_ids = {s.get("id") for s in prioritized_sources}
        remaining_sources = [s for s in data_sources if s.get("id") not in used_ids]
        prioritized_sources.extend(remaining_sources)
        
        # Execute operations in sequence
        current_value = None
        best_confidence = 0.0
        
        for i, operation in enumerate(operations):
            op_result = await self._process_operation(operation, field_name, 
                                                    prioritized_sources, current_value)
            field_result["operations_results"].append(op_result)
            
            # Update current value if operation was successful
            if op_result.get("success") and op_result.get("result"):
                op_confidence = 0.8  # Base confidence for successful operations
                
                if op_result["operation"] == "fuzzy_match":
                    matches = op_result["result"]
                    if matches:
                        best_match = max(matches, key=lambda x: x.get("confidence", 0))
                        current_value = best_match.get("value")
                        op_confidence = best_match.get("confidence", 0.8)
                
                elif op_result["operation"] == "comprehensivematch":
                    comp_result = op_result["result"]
                    if comp_result:
                        current_value = comp_result.get("value") or comp_result.get(field_name)
                        op_confidence = comp_result.get("confidence", 0.9)
                
                elif op_result["operation"] == "regex_match":
                    regex_results = op_result["result"].get("regex_results", [])
                    if regex_results:
                        best_regex = max(regex_results, key=lambda x: x.get("confidence", 0))
                        current_value = best_regex.get("value")
                        op_confidence = best_regex.get("confidence", 0.9)
                
                elif op_result["operation"] == "llmenrichment":
                    enriched_data = op_result["result"]
                    current_value = enriched_data.get(field_name, current_value)
                    op_confidence = 0.8
                
                elif op_result["operation"] == "normalization":
                    normalized_data = op_result["result"]
                    current_value = normalized_data.get(field_name, current_value)
                    op_confidence = best_confidence + 0.1  # Slight boost for normalization
                
                # Update best value if this operation has higher confidence
                if op_confidence > best_confidence:
                    field_result["final_value"] = current_value
                    field_result["confidence"] = op_confidence
                    field_result["source"] = f"operation_{i}_{op_result['operation']}"
                    best_confidence = op_confidence
        
        # If no operations succeeded, try fallback comprehensive match
        if not field_result["final_value"] and prioritized_sources:
            try:
                fallback_result = await self._identify_comprehensive_values(field_name, prioritized_sources)
                comp_values = fallback_result.get("comprehensive_values", {})
                if comp_values:
                    field_result["final_value"] = comp_values.get("value") or comp_values.get(field_name)
                    field_result["confidence"] = comp_values.get("confidence", 0.5)
                    field_result["source"] = "fallback_comprehensive"
            except Exception as e:
                logger.warning(f"Fallback processing failed for {field_name}: {e}")
        
        return field_result

    # === EXISTING METHODS (keeping the good ones from your original code) ===
    
    async def _set_field_dependencies(self, config: Dict):
        """Set up field dependencies from config"""
        logger.info("Setting up field dependencies...")
        
        # Handle both old format and new grouped format
        if "groups" in config:
            for group in config["groups"]:
                for field_config in group.get("field_list", []):
                    field_name = field_config.get("field_name")
                    dependency = field_config.get("dependency")
                    if dependency and dependency.get("depend_on_field"):
                        dependencies = [dependency["depend_on_field"]]
                        self.field_dependencies[field_name] = dependencies
                        
                        result = await self._call_mcp_tool("set_field_dependencies", {
                            "field_name": field_name,
                            "dependencies": dependencies
                        })
                        logger.info(f"Set dependencies for {field_name}: {dependencies}")
        
        elif "fields" in config:
            # Original format
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
        
        target_value = field_config.get("target_value", "")
        threshold = field_config.get("fuzzy_threshold", 0.8)
        
        result = await self._call_mcp_tool("fuzzy_match_sources", {
            "target_value": target_value,
            "sources": data_sources,
            "field_name": field_name,
            "threshold": threshold
        })
        
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

    async def _calculate_confidence_scores(self, data_sources: List[Dict], 
                                         processed_fields: Dict) -> Dict:
        """Calculate overall confidence scores for processed data"""
        logger.info("Calculating confidence scores")
        
        confidence_results = {}
        
        # Local confidence calculation
        for field_name, field_data in processed_fields.items():
            field_confidence = field_data.get("confidence", 0.0)
            
            # Adjust confidence based on source reliability
            source = field_data.get("source", "")
            if "email" in source.lower():
                field_confidence *= 1.1  # Email sources are typically reliable
            elif "acord" in source.lower():
                field_confidence *= 1.2  # ACORD documents are highly reliable
            elif "broker" in source.lower():
                field_confidence *= 0.9   # Broker data might be less standardized
            
            # Cap confidence at 1.0
            field_confidence = min(1.0, field_confidence)
            
            confidence_results[field_name] = {
                "confidence": field_confidence,
                "source": source,
                "reliability_factor": field_confidence / field_data.get("confidence", 1.0)
            }
        
        # Use MCP confidence calculation tool
        try:
            mcp_result = await self._call_mcp_tool("calculate_confidence_scores", {
                "sources": data_sources,
                "processed_fields": processed_fields
            })
            mcp_data = _safe_parse_json(mcp_result, {})
            if mcp_data.get("confidence_scores"):
                # Merge MCP confidence with local calculations
                for field, mcp_conf in mcp_data["confidence_scores"].items():
                    if field in confidence_results:
                        confidence_results[field]["mcp_confidence"] = mcp_conf
                        # Use average of local and MCP confidence
                        local_conf = confidence_results[field]["confidence"]
                        confidence_results[field]["final_confidence"] = (local_conf + mcp_conf) / 2
                    else:
                        confidence_results[field] = {"mcp_confidence": mcp_conf, "final_confidence": mcp_conf}
        except Exception as e:
            logger.warning(f"MCP confidence calculation failed: {e}")
        
        return {"confidence_scores": confidence_results}

    async def _detect_anomalies(self, data_sources: List[Dict], processed_fields: Dict) -> Dict:
        """Detect anomalies in processed data"""
        logger.info("Detecting data anomalies")
        
        anomalies = []
        
        # Local anomaly detection
        for field_name, field_data in processed_fields.items():
            field_value = field_data.get("final_value")
            confidence = field_data.get("confidence", 0.0)
            
            # Check for low confidence
            if confidence < 0.3:
                anomalies.append({
                    "type": "low_confidence",
                    "field": field_name,
                    "value": field_value,
                    "confidence": confidence,
                    "description": f"Field {field_name} has unusually low confidence: {confidence}"
                })
            
            # Check for missing critical fields
            critical_fields = ["Insured_Name", "broker_Name", "client_email", "broker_email"]
            if field_name in critical_fields and not field_value:
                anomalies.append({
                    "type": "missing_critical_field",
                    "field": field_name,
                    "description": f"Critical field {field_name} is missing"
                })
            
            # Check for inconsistent data formats
            if field_name.endswith("_email") and field_value:
                if not re.match(r'^[\w.-]+@[\w.-]+\.\w{2,}, str(field_value)):
                    anomalies.append({
                        "type": "format_inconsistency",
                        "field": field_name,
                        "value": field_value,
                        "description": f"Email field {field_name} has invalid format"
                    })
            
            if field_name.endswith("_phone_number") and field_value:
                if not re.match(r'\d{3}-\d{3}-\d{4}', str(field_value)):
                    anomalies.append({
                        "type": "format_inconsistency",
                        "field": field_name,
                        "value": field_value,
                        "description": f"Phone field {field_name} has invalid format"
                    })
        
        # Use MCP anomaly detection tool
        try:
            mcp_result = await self._call_mcp_tool("detect_anomalies", {
                "sources": data_sources,
                "processed_fields": processed_fields
            })
            mcp_data = _safe_parse_json(mcp_result, {})
            if mcp_data.get("anomalies"):
                # Add MCP-detected anomalies
                mcp_anomalies = mcp_data["anomalies"]
                for anomaly in mcp_anomalies:
                    anomaly["source"] = "mcp_detection"
                anomalies.extend(mcp_anomalies)
        except Exception as e:
            logger.warning(f"MCP anomaly detection failed: {e}")
        
        return {
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "has_critical_anomalies": any(a.get("type") == "missing_critical_field" for a in anomalies)
        }

    # === COMPREHENSIVE DATA PROCESSING ===
    
    async def process_data_comprehensive(self, 
                                       config: Dict,
                                       data_sources: List[Dict],
                                       raw_data: Dict = None,
                                       priority_list: List[str] = None) -> Dict:
        """
        Comprehensive data processing with complete tool coverage
        """
        if not self._session_initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        
        logger.info("Starting comprehensive data processing with complete MCP tool coverage")
        
        # Extract priority list from config if not provided
        if not priority_list:
            priority_list = config.get("global_settings", {}).get("priority_list", ["email", "acord", "broker"])
        
        # Ensure we have raw_data
        if not raw_data:
            raw_data = {"sources": data_sources, "config": config}
        
        try:
            # Step 1: Set up field dependencies
            await self._set_field_dependencies(config)
            
            # Step 2: Process fields according to new configuration format
            field_results = {}
            
            # Handle new grouped configuration format
            if "groups" in config:
                all_fields = []
                for group in config["groups"]:
                    group_name = group.get("group_name")
                    logger.info(f"Processing group: {group_name}")
                    
                    for field_config in group.get("field_list", []):
                        field_name = field_config.get("field_name")
                        
                        # Process field with new configuration-driven approach
                        field_result = await self._process_field_with_config(
                            field_config, data_sources, priority_list
                        )
                        field_results[field_name] = field_result
                        
                        # Update processed fields for dependency tracking
                        if field_result.get("final_value"):
                            self.processed_fields[field_name] = field_result
                        
                        all_fields.append(field_name)
                        
                        # Add delay between field processing
                        await asyncio.sleep(0.1)
            
            # Handle original configuration format as fallback
            elif "fields" in config:
                fields_config = config.get("fields", {})
                
                # Sort fields by dependencies
                def get_dependency_order(fields):
                    ordered = []
                    remaining = set(fields.keys())
                    
                    while remaining:
                        ready = []
                        for field in remaining:
                            deps = self.field_dependencies.get(field, [])
                            if not deps or all(dep in ordered for dep in deps):
                                ready.append(field)
                        
                        if not ready:
                            ready = [next(iter(remaining))]
                        
                        for field in ready:
                            ordered.append(field)
                            remaining.remove(field)
                    
                    return ordered
                
                ordered_fields = get_dependency_order(fields_config)
                
                for field_name in ordered_fields:
                    field_config = fields_config[field_name]
                    # Convert old format to new format for processing
                    converted_config = {
                        "field_name": field_name,
                        "priority": priority_list,
                        "operation": [{"type": "ComprehensiveMatch"}],  # Default operation
                        "dependency": {"depend_on_field": self.field_dependencies.get(field_name, [None])[0]} if field_name in self.field_dependencies else None
                    }
                    
                    field_result = await self._process_field_with_config(
                        converted_config, data_sources, priority_list
                    )
                    field_results[field_name] = field_result
                    
                    if field_result.get("final_value"):
                        self.processed_fields[field_name] = field_result
                    
                    await asyncio.sleep(0.1)
            
            # Step 3: Merge data sources for comprehensive analysis
            merge_result = await self._merge_data_sources(data_sources, "confidence")
            merged_data = merge_result.get("merged_data", {})
            
            # Step 4: Cross-reference fields
            field_relationships = {
                "Insured_Name": ["client_street", "client_city", "client_state"],
                "broker_Name": ["broker_street", "broker_city", "broker_state"],
                "client_street": ["client_city", "client_state", "client_zip_code"],
                "broker_street": ["broker_city", "broker_state", "broker_zip_code"]
            }
            
            cross_ref_result = await self._cross_reference_fields(data_sources, field_relationships)
            
            # Step 5: Calculate confidence scores
            confidence_result = await self._calculate_confidence_scores(data_sources, field_results)
            
            # Step 6: Detect anomalies
            anomaly_result = await self._detect_anomalies(data_sources, field_results)
            
            # Step 7: Advanced RAG queries for missing data
            rag_results = {}
            missing_fields = [name for name, result in field_results.items() 
                            if not result.get("final_value")]
            
            if missing_fields:
                for field in missing_fields[:3]:  # Limit to avoid overprocessing
                    rag_query = f"Find the best value for {field} from the provided data sources"
                    rag_result = await self._rag_query(rag_query, raw_data, priority_list)
                    rag_results[field] = rag_result.get("rag_results", {})
            
            # Step 8: Generate comprehensive summary
            all_results = {
                "processed_fields": field_results,
                "merged_data": merged_data,
                "cross_references": cross_ref_result.get("cross_reference_results", {}),
                "confidence_analysis": confidence_result.get("confidence_scores", {}),
                "anomaly_detection": anomaly_result,
                "rag_enhancement": rag_results,
                "metadata": {
                    "total_fields_processed": len(field_results),
                    "total_sources": len(data_sources),
                    "priority_list": priority_list,
                    "processing_timestamp": asyncio.get_event_loop().time(),
                    "successful_fields": len([r for r in field_results.values() if r.get("final_value")]),
                    "failed_fields": len([r for r in field_results.values() if not r.get("final_value")]),
                    "average_confidence": sum(r.get("confidence", 0) for r in field_results.values()) / len(field_results) if field_results else 0
                }
            }
            
            summary_result = await self._generate_summary(all_results, "comprehensive", True)
            
            # Step 9: Additional comprehensive scenarios
            additional_results = await self._handle_comprehensive_scenarios(
                data_sources, priority_list, raw_data, field_results
            )
            
            final_result = {
                "status": "completed",
                "processed_data": {name: result.get("final_value") for name, result in field_results.items()},
                "detailed_results": field_results,
                "merged_analysis": merged_data,
                "cross_reference_analysis": cross_ref_result.get("cross_reference_results", {}),
                "confidence_analysis": confidence_result.get("confidence_scores", {}),
                "anomaly_report": anomaly_result,
                "rag_enhancements": rag_results,
                "additional_insights": additional_results,
                "summary": summary_result.get("summary", {}),
                "metadata": all_results["metadata"]
            }
            
            logger.info(f"Comprehensive processing completed. Success rate: {final_result['metadata']['successful_fields']}/{final_result['metadata']['total_fields_processed']}")
            return final_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive processing: {e}")
            return {
                "status": "error",
                "error": str(e),
                "partial_results": self.processed_fields,
                "field_results": field_results if 'field_results' in locals() else {}
            }

    async def _handle_comprehensive_scenarios(self, data_sources: List[Dict], 
                                            priority_list: List[str], raw_data: Dict,
                                            processed_fields: Dict) -> Dict:
        """Handle comprehensive scenarios covering all remaining tools"""
        comprehensive_results = {}
        
        logger.info("Executing comprehensive scenarios for complete tool coverage...")
        
        # Scenario 1: Advanced pattern extraction
        try:
            # Extract common business patterns
            patterns = {
                "email_pattern": r'^[\w.-]+@[\w.-]+\.\w{2,},
                "phone_pattern": r'\d{3}-\d{3}-\d{4}',
                "zip_pattern": r'\d{5}(-\d{4})?',
                "state_pattern": r'\b[A-Z]{2}\b'
            }
            
            pattern_results = {}
            for pattern_name, pattern in patterns.items():
                for source in data_sources[:2]:  # Limit sources to avoid overprocessing
                    source_data = source.get("data", {})
                    for field_name, field_value in source_data.items():
                        if isinstance(field_value, str):
                            extract_result = await self._regex_match_field(
                                field_name, [source], pattern, extract_mode=True
                            )
                            if extract_result.get("regex_results"):
                                pattern_results[f"{field_name}_{pattern_name}"] = extract_result
                                break
            
            comprehensive_results["pattern_extraction"] = pattern_results
        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
        
        # Scenario 2: Data normalization across all fields
        try:
            normalization_results = {}
            normalization_rules = {
                "case": "title",
                "trim": True,
                "phone_format": True,
                "state_abbreviation": True
            }
            
            for field_name, field_result in processed_fields.items():
                field_value = field_result.get("final_value")
                if field_value:
                    norm_result = await self._normalize_data(
                        {field_name: field_value}, field_name, normalization_rules
                    )
                    if norm_result.get("normalized_data", {}).get(field_name) != field_value:
                        normalization_results[field_name] = norm_result
            
            comprehensive_results["normalization_analysis"] = normalization_results
        except Exception as e:
            logger.error(f"Normalization analysis failed: {e}")
        
        # Scenario 3: Comprehensive validation
        try:
            validation_results = {}
            validation_rules_map = {
                "email": {"type": "email", "required": True},
                "phone": {"type": "phone", "required": False},
                "zip": {"type": "zip", "required": False},
                "name": {"type": "string", "min_length": 2, "required": True}
            }
            
            for field_name, field_result in processed_fields.items():
                field_value = field_result.get("final_value")
                
                # Determine validation rules based on field name
                validation_rules = {}
                for rule_type, rules in validation_rules_map.items():
                    if rule_type in field_name.lower():
                        validation_rules = rules
                        break
                
                if validation_rules:
                    validation_result = await self._validate_field_data(
                        field_name, field_value, validation_rules
                    )
                    validation_results[field_name] = validation_result
            
            comprehensive_results["comprehensive_validation"] = validation_results
        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
        
        # Scenario 4: Multi-source RAG queries
        try:
            advanced_rag_results = {}
            rag_queries = [
                "What is the complete address information available?",
                "What contact information is available for all parties?",
                "Are there any inconsistencies in the provided data?",
                "What additional information can be inferred from the context?"
            ]
            
            for i, query in enumerate(rag_queries):
                rag_result = await self._rag_query(query, raw_data, priority_list)
                advanced_rag_results[f"advanced_query_{i+1}"] = {
                    "query": query,
                    "result": rag_result.get("rag_results", {})
                }
            
            comprehensive_results["advanced_rag_analysis"] = advanced_rag_results
        except Exception as e:
            logger.error(f"Advanced RAG analysis failed: {e}")
        
        # Scenario 5: Priority-based comprehensive processing
        try:
            all_field_names = list(processed_fields.keys())
            if all_field_names:
                priority_result = await self._process_with_priority(
                    data_sources, priority_list, all_field_names
                )
                comprehensive_results["priority_comprehensive"] = priority_result.get("priority_results", {})
        except Exception as e:
            logger.error(f"Priority comprehensive processing failed: {e}")
        
        return comprehensive_results

    # === CLEANUP AND CONTEXT MANAGEMENT ===
    
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


class EnhancedDataProcessingClientManager:
    """Context manager wrapper for proper resource management"""
    
    def __init__(self, azure_config: Dict[str, str], server_url: str = "http://127.0.0.1:8080"):
        self.azure_config = azure_config
        self.server_url = server_url
        self.client = None
    
    async def __aenter__(self):
        self.client = EnhancedDataProcessingClient(self.azure_config)
        await self.client.initialize(self.server_url)
        return self.client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.cleanup()
        self.client = None


# === MAIN FUNCTIONS ===

async def main_enhanced(config: Dict, extracted_data: List[Dict]):
    """
    Enhanced main function with complete MCP tool coverage
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
    async with EnhancedDataProcessingClientManager(azure_config) as client:
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
                    "confidence": source.get("confidence", 0.7)
                })

            logger.info(f"Processing {len(data_sources)} data sources with enhanced comprehensive processing")
            
            # Use comprehensive processing method
            result = await client.process_data_comprehensive(
                config=config,
                data_sources=data_sources,
                raw_data={"extracted": extracted_data, "config": config}
            )
            
            logger.info("--- ENHANCED PROCESSING COMPLETED ---")
            logger.info(f"Status: {result.get('status', 'unknown')}")
            logger.info(f"Success rate: {result.get('metadata', {}).get('successful_fields', 0)}/{result.get('metadata', {}).get('total_fields_processed', 0)}")
            logger.info(f"Average confidence: {result.get('metadata', {}).get('average_confidence', 0):.3f}")
            logger.info(f"Anomalies detected: {result.get('anomaly_report', {}).get('anomaly_count', 0)}")
            
            return result
            
        except Exception as e:
            logger.error(f"An error occurred during enhanced processing: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}


# === BACKWARDS COMPATIBILITY ===

# Keep the original main function for backwards compatibility
async def main(config: Dict, extracted_data: List[Dict]):
    """Original main function - calls enhanced version"""
    return await main_enhanced(config, extracted_data)


async def main_direct_enhanced(config: Dict, extracted_data: List[Dict]):
    """Enhanced direct usage without context manager"""
    azure_config = {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": "2024-05-01-preview"
    }
    
    if not all([azure_config["api_key"], azure_config["endpoint"]]):
        return {"error": "Azure OpenAI configuration missing"}

    client = EnhancedDataProcessingClient(azure_config)
    
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
                "confidence": source.get("confidence", 0.7)
            })
        
        result = await client.process_data_comprehensive(
            config=config,
            data_sources=data_sources,
            raw_data={"extracted": extracted_data, "config": config}
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Enhanced processing failed: {e}", exc_info=True)
        return {"error": str(e), "status": "failed"}
    
    finally:
        await client.cleanup()


if __name__ == "__main__":
    # Example usage:
    # with open("config.json", "r") as f:
    #     config_data = json.load(f)
    # with open("extracted_data.json", "r") as f:
    #     extracted_data_json = json.load(f)
    # result = asyncio.run(main_enhanced(config_data, extracted_data_json))
    pass

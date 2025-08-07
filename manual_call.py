def validate_tool_parameters(tool_name: str, parameters: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate tool parameters before calling MCP tools
    Returns (is_valid, error_message)
    """
    validations = {
        "set_field_dependencies": {
            "required": ["field_name"],
            "optional": ["dependencies"],
            "types": {"field_name": str, "dependencies": list}
        },
        "fuzzy_match_sources": {
            "required": ["target_value", "sources", "field_name"],
            "optional": ["threshold"],
            "types": {"target_value": str, "sources": list, "field_name": str, "threshold": float}
        },
        "process_with_priority": {
            "required": ["sources", "priority_list", "fields"],
            "optional": [],
            "types": {"sources": list, "priority_list": list, "fields": list}
        },
        "identify_comprehensive_values": {
            "required": ["sources", "field_name"],
            "optional": [],
            "types": {"sources": list, "field_name": str}
        },
        "enrich_data_llm": {
            "required": ["data", "enrichment_prompt"],
            "optional": ["context"],
            "types": {"data": dict, "enrichment_prompt": str, "context": str}
        },
        "rag_query": {
            "required": ["query", "raw_data"],
            "optional": ["context_sources"],
            "types": {"query": str, "raw_data": dict, "context_sources": list}
        },
        "generate_summary": {
            "required": ["data"],
            "optional": ["summary_type", "include_sources"],
            "types": {"data": dict, "summary_type": str, "include_sources": bool}
        },
        "select_best_match": {
            "required": ["matches"],
            "optional": ["priority_list"],
            "types": {"matches": list, "priority_list": list}
        }
    }
    
    if tool_name not in validations:
        return False, f"Unknown tool: {tool_name}"
    
    validation = validations[tool_name]
    
    # Check required parameters
    for required_param in validation["required"]:
        if required_param not in parameters:
            return False, f"Missing required parameter: {required_param}"
        
        # Check parameter type
        expected_type = validation["types"].get(required_param)
        if expected_type and not isinstance(parameters[required_param], expected_type):
            return False, f"Parameter '{required_param}' must be of type {expected_type.__name__}"
    
    # Check parameter types for optional parameters
    for param_name, param_value in parameters.items():
        expected_type = validation["types"].get(param_name)
        if expected_type and not isinstance(param_value, expected_type):
            return False, f"Parameter '{param_name}' must be of type {expected_type.__name__}"
    
    return True, "Valid parameters"


# Enhanced Tool Classes with improved parameter handling

class FuzzyMatchTool(BaseTool):
    """Tool for fuzzy matching across data sources - ENHANCED"""

    name: str = "fuzzy_match_sources"
    description: str = (
        "Perform fuzzy matching across multiple data sources for a specific field"
    )
    mcp_session: Any = None

    class ArgsSchema(BaseModel):
        target_value: str = Field(
            description="Target value to match against (can be empty for auto-detection)"
        )
        sources: List[Dict] = Field(description="List of data sources to search")
        field_name: str = Field(description="Name of the field to match")
        threshold: float = Field(
            default=0.8, description="Similarity threshold (0.0-1.0)"
        )

    args_schema: Type[BaseModel] = ArgsSchema

    def _run(
        self,
        target_value: str,
        sources: List[Dict],
        field_name: str,
        threshold: float = 0.8,
        **kwargs,
    ) -> str:
        """Run the fuzzy match tool"""
        return asyncio.run(
            self._arun(target_value, sources, field_name, threshold, **kwargs)
        )

    async def _arun(
        self,
        target_value: str,
        sources: List[Dict],
        field_name: str,
        threshold: float = 0.8,
        **kwargs,
    ) -> str:
        """Run the fuzzy match tool asynchronously"""
        try:
            # Prepare parameters
            params = {
                "target_value": target_value,
                "sources": sources,
                "field_name": field_name,
                "threshold": threshold,
            }
            
            # Validate parameters
            is_valid, error_msg = validate_tool_parameters("fuzzy_match_sources", params)
            if not is_valid:
                return f"Parameter validation error: {error_msg}"

            # Additional business logic validation
            if not target_value.strip():
                return "Error: target_value cannot be empty"
            if not sources:
                return "Error: sources list cannot be empty"
            if not field_name.strip():
                return "Error: field_name cannot be empty"

            logger.info(f"Performing fuzzy matching for field '{field_name}' with threshold {threshold}")

            result = await self.mcp_session.call_tool("fuzzy_match_sources", params)

            return _parse_mcp_result(result, "No matches found")
        except Exception as e:
            logger.error(f"Error in fuzzy matching tool: {e}", exc_info=True)
            return f"Error in fuzzy matching: {str(e)}"


class SelectBestMatchTool(BaseTool):
    """Tool for selecting the best match from fuzzy match results - ENHANCED"""

    name: str = "select_best_match"
    description: str = "Select the best matching value based on similarity and priority"
    mcp_session: Any = None

    class ArgsSchema(BaseModel):
        matches: List[Dict] = Field(description="List of matches from fuzzy matching")
        priority_list: List[str] = Field(
            default_factory=list,
            description="Priority list for source ranking"
        )

    args_schema: Type[BaseModel] = ArgsSchema

    def _run(
        self, matches: List[Dict], priority_list: List[str] = None, **kwargs
    ) -> str:
        return asyncio.run(self._arun(matches, priority_list or [], **kwargs))

    async def _arun(
        self, matches: List[Dict], priority_list: List[str] = None, **kwargs
    ) -> str:
        try:
            # Handle None case for priority_list
            if priority_list is None:
                priority_list = []

            # Prepare parameters
            params = {"matches": matches}
            if priority_list:  # Only add priority_list if it's not empty
                params["priority_list"] = priority_list

            # Validate parameters
            is_valid, error_msg = validate_tool_parameters("select_best_match", params)
            if not is_valid:
                return f"Parameter validation error: {error_msg}"

            # Additional business logic validation
            if not matches:
                return "Error: matches list cannot be empty"

            logger.info(f"Selecting best match from {len(matches)} matches with priority: {priority_list}")

            result = await self.mcp_session.call_tool("select_best_match", params)

            return _parse_mcp_result(result, "No best match selected")
        except Exception as e:
            logger.error(f"Error in select best match tool: {e}", exc_info=True)
            return f"Error in selecting best match: {str(e)}"


class ComprehensiveValuesTool(BaseTool):
    """Tool for identifying comprehensive values - ENHANCED"""

    name: str = "identify_comprehensive_values"
    description: str = (
        "Identify comprehensive values with source tracking for a specific field"
    )
    mcp_session: Any = None

    class ArgsSchema(BaseModel):
        sources: List[Dict] = Field(description="List of data sources")
        field_name: str = Field(description="Name of the field to analyze")

    args_schema: Type[BaseModel] = ArgsSchema

    def _run(self, sources: List[Dict], field_name: str, **kwargs) -> str:
        return asyncio.run(self._arun(sources, field_name, **kwargs))

    async def _arun(self, sources: List[Dict], field_name: str, **kwargs) -> str:
        try:
            # Prepare parameters
            params = {"sources": sources, "field_name": field_name}
            
            # Validate parameters
            is_valid, error_msg = validate_tool_parameters("identify_comprehensive_values", params)
            if not is_valid:
                return f"Parameter validation error: {error_msg}"

            # Additional business logic validation
            if not sources:
                return "Error: sources list cannot be empty"
            if not field_name.strip():
                return "Error: field_name cannot be empty"

            logger.info(f"Identifying comprehensive values for field '{field_name}' from {len(sources)} sources")

            result = await self.mcp_session.call_tool("identify_comprehensive_values", params)

            return _parse_mcp_result(result, "No comprehensive values found")
        except Exception as e:
            logger.error(f"Error in comprehensive values tool: {e}", exc_info=True)
            return f"Error in identifying comprehensive values: {str(e)}"


class EnrichDataTool(BaseTool):
    """Tool for enriching data using LLM - ENHANCED"""

    name: str = "enrich_data_llm"
    description: str = "Enrich data using LLM with source preservation"
    mcp_session: Any = None

    class ArgsSchema(BaseModel):
        data: Dict = Field(description="Data to enrich")
        enrichment_prompt: str = Field(description="Prompt for enrichment")
        context: str = Field(default="", description="Additional context")

    args_schema: Type[BaseModel] = ArgsSchema

    def _run(
        self, data: Dict, enrichment_prompt: str, context: str = "", **kwargs
    ) -> str:
        return asyncio.run(self._arun(data, enrichment_prompt, context, **kwargs))

    async def _arun(
        self, data: Dict, enrichment_prompt: str, context: str = "", **kwargs
    ) -> str:
        try:
            # Prepare parameters
            params = {"data": data, "enrichment_prompt": enrichment_prompt}
            if context:  # Only add context if it's not empty
                params["context"] = context

            # Validate parameters
            is_valid, error_msg = validate_tool_parameters("enrich_data_llm", params)
            if not is_valid:
                return f"Parameter validation error: {error_msg}"

            # Additional business logic validation
            if not data:
                return "Error: data cannot be empty"
            if not enrichment_prompt.strip():
                return "Error: enrichment_prompt cannot be empty"

            logger.info(f"Enriching data using LLM with prompt: {enrichment_prompt[:50]}...")

            result = await self.mcp_session.call_tool("enrich_data_llm", params)

            return _parse_mcp_result(result, "No enrichment performed")
        except Exception as e:
            logger.error(f"Error in data enrichment tool: {e}", exc_info=True)
            return f"Error in data enrichment: {str(e)}"


class ProcessWithPriorityTool(BaseTool):
    """Tool for processing data with priority - ENHANCED"""

    name: str = "process_with_priority"
    description: str = "Process data according to priority list"
    mcp_session: Any = None

    class ArgsSchema(BaseModel):
        sources: List[Dict] = Field(description="List of data sources")
        priority_list: List[str] = Field(description="Priority list for source ranking")
        fields: List[str] = Field(description="List of fields to process")

    args_schema: Type[BaseModel] = ArgsSchema

    def _run(
        self, sources: List[Dict], priority_list: List[str], fields: List[str], **kwargs
    ) -> str:
        return asyncio.run(self._arun(sources, priority_list, fields, **kwargs))

    async def _arun(
        self, sources: List[Dict], priority_list: List[str], fields: List[str], **kwargs
    ) -> str:
        try:
            # Prepare parameters
            params = {
                "sources": sources,
                "priority_list": priority_list,
                "fields": fields
            }
            
            # Validate parameters
            is_valid, error_msg = validate_tool_parameters("process_with_priority", params)
            if not is_valid:
                return f"Parameter validation error: {error_msg}"

            # Additional business logic validation
            if not sources:
                return "Error: sources list cannot be empty"
            if not priority_list:
                return "Error: priority_list cannot be empty"
            if not fields:
                return "Error: fields list cannot be empty"

            logger.info(f"Processing {len(fields)} fields with priority order: {priority_list}")

            result = await self.mcp_session.call_tool("process_with_priority", params)

            return _parse_mcp_result(result, "No priority processing results")
        except Exception as e:
            logger.error(f"Error in priority processing tool: {e}", exc_info=True)
            return f"Error in priority processing: {str(e)}"


class SetDependenciesTool(BaseTool):
    """Tool for setting field dependencies - ENHANCED"""

    name: str = "set_field_dependencies"
    description: str = "Set dependencies between fields"
    mcp_session: Any = None

    class ArgsSchema(BaseModel):
        field_name: str = Field(description="Name of the field")
        dependencies: List[str] = Field(
            default_factory=list,
            description="List of dependency fields"
        )

    args_schema: Type[BaseModel] = ArgsSchema

    def _run(self, field_name: str, dependencies: List[str] = None, **kwargs) -> str:
        return asyncio.run(self._arun(field_name, dependencies or [], **kwargs))

    async def _arun(self, field_name: str, dependencies: List[str] = None, **kwargs) -> str:
        try:
            # Handle None case for dependencies
            if dependencies is None:
                dependencies = []

            # Prepare parameters
            params = {"field_name": field_name, "dependencies": dependencies}
            
            # Validate parameters
            is_valid, error_msg = validate_tool_parameters("set_field_dependencies", params)
            if not is_valid:
                return f"Parameter validation error: {error_msg}"

            # Additional business logic validation
            if not field_name.strip():
                return "Error: field_name cannot be empty"
            if not dependencies:
                return f"No dependencies specified for field '{field_name}'. Use this tool only when you have dependencies to set."

            logger.info(f"Setting dependencies for {field_name}: {dependencies}")

            result = await self.mcp_session.call_tool("set_field_dependencies", params)

            return _parse_mcp_result(result, f"Dependencies set for {field_name}")
        except Exception as e:
            logger.error(f"Error in set dependencies tool: {e}", exc_info=True)
            return f"Error in setting dependencies: {str(e)}"


class RAGQueryTool(BaseTool):
    """Tool for RAG-based queries - ENHANCED"""

    name: str = "rag_query"
    description: str = "Perform RAG-based query for data enrichment"
    mcp_session: Any = None

    class ArgsSchema(BaseModel):
        query: str = Field(description="Query for RAG system")
        raw_data: Dict = Field(description="Raw data for context")
        context_sources: List[str] = Field(
            default_factory=list,
            description="Additional context sources"
        )

    args_schema: Type[BaseModel] = ArgsSchema

    def _run(
        self,
        query: str,
        raw_data: Dict,
        context_sources: List[str] = None,
        **kwargs,
    ) -> str:
        return asyncio.run(self._arun(query, raw_data, context_sources or [], **kwargs))

    async def _arun(
        self,
        query: str,
        raw_data: Dict,
        context_sources: List[str] = None,
        **kwargs,
    ) -> str:
        try:
            # Handle None case for context_sources
            if context_sources is None:
                context_sources = []

            # Prepare parameters
            params = {"query": query, "raw_data": raw_data}
            if context_sources:  # Only add context_sources if it's not empty
                params["context_sources"] = context_sources

            # Validate parameters
            is_valid, error_msg = validate_tool_parameters("rag_query", params)
            if not is_valid:
                return f"Parameter validation error: {error_msg}"

            # Additional business logic validation
            if not query.strip():
                return "Error: query cannot be empty"
            if not raw_data:
                return "Error: raw_data cannot be empty"

            logger.info(f"Performing RAG query: {query[:50]}... with {len(context_sources)} context sources")

            result = await self.mcp_session.call_tool("rag_query", params)

            return _parse_mcp_result(result, "No RAG results")
        except Exception as e:
            logger.error(f"Error in RAG query tool: {e}", exc_info=True)
            return f"Error in RAG query: {str(e)}"


class GenerateSummaryTool(BaseTool):
    """Tool for generating summaries - ENHANCED"""

    name: str = "generate_summary"
    description: str = "Generate comprehensive summary using RAG and LLM"
    mcp_session: Any = None

    class ArgsSchema(BaseModel):
        data: Dict = Field(description="Data to summarize")
        summary_type: str = Field(
            default="comprehensive", 
            description="Type of summary"
        )
        include_sources: bool = Field(
            default=True, 
            description="Whether to include sources"
        )

    args_schema: Type[BaseModel] = ArgsSchema

    def _run(
        self,
        data: Dict,
        summary_type: str = "comprehensive",
        include_sources: bool = True,
        **kwargs,
    ) -> str:
        return asyncio.run(self._arun(data, summary_type, include_sources, **kwargs))

    async def _arun(
        self,
        data: Dict,
        summary_type: str = "comprehensive",
        include_sources: bool = True,
        **kwargs,
    ) -> str:
        try:
            # Prepare parameters - always include all parameters for consistency
            params = {
                "data": data,
                "summary_type": summary_type,
                "include_sources": include_sources
            }

            # Validate parameters
            is_valid, error_msg = validate_tool_parameters("generate_summary", params)
            if not is_valid:
                return f"Parameter validation error: {error_msg}"

            # Additional business logic validation
            if not data:
                return "Error: data cannot be empty"

            logger.info(f"Generating {summary_type} summary with sources: {include_sources}")

            result = await self.mcp_session.call_tool("generate_summary", params)

            return _parse_mcp_result(result, "No summary generated")
        except Exception as e:
            logger.error(f"Error in generate summary tool: {e}", exc_info=True)
            return f"Error in generating summary: {str(e)}"



















def validate_tool_parameters(tool_name: str, parameters: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate tool parameters before calling MCP tools
    Returns (is_valid, error_message)
    """
    validations = {
        "set_field_dependencies": {
            "required": ["field_name"],
            "optional": ["dependencies"],
            "types": {"field_name": str, "dependencies": list}
        },
        "fuzzy_match_sources": {
            "required": ["target_value", "sources", "field_name"],
            "optional": ["threshold"],
            "types": {"target_value": str, "sources": list, "field_name": str, "threshold": float}
        },
        "process_with_priority": {
            "required": ["sources", "priority_list", "fields"],
            "optional": [],
            "types": {"sources": list, "priority_list": list, "fields": list}
        },
        "identify_comprehensive_values": {
            "required": ["sources", "field_name"],
            "optional": [],
            "types": {"sources": list, "field_name": str}
        },
        "enrich_data_llm": {
            "required": ["data", "enrichment_prompt"],
            "optional": ["context"],
            "types": {"data": dict, "enrichment_prompt": str, "context": str}
        },
        "rag_query": {
            "required": ["query", "raw_data"],
            "optional": ["context_sources"],
            "types": {"query": str, "raw_data": dict, "context_sources": list}
        },
        "generate_summary": {
            "required": ["data"],
            "optional": ["summary_type", "include_sources"],
            "types": {"data": dict, "summary_type": str, "include_sources": bool}
        },
        "select_best_match": {
            "required": ["matches"],
            "optional": ["priority_list"],
            "types": {"matches": list, "priority_list": list}
        }
    }
    
    if tool_name not in validations:
        return False, f"Unknown tool: {tool_name}"
    
    validation = validations[tool_name]
    
    # Check required parameters
    for required_param in validation["required"]:
        if required_param not in parameters:
            return False, f"Missing required parameter: {required_param}"
        
        # Check parameter type
        expected_type = validation["types"].get(required_param)
        if expected_type and not isinstance(parameters[required_param], expected_type):
            return False, f"Parameter '{required_param}' must be of type {expected_type.__name__}"
    
    # Check parameter types for optional parameters
    for param_name, param_value in parameters.items():
        expected_type = validation["types"].get(param_name)
        if expected_type and not isinstance(param_value, expected_type):
            return False, f"Parameter '{param_name}' must be of type {expected_type.__name__}"
    
    return True, "Valid parameters"


# Enhanced version of SetDependenciesTool with validation
class SetDependenciesTool(BaseTool):
    """Tool for setting field dependencies - ENHANCED"""

    name: str = "set_field_dependencies"
    description: str = "Set dependencies between fields"
    mcp_session: Any = None

    class ArgsSchema(BaseModel):
        field_name: str = Field(description="Name of the field")
        dependencies: List[str] = Field(
            default_factory=list,
            description="List of dependency fields"
        )

    args_schema: Type[BaseModel] = ArgsSchema

    def _run(self, field_name: str, dependencies: List[str] = None, **kwargs) -> str:
        return asyncio.run(self._arun(field_name, dependencies or [], **kwargs))

    async def _arun(self, field_name: str, dependencies: List[str] = None, **kwargs) -> str:
        try:
            # Handle None case for dependencies
            if dependencies is None:
                dependencies = []

            # Validate parameters
            params = {"field_name": field_name, "dependencies": dependencies}
            is_valid, error_msg = validate_tool_parameters("set_field_dependencies", params)
            
            if not is_valid:
                return f"Parameter validation error: {error_msg}"

            # Validate that we have actual dependencies to set
            if not dependencies:
                return f"No dependencies specified for field '{field_name}'. Use this tool only when you have dependencies to set."

            logger.info(f"Setting dependencies for {field_name}: {dependencies}")

            result = await self.mcp_session.call_tool("set_field_dependencies", params)

            return _parse_mcp_result(result, f"Dependencies set for {field_name}")
        except Exception as e:
            logger.error(f"Error in set dependencies tool: {e}", exc_info=True)
            return f"Error in setting dependencies: {str(e)}"

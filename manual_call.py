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

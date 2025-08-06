async def process_by_config(self, config: Dict, extracted_data: List[Dict]) -> Dict:
        if not self.mcp_session:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        results = {}
        errors = []

        # Normalize data sources
        source_map = {src["source_id"]: src for src in extracted_data}

        # Prepare flat source list for tools
        flat_sources = [
            {
                "source_id": s["source_id"],
                "doc_type": s["doc_type"],
                "data": s["data"],
                "timestamp": s["metadata"].get("timestamp", ""),
                "confidence": s["metadata"].get("extraction_confidence", 0.0)
            }
            for s in extracted_data
        ]

        for section, operations in config.items():
            results[section] = {}
            for op in operations:
                try:
                    field = op["field_name"]
                    operation = op["operation"]
                    sources_order = op.get("priority_sources", [])
                    dependencies = op.get("dependencies", [])
                    params = op.get("parameters", {})

                    # Get source-specific data for the field
                    relevant_sources = [
                        {
                            "source_id": src["source_id"],
                            "doc_type": src["doc_type"],
                            "value": src["data"].get(field, ""),
                            "metadata": src["metadata"]
                        }
                        for src in extracted_data
                        if src["doc_type"] in sources_order and field in src["data"]
                    ]

                    if not relevant_sources:
                        continue

                    if operation == "fuzzy_match":
                        threshold = params.get("threshold", 0.8)
                        target_value = relevant_sources[0]["value"]

                        match_result = await self.mcp_session.call_tool("fuzzy_match_sources", {
                            "target_value": target_value,
                            "sources": relevant_sources,
                            "field_name": field,
                            "threshold": threshold
                        })
                        results[section][field] = _parse_mcp_result(match_result, "No matches")

                    elif operation == "comprehensive_analysis":
                        comp_result = await self.mcp_session.call_tool("identify_comprehensive_values", {
                            "sources": relevant_sources,
                            "field_name": field
                        })
                        results[section][field] = _parse_mcp_result(comp_result, "No comprehensive result")

                    elif operation == "llm_enrich":
                        enrich_result = await self.mcp_session.call_tool("enrich_data_llm", {
                            "data": {field: relevant_sources[0]["value"]},
                            "enrichment_prompt": f"Enrich the {field} field"
                        })
                        results[section][field] = _parse_mcp_result(enrich_result, "No enrichment result")

                    # Support for other operations can be added similarly

                except Exception as e:
                    errors.append({"field": field, "error": str(e)})

        return {
            "final_result": results,
            "errors": errors,
            "source_summary": flat_sources,
            "agent_type": "config_orchestration"
        }
async def process_by_config(self, config: Dict, extracted_data: List[Dict]) -> Dict:
        if not self.mcp_session:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        results = {}
        errors = []

        # Normalize data sources
        source_map = {src["source_id"]: src for src in extracted_data}

        # Prepare flat source list for tools
        flat_sources = [
            {
                "source_id": s["source_id"],
                "doc_type": s["doc_type"],
                "data": s["data"],
                "timestamp": s["metadata"].get("timestamp", ""),
                "confidence": s["metadata"].get("extraction_confidence", 0.0)
            }
            for s in extracted_data
        ]

        for section, operations in config.items():
            results[section] = {}
            for op in operations:
                try:
                    field = op["field_name"]
                    operation = op["operation"]
                    sources_order = op.get("priority_sources", [])
                    dependencies = op.get("dependencies", [])
                    params = op.get("parameters", {})

                    # Get source-specific data for the field
                    relevant_sources = [
                        {
                            "source_id": src["source_id"],
                            "doc_type": src["doc_type"],
                            "value": src["data"].get(field, ""),
                            "metadata": src["metadata"]
                        }
                        for src in extracted_data
                        if src["doc_type"] in sources_order and field in src["data"]
                    ]

                    if not relevant_sources:
                        continue

                    if operation == "fuzzy_match":
                        threshold = params.get("threshold", 0.8)
                        target_value = relevant_sources[0]["value"]

                        match_result = await self.mcp_session.call_tool("fuzzy_match_sources", {
                            "target_value": target_value,
                            "sources": relevant_sources,
                            "field_name": field,
                            "threshold": threshold
                        })
                        results[section][field] = _parse_mcp_result(match_result, "No matches")

                    elif operation == "comprehensive_analysis":
                        comp_result = await self.mcp_session.call_tool("identify_comprehensive_values", {
                            "sources": relevant_sources,
                            "field_name": field
                        })
                        results[section][field] = _parse_mcp_result(comp_result, "No comprehensive result")

                    elif operation == "llm_enrich":
                        enrich_result = await self.mcp_session.call_tool("enrich_data_llm", {
                            "data": {field: relevant_sources[0]["value"]},
                            "enrichment_prompt": f"Enrich the {field} field"
                        })
                        results[section][field] = _parse_mcp_result(enrich_result, "No enrichment result")

                    # Support for other operations can be added similarly

                except Exception as e:
                    errors.append({"field": field, "error": str(e)})

        return {
            "final_result": results,
            "errors": errors,
            "source_summary": flat_sources,
            "agent_type": "config_orchestration"
        }
async def process_by_config(self, config: Dict, extracted_data: List[Dict]) -> Dict:
        if not self.mcp_session:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        results = {}
        errors = []

        # Normalize data sources
        source_map = {src["source_id"]: src for src in extracted_data}

        # Prepare flat source list for tools
        flat_sources = [
            {
                "source_id": s["source_id"],
                "doc_type": s["doc_type"],
                "data": s["data"],
                "timestamp": s["metadata"].get("timestamp", ""),
                "confidence": s["metadata"].get("extraction_confidence", 0.0)
            }
            for s in extracted_data
        ]

        for section, operations in config.items():
            results[section] = {}
            for op in operations:
                try:
                    field = op["field_name"]
                    operation = op["operation"]
                    sources_order = op.get("priority_sources", [])
                    dependencies = op.get("dependencies", [])
                    params = op.get("parameters", {})

                    # Get source-specific data for the field
                    relevant_sources = [
                        {
                            "source_id": src["source_id"],
                            "doc_type": src["doc_type"],
                            "value": src["data"].get(field, ""),
                            "metadata": src["metadata"]
                        }
                        for src in extracted_data
                        if src["doc_type"] in sources_order and field in src["data"]
                    ]

                    if not relevant_sources:
                        continue

                    if operation == "fuzzy_match":
                        threshold = params.get("threshold", 0.8)
                        target_value = relevant_sources[0]["value"]

                        match_result = await self.mcp_session.call_tool("fuzzy_match_sources", {
                            "target_value": target_value,
                            "sources": relevant_sources,
                            "field_name": field,
                            "threshold": threshold
                        })
                        results[section][field] = _parse_mcp_result(match_result, "No matches")

                    elif operation == "comprehensive_analysis":
                        comp_result = await self.mcp_session.call_tool("identify_comprehensive_values", {
                            "sources": relevant_sources,
                            "field_name": field
                        })
                        results[section][field] = _parse_mcp_result(comp_result, "No comprehensive result")

                    elif operation == "llm_enrich":
                        enrich_result = await self.mcp_session.call_tool("enrich_data_llm", {
                            "data": {field: relevant_sources[0]["value"]},
                            "enrichment_prompt": f"Enrich the {field} field"
                        })
                        results[section][field] = _parse_mcp_result(enrich_result, "No enrichment result")

                    # Support for other operations can be added similarly

                except Exception as e:
                    errors.append({"field": field, "error": str(e)})

        return {
            "final_result": results,
            "errors": errors,
            "source_summary": flat_sources,
            "agent_type": "config_orchestration"
        }

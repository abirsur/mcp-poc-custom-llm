#!/usr/bin/env python3
"""
MCP Client with LangChain ReAct Agent for Data Post-Processing
Integrates with Azure OpenAI GPT-4o and uses LangChain's ReAct implementation
Fixed version with proper MCP tool input schema alignment
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Type
import os
from dotenv import load_dotenv

# Azure OpenAI imports
from openai import AsyncAzureOpenAI

# MCP client imports
from mcp import ClientSession, HttpPostTransport, ServerId
from mcp.client.http import http_client

# LangChain imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import BaseModel, Field

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_mcp_result(result: Any, default_message: str) -> str:
    """Helper function to parse MCP tool call results."""
    if result and result.content:
        # Filter for content objects that have a 'text' attribute and join them
        texts = [
            content.text
            for content in result.content
            if hasattr(content, "text") and content.text
        ]
        if texts:
            return "\n".join(texts)
    return default_message


from langchain.schema import LLMResult
from langchain.schema.output import Generation


class AzureOpenAILLM(BaseLLM):
    """LangChain LLM wrapper for Azure OpenAI"""

    azure_client: Any = None
    model_name: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 2000

    def __init__(self, azure_config: Dict[str, str], **kwargs):
        super().__init__(**kwargs)
        self.azure_client = AsyncAzureOpenAI(
            api_key=azure_config["api_key"],
            api_version=azure_config["api_version"],
            azure_endpoint=azure_config["endpoint"],
        )
        self.model_name = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME", "gpt-4o")

    @property
    def _llm_type(self) -> str:
        return "azure_openai"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return asyncio.run(self._acall(prompt, stop, run_manager, **kwargs))

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            response = await self.azure_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Azure OpenAI: {e}")
            return f"Error: {str(e)}"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            output = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=output)])
        return LLMResult(generations=generations)


# Custom LangChain Tools for MCP Server Integration - FIXED VERSIONS


class FuzzyMatchTool(BaseTool):
    """Tool for fuzzy matching across data sources - FIXED"""

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
            # Validate required parameters
            if not target_value:
                return "Error: target_value is required"
            if not sources:
                return "Error: sources list is required"
            if not field_name:
                return "Error: field_name is required"

            result = await self.mcp_session.call_tool(
                "fuzzy_match_sources",
                {
                    "target_value": target_value,
                    "sources": sources,
                    "field_name": field_name,
                    "threshold": threshold,
                },
            )

            return _parse_mcp_result(result, "No matches found")
        except Exception as e:
            logger.error(f"Error in fuzzy matching tool: {e}", exc_info=True)
            return f"Error in fuzzy matching: {str(e)}"


class SelectBestMatchTool(BaseTool):
    """Tool for selecting the best match from fuzzy match results - FIXED"""

    name: str = "select_best_match"
    description: str = "Select the best matching value based on similarity and priority"
    mcp_session: Any = None

    class ArgsSchema(BaseModel):
        matches: List[Dict] = Field(description="List of matches from fuzzy matching")
        priority_list: Optional[List[str]] = Field(
            default=None, description="Priority list for source ranking"
        )

    args_schema: Type[BaseModel] = ArgsSchema

    def _run(
        self, matches: List[Dict], priority_list: Optional[List[str]] = None, **kwargs
    ) -> str:
        return asyncio.run(self._arun(matches, priority_list, **kwargs))

    async def _arun(
        self, matches: List[Dict], priority_list: Optional[List[str]] = None, **kwargs
    ) -> str:
        try:
            # Validate required parameters
            if not matches:
                return "Error: matches list is required"

            # Prepare parameters - priority_list is optional
            params = {"matches": matches}
            if priority_list is not None:
                params["priority_list"] = priority_list

            result = await self.mcp_session.call_tool("select_best_match", params)

            return _parse_mcp_result(result, "No best match selected")
        except Exception as e:
            logger.error(f"Error in select best match tool: {e}", exc_info=True)
            return f"Error in selecting best match: {str(e)}"


class ComprehensiveValuesTool(BaseTool):
    """Tool for identifying comprehensive values - FIXED"""

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
            # Validate required parameters
            if not sources:
                return "Error: sources list is required"
            if not field_name:
                return "Error: field_name is required"

            result = await self.mcp_session.call_tool(
                "identify_comprehensive_values",
                {"sources": sources, "field_name": field_name},
            )

            return _parse_mcp_result(result, "No comprehensive values found")
        except Exception as e:
            logger.error(f"Error in comprehensive values tool: {e}", exc_info=True)
            return f"Error in identifying comprehensive values: {str(e)}"


class EnrichDataTool(BaseTool):
    """Tool for enriching data using LLM - FIXED"""

    name: str = "enrich_data_llm"
    description: str = "Enrich data using LLM with source preservation"
    mcp_session: Any = None

    class ArgsSchema(BaseModel):
        data: Dict = Field(description="Data to enrich")
        enrichment_prompt: str = Field(description="Prompt for enrichment")
        context: Optional[str] = Field(default="", description="Additional context")

    args_schema: Type[BaseModel] = ArgsSchema

    def _run(
        self, data: Dict, enrichment_prompt: str, context: str = "", **kwargs
    ) -> str:
        return asyncio.run(self._arun(data, enrichment_prompt, context, **kwargs))

    async def _arun(
        self, data: Dict, enrichment_prompt: str, context: str = "", **kwargs
    ) -> str:
        try:
            # Validate required parameters
            if not data:
                return "Error: data is required"
            if not enrichment_prompt:
                return "Error: enrichment_prompt is required"

            # Prepare parameters - context is optional
            params = {"data": data, "enrichment_prompt": enrichment_prompt}
            if context:
                params["context"] = context

            result = await self.mcp_session.call_tool("enrich_data_llm", params)

            return _parse_mcp_result(result, "No enrichment performed")
        except Exception as e:
            logger.error(f"Error in data enrichment tool: {e}", exc_info=True)
            return f"Error in data enrichment: {str(e)}"


class ProcessWithPriorityTool(BaseTool):
    """Tool for processing data with priority - FIXED"""

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
            # Validate required parameters
            if not sources:
                return "Error: sources list is required"
            if not priority_list:
                return "Error: priority_list is required"
            if not fields:
                return "Error: fields list is required"

            result = await self.mcp_session.call_tool(
                "process_with_priority",
                {"sources": sources, "priority_list": priority_list, "fields": fields},
            )

            return _parse_mcp_result(result, "No priority processing results")
        except Exception as e:
            logger.error(f"Error in priority processing tool: {e}", exc_info=True)
            return f"Error in priority processing: {str(e)}"


class SetDependenciesTool(BaseTool):
    """Tool for setting field dependencies - FIXED"""

    name: str = "set_field_dependencies"
    description: str = "Set dependencies between fields"
    mcp_session: Any = None

    class ArgsSchema(BaseModel):
        field_name: str = Field(description="Name of the field")
        dependencies: List[str] = Field(description="List of dependency fields")

    args_schema: Type[BaseModel] = ArgsSchema

    def _run(self, field_name: str, dependencies: List[str], **kwargs) -> str:
        return asyncio.run(self._arun(field_name, dependencies, **kwargs))

    async def _arun(self, field_name: str, dependencies: List[str], **kwargs) -> str:
        try:
            # Validate required parameters
            if not field_name:
                return "Error: field_name is required"
            if not dependencies:
                return "Error: dependencies list is required"

            result = await self.mcp_session.call_tool(
                "set_field_dependencies",
                {"field_name": field_name, "dependencies": dependencies},
            )

            return _parse_mcp_result(result, f"Dependencies set for {field_name}")
        except Exception as e:
            logger.error(f"Error in set dependencies tool: {e}", exc_info=True)
            return f"Error in setting dependencies: {str(e)}"


class RAGQueryTool(BaseTool):
    """Tool for RAG-based queries - FIXED"""

    name: str = "rag_query"
    description: str = "Perform RAG-based query for data enrichment"
    mcp_session: Any = None

    class ArgsSchema(BaseModel):
        query: str = Field(description="Query for RAG system")
        raw_data: Dict = Field(description="Raw data for context")
        context_sources: Optional[List[str]] = Field(
            default=None, description="Additional context sources"
        )

    args_schema: Type[BaseModel] = ArgsSchema

    def _run(
        self,
        query: str,
        raw_data: Dict,
        context_sources: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        return asyncio.run(self._arun(query, raw_data, context_sources, **kwargs))

    async def _arun(
        self,
        query: str,
        raw_data: Dict,
        context_sources: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        try:
            # Validate required parameters
            if not query:
                return "Error: query is required"
            if not raw_data:
                return "Error: raw_data is required"

            # Prepare parameters - context_sources is optional
            params = {"query": query, "raw_data": raw_data}
            if context_sources is not None:
                params["context_sources"] = context_sources

            result = await self.mcp_session.call_tool("rag_query", params)

            return _parse_mcp_result(result, "No RAG results")
        except Exception as e:
            logger.error(f"Error in RAG query tool: {e}", exc_info=True)
            return f"Error in RAG query: {str(e)}"


class GenerateSummaryTool(BaseTool):
    """Tool for generating summaries - FIXED"""

    name: str = "generate_summary"
    description: str = "Generate comprehensive summary using RAG and LLM"
    mcp_session: Any = None

    class ArgsSchema(BaseModel):
        data: Dict = Field(description="Data to summarize")
        summary_type: Optional[str] = Field(
            default="comprehensive", description="Type of summary"
        )
        include_sources: Optional[bool] = Field(
            default=True, description="Whether to include sources"
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
            # Validate required parameters
            if not data:
                return "Error: data is required"

            # Prepare parameters - summary_type and include_sources are optional
            params = {"data": data}
            if summary_type != "comprehensive":
                params["summary_type"] = summary_type
            if include_sources != True:
                params["include_sources"] = include_sources

            result = await self.mcp_session.call_tool("generate_summary", params)

            return _parse_mcp_result(result, "No summary generated")
        except Exception as e:
            logger.error(f"Error in generate summary tool: {e}", exc_info=True)
            return f"Error in generating summary: {str(e)}"


class DataProcessingClient:
    """
    Main client for data processing pipeline using LangChain ReAct
    """

    def __init__(self, azure_config: Dict[str, str]):
        self.azure_config = azure_config
        self.llm = None
        self.mcp_session = None
        self.mcp_client = None
        self.tools = []
        self.agent = None
        self.agent_executor = None
        self._session_initialized = False

    async def initialize(self, server_url: str = "http://127.0.0.1:8080"):
        """Initialize the client with Azure OpenAI and MCP server"""
        try:
            # Initialize LLM
            self.llm = AzureOpenAILLM(self.azure_config)

            # Initialize MCP session with proper context management
            server_id = ServerId(url=server_url, name="data-postprocessor")
            transport = HttpPostTransport(server_id)

            # Create the client context manager
            self.mcp_client = http_client(transport)
            # Enter the context properly
            self.mcp_session = await self.mcp_client.__aenter__()
            self._session_initialized = True

            # Initialize tools with MCP session
            self.tools = [
                FuzzyMatchTool(mcp_session=self.mcp_session),
                SelectBestMatchTool(mcp_session=self.mcp_session),
                ComprehensiveValuesTool(mcp_session=self.mcp_session),
                EnrichDataTool(mcp_session=self.mcp_session),
                ProcessWithPriorityTool(mcp_session=self.mcp_session),
                SetDependenciesTool(mcp_session=self.mcp_session),
                RAGQueryTool(mcp_session=self.mcp_session),
                GenerateSummaryTool(mcp_session=self.mcp_session),
            ]
        

            # Create ReAct prompt template
            react_prompt = PromptTemplate.from_template(
                """
                    You are an expert data processing agent. You have access to tools that can help you process data from multiple sources.

                    Your task is to process data using these operations:
                    1. FUZZY MATCHING: Match data across sources
                    2. COMPREHENSIVE VALUE IDENTIFICATION: Find the most complete values  
                    3. LLM ENRICHMENT: Enhance data using language models
                    4. PRIORITY PROCESSING: Handle data based on priority lists
                    5. DEPENDENCY MANAGEMENT: Handle field dependencies
                    6. RAG OPERATIONS: Use retrieval-augmented generation when needed

                    TOOLS:
                    {tools}

                    Use the following format:

                    Question: the input question you must answer
                    Thought: you should always think about what to do
                    Action: the action to take, should be one of [{tool_names}]
                    Action Input: the input to the action
                    Observation: the result of the action
                    ... (this Thought/Action/Action Input/Observation can repeat N times)
                    Thought: I now know the final answer
                    Final Answer: the final answer to the original input question

                    Begin!

                    Question: {input}
                    Thought: {agent_scratchpad}
                    """
            )

            # Create ReAct agent
            self.agent = create_react_agent(self.llm, self.tools, react_prompt)

            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                max_iterations=10,
                handle_parsing_errors=True,
            )

            logger.info(
                "Data Processing Client with LangChain ReAct initialized successfully"
            )

        except Exception as e:
            logger.error(f"Failed to complete initialization: {e}")
            await self.cleanup()
            raise

    async def process_data(
        self,
        prompt: str,
        data_sources: List[Dict],
        raw_data: Dict = None,
        priority_list: List[str] = None,
        processing_options: Dict = None,
    ) -> Dict:
        """
        Main method to process data using LangChain ReAct agent
        """
        if not self.agent_executor:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        # Prepare context for the agent
        context_info = {
            "data_sources": data_sources,
            "raw_data": raw_data or {},
            "priority_list": priority_list or [],
            "processing_options": processing_options or {},
        }

        # Create comprehensive input for the agent
        agent_input = f"""
{prompt}

Available Data Sources ({len(data_sources)} sources):
{json.dumps(data_sources, indent=2)}

Priority List: {priority_list or []}

Raw Data Available: {'Yes' if raw_data else 'No'}

Context Information:
{json.dumps(context_info, indent=2)}

Please process this data according to the requirements and provide a comprehensive result.
"""

        logger.info("Starting data processing with LangChain ReAct agent")

        try:
            # Run the agent with timeout to prevent hanging
            result = await asyncio.wait_for(
                self.agent_executor.ainvoke({"input": agent_input}),
                timeout=300,  # 5 minute timeout
            )

            return {
                "final_result": result.get("output", {}),
                "intermediate_steps": result.get("intermediate_steps", []),
                "context": context_info,
                "agent_type": "langchain_react",
            }

        except asyncio.TimeoutError:
            logger.error("Agent execution timed out after 5 minutes")
            return {
                "error": "Agent execution timed out",
                "context": context_info,
                "agent_type": "langchain_react",
            }
        except Exception as e:
            logger.error(f"Error in LangChain ReAct processing: {e}")
            return {
                "error": str(e),
                "context": context_info,
                "agent_type": "langchain_react",
            }

    async def cleanup(self):
        """Cleanup resources properly"""
        try:
            if self.mcp_client and self._session_initialized:
                # Exit the context manager properly
                await self.mcp_client.__aexit__(None, None, None)
                self._session_initialized = False
                logger.info("MCP session cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            self.mcp_session = None
            self.mcp_client = None


# Context manager wrapper for better resource management
class DataProcessingClientManager:
    """
    Context manager wrapper for DataProcessingClient to ensure proper cleanup
    """

    def __init__(
        self, azure_config: Dict[str, str], server_url: str = "http://127.0.0.1:8080"
    ):
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


# Example usage and testing with improved error handling
async def main(config: Dict, extracted_data: List[Dict]):
    """
    Main function to run the data processing client based on config with proper resource management.
    """
    # Azure OpenAI configuration
    azure_config = {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": "2024-05-01-preview",
    }

    # Check if Azure configuration is available
    if not all(azure_config.values()):
        logger.error("Azure OpenAI environment variables not set.")
        logger.error("Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT.")
        return

    # Use the context manager for proper resource management
    async with DataProcessingClientManager(azure_config) as client:
        try:
            # Transform data sources to the required format
            data_sources = []
            for i, source in enumerate(extracted_data):
                data_sources.append(
                    {
                        "id": source["source_id"],
                        "name": source["doc_type"],
                        "priority": i + 1,
                        "data": source["data"],
                        "timestamp": source.get("metadata", {}).get("timestamp", ""),
                        "confidence": 0.0,
                    }
                )

            # Create a comprehensive prompt for the ReAct agent
            prompt = f"""
Process the data from the provided data sources to determine the best value for each field specified in the configuration.

Configuration:
{json.dumps(config, indent=2)}

PROCESSING REQUIREMENTS:
1. Set field dependencies as defined in the config
2. Use fuzzy matching to find similar values across sources
3. Apply priority-based processing according to the priority list
4. For each field, identify the most comprehensive value
5. Use LLM enrichment when needed for data enhancement
6. Handle any dependencies between fields (e.g., process 'client_name' before 'policy_number')

Your final answer should be a single JSON object containing the processed data for all fields listed in the configuration.
"""

            logger.info("--- Starting LangChain ReAct Data Processing ---")

            # Extract priority list from config if available
            priority_list = []
            if (
                "global_settings" in config
                and "priority_list" in config["global_settings"]
            ):
                priority_list = config["global_settings"]["priority_list"]

            # Process data using LangChain ReAct agent
            result = await client.process_data(
                prompt=prompt,
                data_sources=data_sources,
                priority_list=priority_list,
                raw_data={"extracted": extracted_data, "config": config},
            )

            # Print results
            logger.info("\n--- FINAL PROCESSED DATA ---")
            logger.info(json.dumps(result.get("final_result", {}), indent=2))

            if "intermediate_steps" in result:
                logger.info("\n--- INTERMEDIATE STEPS ---")
                for i, step in enumerate(result["intermediate_steps"]):
                    logger.info(f"Step {i + 1}: {step}")

            return result

        except Exception as e:
            logger.error(f"An error occurred during processing: {e}", exc_info=True)
            return {"error": str(e)}


# Alternative direct usage without context manager (for compatibility)
async def main_direct(config: Dict, extracted_data: List[Dict]):
    """
    Alternative main function using direct client management
    """
    # Azure OpenAI configuration
    azure_config = {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": "2024-05-01-preview",
    }

    if not all(azure_config.values()):
        logger.error("Azure OpenAI environment variables not set.")
        return

    client = DataProcessingClient(azure_config)

    try:
        # Initialize client with better error handling
        await client.initialize()
        logger.info("Client initialized successfully")

        # Transform data sources
        data_sources = []
        for i, source in enumerate(extracted_data):
            data_sources.append(
                {
                    "id": source["source_id"],
                    "name": source["doc_type"],
                    "priority": i + 1,
                    "data": source["data"],
                    "timestamp": source.get("metadata", {}).get("timestamp", ""),
                    "confidence": 0.0,
                }
            )

        # Create prompt
        prompt = f"""
Process the data from the provided data sources to determine the best value for each field specified in the configuration.

Configuration:
{json.dumps(config, indent=2)}

PROCESSING REQUIREMENTS:
1. Set field dependencies as defined in the config
2. Use fuzzy matching to find similar values across sources
3. Apply priority-based processing according to the priority list
4. For each field, identify the most comprehensive value
5. Use LLM enrichment when needed for data enhancement
6. Handle any dependencies between fields

Your final answer should be a single JSON object containing the processed data for all fields listed in the configuration.
"""

        # Extract priority list
        priority_list = []
        if "global_settings" in config and "priority_list" in config["global_settings"]:
            priority_list = config["global_settings"]["priority_list"]

        # Process data
        result = await client.process_data(
            prompt=prompt,
            data_sources=data_sources,
            priority_list=priority_list,
            raw_data={"extracted": extracted_data, "config": config},
        )

        logger.info("Processing completed successfully")
        return result

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        return {"error": str(e)}

    finally:
        # Ensure cleanup happens
        try:
            await client.cleanup()
            logger.info("Client cleanup completed")
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")


if __name__ == "__main__":
    # To run this file directly, you can use:
    # with open("config.json", "r") as f:
    #     config_data = json.load(f)
    # with open("extracted_data.json", "r") as f:
    #     extracted_data_json = json.load(f)
    # asyncio.run(main(config_data, extracted_data_json))
    pass

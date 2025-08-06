#!/usr/bin/env python3
"""
MCP Client with LangChain ReAct Agent for Data Post-Processing
Integrates with Azure OpenAI GPT-4o and uses LangChain's ReAct implementation
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
        texts = [content.text for content in result.content if hasattr(content, 'text') and content.text]
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
            azure_endpoint=azure_config["endpoint"]
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
                **kwargs
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


# Custom LangChain Tools for MCP Server Integration

class FuzzyMatchTool(BaseTool):
    """Tool for fuzzy matching across data sources"""
    name: str = "fuzzy_match_sources"
    description: str = "Perform fuzzy matching across multiple data sources for a specific field"
    mcp_session: Any = None
    
    class ArgsSchema(BaseModel):
        target_value: str = Field(description="Target value to match against (can be empty for auto-detection)")
        field_name: str = Field(description="Name of the field to match")
        threshold: float = Field(default=0.8, description="Similarity threshold (0.0-1.0)")
        sources: List[Dict] = Field(description="List of data sources to search")
    
    args_schema: Type[BaseModel] = ArgsSchema
    
    def _run(self, target_value: str, field_name: str, threshold: float = 0.8, sources: List[Dict] = None, **kwargs) -> str:
        """Run the fuzzy match tool"""
        return asyncio.run(self._arun(target_value, field_name, threshold, sources, **kwargs))
    
    async def _arun(self, target_value: str, field_name: str, threshold: float = 0.8, sources: List[Dict] = None, **kwargs) -> str:
        """Run the fuzzy match tool asynchronously"""
        try:
            result = await self.mcp_session.call_tool("fuzzy_match_sources", {
                "target_value": target_value,
                "field_name": field_name,
                "threshold": threshold,
                "sources": sources or []
            })
            
            return _parse_mcp_result(result, "No matches found")
        except Exception as e:
            logger.error(f"Error in fuzzy matching tool: {e}", exc_info=True)
            return f"Error in fuzzy matching: {str(e)}"

class SelectBestMatchTool(BaseTool):
    """Tool for selecting the best match from fuzzy match results"""
    name: str = "select_best_match"
    description: str = "Select the best matching value based on similarity and priority"
    mcp_session: Any = None
    
    class ArgsSchema(BaseModel):
        matches: List[Dict] = Field(description="List of matches from fuzzy matching")
        priority_list: Optional[List[str]] = Field(default=None, description="Priority list for source ranking")
    
    args_schema: Type[BaseModel] = ArgsSchema
    
    def _run(self, matches: List[Dict], priority_list: Optional[List[str]] = None, **kwargs) -> str:
        return asyncio.run(self._arun(matches, priority_list, **kwargs))
    
    async def _arun(self, matches: List[Dict], priority_list: Optional[List[str]] = None, **kwargs) -> str:
        try:
            result = await self.mcp_session.call_tool("select_best_match", {
                "matches": matches,
                "priority_list": priority_list or []
            })
            
            return _parse_mcp_result(result, "No best match selected")
        except Exception as e:
            logger.error(f"Error in select best match tool: {e}", exc_info=True)
            return f"Error in selecting best match: {str(e)}"

class ComprehensiveValuesTool(BaseTool):
    """Tool for identifying comprehensive values"""
    name: str = "identify_comprehensive_values"
    description: str = "Identify comprehensive values with source tracking for a specific field"
    mcp_session: Any = None
    
    class ArgsSchema(BaseModel):
        sources: List[Dict] = Field(description="List of data sources")
        field_name: str = Field(description="Name of the field to analyze")
    
    args_schema: Type[BaseModel] = ArgsSchema
    
    def _run(self, sources: List[Dict], field_name: str, **kwargs) -> str:
        return asyncio.run(self._arun(sources, field_name, **kwargs))
    
    async def _arun(self, sources: List[Dict], field_name: str, **kwargs) -> str:
        try:
            result = await self.mcp_session.call_tool("identify_comprehensive_values", {
                "sources": sources,
                "field_name": field_name
            })
            
            return _parse_mcp_result(result, "No comprehensive values found")
        except Exception as e:
            logger.error(f"Error in comprehensive values tool: {e}", exc_info=True)
            return f"Error in identifying comprehensive values: {str(e)}"

class EnrichDataTool(BaseTool):
    """Tool for enriching data using LLM"""
    name: str = "enrich_data_llm"
    description: str = "Enrich data using LLM with source preservation"
    mcp_session: Any = None
    
    class ArgsSchema(BaseModel):
        data: Dict = Field(description="Data to enrich")
        enrichment_prompt: str = Field(description="Prompt for enrichment")
        context: Optional[str] = Field(default="", description="Additional context")
    
    args_schema: Type[BaseModel] = ArgsSchema
    
    def _run(self, data: Dict, enrichment_prompt: str, context: str = "", **kwargs) -> str:
        return asyncio.run(self._arun(data, enrichment_prompt, context, **kwargs))
    
    async def _arun(self, data: Dict, enrichment_prompt: str, context: str = "", **kwargs) -> str:
        try:
            result = await self.mcp_session.call_tool("enrich_data_llm", {
                "data": data,
                "enrichment_prompt": enrichment_prompt,
                "context": context
            })
            
            return _parse_mcp_result(result, "No enrichment performed")
        except Exception as e:
            logger.error(f"Error in data enrichment tool: {e}", exc_info=True)
            return f"Error in data enrichment: {str(e)}"

class ProcessWithPriorityTool(BaseTool):
    """Tool for processing data with priority"""
    name: str = "process_with_priority"
    description: str = "Process data according to priority list"
    mcp_session: Any = None
    
    class ArgsSchema(BaseModel):
        sources: List[Dict] = Field(description="List of data sources")
        priority_list: List[str] = Field(description="Priority list for source ranking")
        fields: List[str] = Field(description="List of fields to process")
    
    args_schema: Type[BaseModel] = ArgsSchema
    
    def _run(self, sources: List[Dict], priority_list: List[str], fields: List[str], **kwargs) -> str:
        return asyncio.run(self._arun(sources, priority_list, fields, **kwargs))
    
    async def _arun(self, sources: List[Dict], priority_list: List[str], fields: List[str], **kwargs) -> str:
        try:
            result = await self.mcp_session.call_tool("process_with_priority", {
                "sources": sources,
                "priority_list": priority_list,
                "fields": fields
            })
            
            return _parse_mcp_result(result, "No priority processing results")
        except Exception as e:
            logger.error(f"Error in priority processing tool: {e}", exc_info=True)
            return f"Error in priority processing: {str(e)}"

class SetDependenciesTool(BaseTool):
    """Tool for setting field dependencies"""
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
            result = await self.mcp_session.call_tool("set_field_dependencies", {
                "field_name": field_name,
                "dependencies": dependencies
            })
            
            return _parse_mcp_result(result, f"Dependencies set for {field_name}")
        except Exception as e:
            logger.error(f"Error in set dependencies tool: {e}", exc_info=True)
            return f"Error in setting dependencies: {str(e)}"

class RAGQueryTool(BaseTool):
    """Tool for RAG-based queries"""
    name: str = "rag_query"
    description: str = "Perform RAG-based query for data enrichment"
    mcp_session: Any = None
    
    class ArgsSchema(BaseModel):
        query: str = Field(description="Query for RAG system")
        raw_data: Dict = Field(description="Raw data for context")
        context_sources: Optional[List[str]] = Field(default=None, description="Additional context sources")
    
    args_schema: Type[BaseModel] = ArgsSchema
    
    def _run(self, query: str, raw_data: Dict, context_sources: Optional[List[str]] = None, **kwargs) -> str:
        return asyncio.run(self._arun(query, raw_data, context_sources, **kwargs))
    
    async def _arun(self, query: str, raw_data: Dict, context_sources: Optional[List[str]] = None, **kwargs) -> str:
        try:
            result = await self.mcp_session.call_tool("rag_query", {
                "query": query,
                "raw_data": raw_data,
                "context_sources": context_sources or []
            })
            
            return _parse_mcp_result(result, "No RAG results")
        except Exception as e:
            logger.error(f"Error in RAG query tool: {e}", exc_info=True)
            return f"Error in RAG query: {str(e)}"

class GenerateSummaryTool(BaseTool):
    """Tool for generating summaries"""
    name: str = "generate_summary"
    description: str = "Generate comprehensive summary using RAG and LLM"
    mcp_session: Any = None
    
    class ArgsSchema(BaseModel):
        data: Dict = Field(description="Data to summarize")
        summary_type: Optional[str] = Field(default="comprehensive", description="Type of summary")
        include_sources: Optional[bool] = Field(default=True, description="Whether to include sources")
    
    args_schema: Type[BaseModel] = ArgsSchema
    
    def _run(self, data: Dict, summary_type: str = "comprehensive", include_sources: bool = True, **kwargs) -> str:
        return asyncio.run(self._arun(data, summary_type, include_sources, **kwargs))
    
    async def _arun(self, data: Dict, summary_type: str = "comprehensive", include_sources: bool = True, **kwargs) -> str:
        try:
            result = await self.mcp_session.call_tool("generate_summary", {
                "data": data,
                "summary_type": summary_type,
                "include_sources": include_sources
            })
            
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
        self.tools = []
        self.agent = None
        self.agent_executor = None
    
    async def initialize(self, server_url: str = "http://127.0.0.1:8080"):
        """Initialize the client with Azure OpenAI and MCP server"""
        # Initialize LLM
        self.llm = AzureOpenAILLM(self.azure_config)
        
        # Initialize MCP session
        server_id = ServerId(
            url=server_url,
            name="data-postprocessor"
        )
        transport = HttpPostTransport(server_id)
        self.mcp_session = await http_client(transport).__aenter__()
        
        # Initialize tools with MCP session
        self.tools = [
            FuzzyMatchTool(mcp_session=self.mcp_session),
            SelectBestMatchTool(mcp_session=self.mcp_session),
            ComprehensiveValuesTool(mcp_session=self.mcp_session),
            EnrichDataTool(mcp_session=self.mcp_session),
            ProcessWithPriorityTool(mcp_session=self.mcp_session),
            SetDependenciesTool(mcp_session=self.mcp_session),
            RAGQueryTool(mcp_session=self.mcp_session),
            GenerateSummaryTool(mcp_session=self.mcp_session)
        ]
        
        # Create ReAct prompt template
        react_prompt = PromptTemplate.from_template("""
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
""")
        
        # Create ReAct agent
        self.agent = create_react_agent(self.llm, self.tools, react_prompt)
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )
        
        logger.info("Data Processing Client with LangChain ReAct initialized successfully")
    
    async def process_data(self, 
                          prompt: str,
                          data_sources: List[Dict],
                          raw_data: Dict = None,
                          priority_list: List[str] = None,
                          processing_options: Dict = None) -> Dict:
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
            "processing_options": processing_options or {}
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
            # Run the agent
            result = await self.agent_executor.ainvoke({"input": agent_input})
            
            return {
                "final_result": result.get("output", {}),
                "intermediate_steps": result.get("intermediate_steps", []),
                "context": context_info,
                "agent_type": "langchain_react"
            }
            
        except Exception as e:
            logger.error(f"Error in LangChain ReAct processing: {e}")
            return {
                "error": str(e),
                "context": context_info,
                "agent_type": "langchain_react"
            }
    
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
                "priority": i + 1,
                "data": source["data"],
                "timestamp": source.get("metadata", {}).get("timestamp", ""),
                "confidence": 0.0
            })

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
        if "global_settings" in config and "priority_list" in config["global_settings"]:
            priority_list = config["global_settings"]["priority_list"]
        
        # Process data using LangChain ReAct agent
        result = await client.process_data(
            prompt=prompt,
            data_sources=data_sources,
            priority_list=priority_list,
            raw_data={"extracted": extracted_data, "config": config}
        )
        
        # Print results
        logger.info("\n--- FINAL PROCESSED DATA ---")
        logger.info(json.dumps(result.get('final_result', {}), indent=2))
        
        if 'intermediate_steps' in result:
            logger.info("\n--- INTERMEDIATE STEPS ---")
            for i, step in enumerate(result['intermediate_steps']):
                logger.info(f"Step {i + 1}: {step}")
        
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


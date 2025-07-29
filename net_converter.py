import os
import subprocess
import logging
from typing import List, Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_community.chat_models import AzureChatOpenAI
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
import json


# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("conversion_report.log"),
        logging.StreamHandler()
    ]
)


# 1. Pydantic Model for Grounding Requirements
class CodeConversionRequest(BaseModel):
    """Defines the structure for a code conversion request."""
    source_code: str = Field(description="The source code to be converted.")
    source_framework: str = Field(default=".NET Framework 4.0", description="The source .NET framework version.")
    target_framework: str = Field(description="The target .NET framework version.")
    file_path: str = Field(description="The original file path of the code being converted.")


class CodeCorrectionRequest(BaseModel):
    """Defines the structure for a code correction request."""
    failed_code: str = Field(description="The generated C# code that failed validation.")
    error_feedback: str = Field(description="The validation errors that need to be fixed.")
    target_framework: str = Field(description="The target .NET framework version.")


class CodeValidationResponse(BaseModel):
    """Defines the structure for the validation result."""
    is_valid: bool = Field(description="True if the code is valid, False otherwise.")
    errors: Optional[str] = Field(description="A detailed description of validation errors, if any.")

    @field_validator('errors', mode='before')
    def empty_str_to_none(cls, v):
        if isinstance(v, str) and not v.strip():
            return None
        return v


class CodeConversionResponse(BaseModel):
    """Defines the structure for the converted code response."""
    converted_code: str = Field(description="The converted C# code.")
    new_file_path: str = Field(description="The suggested new file path in the .NET Core structure.")
    success: bool = Field(description="Indicates if the conversion was successful.")
    error_message: Optional[str] = Field(default=None, description="Error message if conversion failed.")


# 2. LLM Chain for Code Conversion
def get_conversion_chain() -> LLMChain:
    """Initializes and returns the LLMChain for the initial code conversion."""
    prompt_template = """
    You are an expert .NET developer specializing in framework migration. Your **critical and only task** is to convert the following C# code from {source_framework} into a valid component for a **{target_framework} Web API**.

    **Conversion Rules (Non-negotiable):**
    1.  **Web API Component MANDATORY:** The output code MUST be a valid Web API component. This means it must be a Controller, a Minimal API endpoint, a service class designed for dependency injection, a model, or part of the `Program.cs` setup for a Web API. **Generating any other type of code (e.g., a class with a `Main` method) is a failure.**
    2.  **Convert Entry Points:** If the source code contains an application entry point (like a `Main` method or `Global.asax`), you MUST convert it into a suitable Web API structure. For example, a console application's logic should be refactored into one or more API endpoints in a Controller.
    3.  **Modern Patterns:** You MUST use the latest C# and {target_framework} Web API patterns (e.g., Dependency Injection, async/await, `[ApiController]` attribute).
    4.  **Code Only:** Your response MUST contain ONLY the raw C# code for the converted file. Do NOT include any explanations or markdown formatting.
    5.  **File Path:** Based on the original file path (`{file_path}`), suggest a new path and filename appropriate for a standard {target_framework} Web API project. The path should be a comment on the first line, like this: `// New Path: Controllers/MyController.cs`

    **Original File Path:** {file_path}
    **Source Code ({source_framework}):**
    ```csharp
    {source_code}
    ```

    **Converted Code ({target_framework}):**
    """
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.1,
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["source_code", "source_framework", "target_framework", "file_path"]
    )
    return LLMChain(llm=llm, prompt=prompt)


def get_validation_chain() -> LLMChain:
    """Initializes and returns the LLMChain for code validation."""
    parser = PydanticOutputParser(pydantic_object=CodeValidationResponse)

    prompt_template = """
    You are an extremely strict .NET code reviewer. Your task is to validate if the following C# code is a valid component for a **{target_framework} Web API**.

    **Validation Criteria (Must Pass All):**
    1.  **CRITICAL: Web API Suitability:** The code's primary purpose MUST be to function within a .NET Core Web API. It must be a Controller, a minimal API endpoint, a service, a model, or Web API configuration. If it contains a `Main` method that is not part of the standard Web API `Program.cs` template, it is **invalid**. Any code that looks like it belongs in a Console App or a different project type is **invalid**.
    2.  **Valid Syntax & Structure:** The code must be syntactically correct and structurally sound. It should be "compilation-ready" from a code-quality perspective.
    3.  **Ignore Missing References:** You MUST ignore errors related to missing package references (e.g., `using` statements for libraries that are not yet installed). Your validation should focus only on the quality and correctness of the code provided.
    4.  **Completeness:** The code must be a complete, valid file, not a partial snippet.
    5.  **No Placeholders:** The code must not contain placeholder comments like "// TODO" or "// Implement here".

    Analyze the code and provide your response in the following JSON format:
    {format_instructions}

    **C# Code to Validate:**
    ```csharp
    {source_code}
    ```
    """

    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}}, # Enforce JSON output
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["source_code", "target_framework"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    return LLMChain(llm=llm, prompt=prompt, output_parser=parser)


def get_correction_chain() -> LLMChain:
    """Initializes and returns the LLMChain for correcting faulty code."""
    prompt_template = """
    You are a senior .NET developer tasked with fixing code that has failed validation.
    Your **primary goal** is to rewrite the code to be a valid **{target_framework} Web API** component, correcting the specific errors provided.

    **Target Architecture:** {target_framework} Web API. This is not optional.

    **Validation Errors to Fix:**
    {error_feedback}

    **Faulty C# Code:**
    ```csharp
    {failed_code}
    ```

    **Instructions:**
    -   Analyze the faulty code and the validation errors.
    -   Rewrite the code to fix the identified issues.
    -   Your response MUST contain ONLY the raw, corrected C# code.
    -   Do NOT include any explanations, markdown, or any text other than the code itself.

    **Corrected C# Code:**
    """
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.2, # Give it some creativity to fix issues
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["failed_code", "error_feedback", "target_framework"]
    )
    return LLMChain(llm=llm, prompt=prompt)


# 3. Three-Phase Conversion, Validation, and Correction
def process_file(
    conversion_chain: LLMChain,
    validation_chain: LLMChain,
    correction_chain: LLMChain,
    request: CodeConversionRequest,
    new_project_path: str,
    max_retries: int
) -> CodeConversionResponse:
    """
    Orchestrates the three-phase process of conversion, validation, and correction.
    """
    try:
        ###
        # ### PHASE 1: INITIAL CONVERSION ###
        # This step runs only once to generate the initial version of the code.
        ###
        logging.info("### PHASE 1: Running initial code conversion... ###")
        raw_llm_response = conversion_chain.run(request.dict())
        raw_llm_response = raw_llm_response.strip()
        lines = raw_llm_response.split('\n')

        if lines and lines[0].strip().startswith("// New Path:"):
            suggested_relative_path = lines[0].split('// New Path: ')[1].strip()
            new_file_path = os.path.join(new_project_path, suggested_relative_path)
            generated_code = "\n".join(lines[1:])
        else:
            original_filename = os.path.basename(request.file_path)
            new_file_path = os.path.join(new_project_path, original_filename)
            generated_code = raw_llm_response

        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        with open(new_file_path, "w", encoding="utf-8") as f:
            f.write(generated_code)
        logging.info(f"Initial code saved to '{new_file_path}'")

    except Exception as e:
        logging.error(f"Fatal error during Phase 1 (Conversion): {e}")
        return CodeConversionResponse(converted_code="", new_file_path="", success=False, error_message=str(e))

    # This loop manages the iterative validation and correction process.
    last_error = ""
    for attempt in range(max_retries):
        logging.info(f"--- Validation/Correction Loop: Attempt {attempt + 1}/{max_retries} ---")
        try:
            ###
            # ### PHASE 2: VALIDATE THE CURRENT CODE ###
            # This step reads the latest version of the code and validates it.
            ###
            logging.info("### PHASE 2: Performing AI validation... ###")
            with open(new_file_path, "r", encoding="utf-8") as f:
                current_code = f.read()

            validation_result: CodeValidationResponse = validation_chain.run({
                "source_code": current_code,
                "target_framework": request.target_framework
            })

            # If validation passes, the process for this file is successful.
            if validation_result.is_valid:
                logging.info("Validation successful. File processing complete.")
                return CodeConversionResponse(
                    converted_code=current_code,
                    new_file_path=new_file_path,
                    success=True
                )
            
            last_error = validation_result.errors
            logging.warning(f"Validation failed. Errors: {last_error}")

            ###
            # ### PHASE 3: CORRECT THE CODE ###
            # If validation fails, this step attempts to fix the errors.
            ###
            logging.info("### PHASE 3: Attempting to correct the code... ###")
            correction_request = CodeCorrectionRequest(
                failed_code=current_code,
                error_feedback=last_error,
                target_framework=request.target_framework
            )
            corrected_code = correction_chain.run(correction_request.dict())

            # The corrected code overwrites the old file and the loop continues,
            # sending the new code back to Phase 2 for re-validation.
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(corrected_code)
            logging.info(f"Corrected code saved. Returning to Phase 2 for re-validation.")

        except json.JSONDecodeError as e:
            last_error = f"Error decoding validation response: {e}. The LLM did not return valid JSON."
            logging.error(last_error)
        except Exception as e:
            last_error = f"An unexpected error occurred during validation/correction: {e}"
            logging.error(last_error)
            # If a critical error occurs, stop retrying for this file.
            break

    logging.error(f"Max retries reached for {request.file_path}. Could not fix the code.")
    return CodeConversionResponse(
        converted_code="",
        new_file_path=new_file_path,
        success=False,
        error_message=f"Max retries reached. Last error: {last_error}"
    )


# 4. Main Orchestration
def convert_project(source_directory: str, target_directory: str, target_framework: str, max_retries: int):
    """
    Orchestrates the conversion of a .NET Framework project.
    """
    if not os.path.isdir(source_directory):
        logging.error(f"Error: Source directory '{source_directory}' not found.")
        return

    # Create a new .NET Core Web API project structure
    try:
        framework_version_for_cli = target_framework.replace(".NET Core ", "net")
        project_name = os.path.basename(os.path.normpath(target_directory))
        subprocess.run(["dotnet", "new", "webapi", "-n", project_name, "-o", target_directory, "--framework", framework_version_for_cli], check=True)
        logging.info(f"Created new {target_framework} Web API project at '{target_directory}'")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to create new .NET project. Is the .NET SDK installed and in your PATH? Error: {e}")
        return

    conversion_chain = get_conversion_chain()
    validation_chain = get_validation_chain()
    correction_chain = get_correction_chain()

    # Walk through the old directory
    for root, _, files in os.walk(source_directory):
        for file in files:
            if file.endswith(".cs"):
                file_path = os.path.join(root, file)
                logging.info(f"\n=================================================")
                logging.info(f"Processing: {file_path}")
                logging.info(f"=================================================")
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    source_code = f.read()

                request = CodeConversionRequest(
                    source_code=source_code,
                    file_path=file_path,
                    target_framework=target_framework
                )

                result = process_file(
                    conversion_chain=conversion_chain,
                    validation_chain=validation_chain,
                    correction_chain=correction_chain,
                    request=request,
                    new_project_path=target_directory,
                    max_retries=max_retries
                )

                if result.success:
                    logging.info(f"SUCCESS: Converted '{file_path}' -> '{result.new_file_path}'")
                else:
                    logging.error(f"FAILURE: Failed to convert '{file_path}'. Last error: {result.error_message}")

    logging.info("\n--- Conversion process completed. ---")
    logging.info(f"Final project available at: {target_directory}")
    logging.info("Review 'conversion_report.log' for a detailed summary of the process.")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    OLD_PROJECT_DIR = "C:/path/to/your/old/dotnet_framework_app"
    NEW_PROJECT_DIR = "./converted_net_core_app"

    # Load settings from environment file
    TARGET_FRAMEWORK = os.getenv("TARGET_FRAMEWORK", ".NET Core 8")
    try:
        MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
    except (ValueError, TypeError):
        logging.warning("Invalid MAX_RETRIES in .env file. Defaulting to 3.")
        MAX_RETRIES = 3

    logging.info("--- Starting .NET Code Conversion Process ---")
    logging.info(f"Target Framework: {TARGET_FRAMEWORK}")
    logging.info(f"Max Retries per File: {MAX_RETRIES}")

    if "C:/path/to/your/old/dotnet_framework_app" in OLD_PROJECT_DIR:
        logging.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.warning("!!! PLEASE UPDATE 'OLD_PROJECT_DIR' TO YOUR PROJECT. !!!")
        logging.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        convert_project(OLD_PROJECT_DIR, NEW_PROJECT_DIR, TARGET_FRAMEWORK, MAX_RETRIES)

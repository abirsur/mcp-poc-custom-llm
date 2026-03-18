import os
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from file_zip_base64_service import FileZipBase64Service


SERVER_NAME = "file-zip-base64-mcp"
MCP_PATH = os.getenv("MCP_PATH", "/mcp")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


STATELESS_HTTP = _get_env_bool("STATELESS_HTTP", False)

service = FileZipBase64Service()

mcp = FastMCP(
    name=SERVER_NAME,
    instructions=(
        "Tools for zipping files by extension into base64-encoded zip payloads and "
        "restoring base64 zip payloads back into extracted files."
    ),
)


@mcp.tool
def zip_files_by_extension(
    directory_path: Annotated[
        str,
        Field(description="Directory to scan recursively for matching files."),
    ],
    extension: Annotated[
        str,
        Field(description="File extension to match, for example '.txt' or 'pdf'."),
    ],
) -> list[str]:
    """Zip each matching file individually and return base64-encoded zip archives."""
    return service.zip_files_by_extension(directory_path=directory_path, extension=extension)


@mcp.tool
def unzip_base64_files_to_temp(
    base64_zip_files: Annotated[
        list[str],
        Field(description="Array of base64-encoded zip archives."),
    ],
) -> list[str]:
    """Decode base64 zip archives, extract them into temp folders, and return file paths."""
    return service.unzip_base64_files_to_temp(base64_zip_files=base64_zip_files)


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(_: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


@mcp.custom_route("/readyz", methods=["GET"])
async def readyz(_: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ready",
            "server": SERVER_NAME,
            "transport": "streamable-http",
            "mcp_path": MCP_PATH,
            "stateless_http": STATELESS_HTTP,
        }
    )


app = mcp.http_app(
    path=MCP_PATH,
    transport="streamable-http",
    stateless_http=STATELESS_HTTP,
)


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host=HOST,
        port=PORT,
        path=MCP_PATH,
        stateless_http=STATELESS_HTTP,
        show_banner=False,
    )

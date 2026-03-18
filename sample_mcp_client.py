import argparse
import base64
import io
import shutil
import zipfile
from pathlib import Path

import anyio
from mcp_client import MCPClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample MCP client for the file zip/base64 server.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/mcp",
        help="MCP server URL.",
    )
    parser.add_argument(
        "--workdir",
        default="client_test_data",
        help="Directory used for sample input and output files.",
    )
    parser.add_argument(
        "--extension",
        default=".txt",
        help="Extension to use for the zip test.",
    )
    return parser.parse_args()


def prepare_test_files(workdir: Path, extension: str) -> Path:
    if workdir.exists():
        shutil.rmtree(workdir)

    source_dir = workdir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / f"example_1{extension}").write_text("hello from MCP\n", encoding="utf-8")
    (source_dir / f"example_2{extension}").write_text("second file\n", encoding="utf-8")
    (source_dir / "ignore.me").write_text("not included\n", encoding="utf-8")
    return source_dir


def summarize_zip_payloads(payloads: list[str]) -> list[str]:
    summaries: list[str] = []
    for index, payload in enumerate(payloads, start=1):
        zip_bytes = base64.b64decode(payload)
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zip_file:
            summaries.append(f"archive_{index}: {zip_file.namelist()}")
    return summaries


async def run_demo(url: str, source_dir: Path, extension: str) -> None:
    client = MCPClient(url)

    server_info = await client.get_server_info()
    print(f"Connected to: {server_info['name']} ({server_info['version']})")

    tools = await client.list_tools()
    print("Available tools:")
    for tool in tools:
        print(f"  - {tool['name']}")

    zip_result = await client.call_tool(
        "zip_files_by_extension",
        {
            "directory_path": str(source_dir),
            "extension": extension,
        },
    )
    if zip_result["is_error"]:
        raise RuntimeError(f"zip_files_by_extension failed: {zip_result['content']}")

    payloads = zip_result["structured_content"]["result"]
    print(f"Created {len(payloads)} base64 zip payload(s)")
    for summary in summarize_zip_payloads(payloads):
        print(f"  {summary}")

    unzip_result = await client.call_tool(
        "unzip_base64_files_to_temp",
        {
            "base64_zip_files": payloads,
        },
    )
    if unzip_result["is_error"]:
        raise RuntimeError(f"unzip_base64_files_to_temp failed: {unzip_result['content']}")

    restored_paths = unzip_result["structured_content"]["result"]
    print("Extracted files:")
    for restored_path in restored_paths:
        print(f"  - {restored_path}")


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    source_dir = prepare_test_files(workdir, args.extension)
    anyio.run(run_demo, args.url, source_dir, args.extension)


if __name__ == "__main__":
    main()

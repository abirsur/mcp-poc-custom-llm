import base64
import io
import tempfile
import uuid
import zipfile
from pathlib import Path


class FileZipBase64Service:
    def zip_files_by_extension(self, directory_path: str, extension: str) -> list[str]:
        """
        Find all files in the given directory tree matching the extension,
        zip each file individually in memory, and return a base64 string
        for each zip archive.
        """
        root = Path(directory_path)
        if not root.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory_path}")
        if not root.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory_path}")

        normalized_extension = extension if extension.startswith(".") else f".{extension}"
        matched_files = [
            path for path in root.rglob(f"*{normalized_extension}") if path.is_file()
        ]

        zipped_base64_files: list[str] = []
        for file_path in matched_files:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
                # Preserve the relative path inside the archive when possible.
                zip_file.write(file_path, arcname=file_path.relative_to(root))

            zipped_base64_files.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

        return zipped_base64_files

    def unzip_base64_files_to_temp(self, base64_zip_files: list[str]) -> list[str]:
        """
        Decode each base64 zip archive, extract its contents into a unique temp
        directory, and return the extracted file paths.
        """
        extracted_file_paths: list[str] = []

        for index, base64_zip in enumerate(base64_zip_files):
            zip_bytes = base64.b64decode(base64_zip)
            temp_directory = self._create_temp_directory(index)

            with zipfile.ZipFile(io.BytesIO(zip_bytes), mode="r") as zip_file:
                try:
                    extracted_paths = self._extract_zip_file(zip_file, temp_directory)
                except PermissionError:
                    temp_directory = self._create_temp_directory(index, use_local_fallback=True)
                    extracted_paths = self._extract_zip_file(zip_file, temp_directory)

                extracted_file_paths.extend(str(path) for path in extracted_paths)

        return extracted_file_paths

    def _create_temp_directory(self, index: int, use_local_fallback: bool = False) -> Path:
        if not use_local_fallback:
            return Path(tempfile.mkdtemp(prefix=f"unzipped_{index}_"))

        local_temp_root = Path.cwd() / ".temp_unzipped"
        local_temp_root.mkdir(exist_ok=True)
        temp_directory = local_temp_root / f"unzipped_{index}_{uuid.uuid4().hex}"
        temp_directory.mkdir(parents=True, exist_ok=True)
        return temp_directory

    def _extract_zip_file(self, zip_file: zipfile.ZipFile, target_directory: Path) -> list[Path]:
        extracted_paths: list[Path] = []

        for member in zip_file.infolist():
            if member.is_dir():
                continue

            extracted_path = target_directory / member.filename
            extracted_path.parent.mkdir(parents=True, exist_ok=True)

            with zip_file.open(member, mode="r") as source, extracted_path.open("wb") as target:
                target.write(source.read())

            extracted_paths.append(extracted_path)

        return extracted_paths

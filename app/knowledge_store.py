"""Utilities for working with paths inside the virtual knowledge space."""


def normalize_virtual_path(path: str) -> str:
    """Validate a virtual POSIX path and return its canonical form.

    This function only handles path syntax. It does not access the filesystem,
    check whether the path exists, or map the path to ``KNOWLEDGE_ROOT``.
    """

    if not isinstance(path, str):
        raise TypeError("virtual path must be a string")
    if not path:
        raise ValueError("virtual path must not be empty")
    if "\x00" in path:
        raise ValueError("virtual path must not contain a NUL character")
    if "\\" in path:
        raise ValueError("virtual path must use forward slashes")
    if not path.startswith("/"):
        raise ValueError("virtual path must start with '/'")
    if "//" in path:
        raise ValueError("virtual path must not contain repeated slashes")

    canonical_path = path[:-1] if path != "/" and path.endswith("/") else path
    parts = canonical_path.split("/")[1:]

    if any(part in {".", ".."} for part in parts):
        raise ValueError("virtual path must not contain '.' or '..' segments")

    first_part = parts[0] if parts else ""
    if (
        len(first_part) >= 2
        and first_part[0].isascii()
        and first_part[0].isalpha()
        and first_part[1] == ":"
    ):
        raise ValueError("virtual path must not contain a Windows drive")

    return canonical_path

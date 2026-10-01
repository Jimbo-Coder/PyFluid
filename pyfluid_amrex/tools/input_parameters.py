"""Read AMReX assignment files for reporting and generated validation inputs.

AMReX owns expression evaluation and solver configuration. This reader preserves
serialized values for rewriting and decodes quoted tokens for Python consumers.
It does not evaluate expressions, includes, or preprocessor directives.
"""

from pathlib import Path
import re


# AMReX quoted strings contain literal backslashes (not shell/Python escapes).
_PART = re.compile(r'"""[\s\S]*?"""|"[^"\n]*"|#[^\n]*|\\\n|[^"#\\\n]+|\\|\n')
_QUOTED = re.compile(r'"""([\s\S]*?)"""|"([^"\n]*)"')


def _decoded_value(value: str) -> str:
    def unquote(match: re.Match[str]) -> str:
        triple, double = match.groups()
        if triple is not None:
            return triple.removeprefix("\n")
        return double

    return _QUOTED.sub(unquote, value)


def read_key_values(
    path: Path | None, *, serialized: bool = False
) -> dict[str, str]:
    """Read assignments; last definition wins, as in canonical ``run.in``.

    Missing optional report inputs yield an empty mapping. Execution callers
    must separately require their input file. ``serialized=True`` retains
    quoting, so generated inputs do not turn a string into comments or tokens.
    """
    if path is None or not path.exists():
        return {}
    text = path.read_text()
    values: dict[str, str] = {}
    parts: list[str] = []
    position = 0

    def finish_assignment() -> None:
        line = "".join(parts).strip()
        parts.clear()
        if not line:
            return
        if "=" not in line:
            raise ValueError(f"{path}: expected an AMReX assignment: {line}")
        key, value = (part.strip() for part in line.split("=", 1))
        if key in ("FILE", "UNSET"):
            raise ValueError(f"{path}: {key} requires AMReX; use a self-contained assignment file")
        values[key] = value if serialized else _decoded_value(value)

    for match in _PART.finditer(text):
        if match.start() != position:
            raise ValueError(f"{path}: unterminated quoted input value")
        position = match.end()
        token = match.group()
        if re.match(r"#\s*(if|ifdef|ifndef|elif|else|endif)\b", token):
            raise ValueError(f"{path}: conditional inputs require AMReX; use resolved assignments")
        if token == "\n":
            finish_assignment()
        elif token == "\\\n":
            parts.append(" ")
        elif not token.startswith("#"):
            parts.append(token)
    if position != len(text):
        raise ValueError(f"{path}: unterminated quoted input value")
    finish_assignment()
    return values

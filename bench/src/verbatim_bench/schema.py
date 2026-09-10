# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Dependency-free validation of a results document against `vb-results/1`.

No `jsonschema` dependency, by design: `verify` must stay installable anywhere so
that anyone can check a number this project publishes without owning the hardware
that produced it. The schema file uses only a small subset of JSON Schema
(`type`, `required`, `properties`, `additionalProperties`, `items`, `enum`,
`pattern`, `minimum`, `maximum`, `minItems`, `maxItems`, `const`, and `null`
unions via `["number", "null"]`), and this validator implements exactly that
subset. If the schema ever needs a keyword outside it, extend this module first.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

SCHEMA_ID: Final = "vb-results/1"
SCHEMA_ID_V2: Final = "vb-results/2"
SCHEMA_RELATIVE_PATH: Final = Path("benchmarks/schema/row.schema.json")
SCHEMA_RELATIVE_PATH_V2: Final = Path("benchmarks/schema/row.schema.v2.json")
SCHEMA_FILES: Final = {
    SCHEMA_ID: SCHEMA_RELATIVE_PATH,
    SCHEMA_ID_V2: SCHEMA_RELATIVE_PATH_V2,
}


@dataclass(frozen=True, slots=True)
class SchemaError:
    path: str  # JSON-pointer-ish, e.g. "result.latency_ms.partial.p95"
    message: str


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _index(path: str, i: int) -> str:
    return f"{path}[{i}]" if path else f"[{i}]"


def _type_matches(value: Any, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, list)
    if name == "object":
        return isinstance(value, dict)
    return False


def _check(value: Any, schema: Mapping[str, Any], path: str, errors: list[SchemaError]) -> None:
    if not isinstance(schema, Mapping):
        return
    expected = schema.get("type")
    if expected is not None:
        names = [expected] if isinstance(expected, str) else list(expected)
        if not any(_type_matches(value, name) for name in names):
            errors.append(
                SchemaError(
                    path or "(root)",
                    f"expected type {'/'.join(names)}, got {_typename(value)}",
                )
            )
            return
    if "const" in schema and value != schema["const"]:
        errors.append(SchemaError(path or "(root)", f"expected const {schema['const']!r}"))
    if "enum" in schema and value not in schema["enum"]:
        errors.append(SchemaError(path or "(root)", f"expected one of {list(schema['enum'])!r}"))
    if (
        isinstance(value, str)
        and "pattern" in schema
        and re.search(str(schema["pattern"]), value) is None
    ):
        errors.append(
            SchemaError(path or "(root)", f"value does not match pattern {schema['pattern']!r}")
        )
    if isinstance(value, bool):
        pass
    elif isinstance(value, int | float):
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            errors.append(
                SchemaError(path or "(root)", f"value {value!r} is below minimum {minimum!r}")
            )
        maximum = schema.get("maximum")
        if maximum is not None and value > maximum:
            errors.append(
                SchemaError(path or "(root)", f"value {value!r} is above maximum {maximum!r}")
            )
    if isinstance(value, list):
        min_items = schema.get("minItems")
        if min_items is not None and len(value) < min_items:
            errors.append(
                SchemaError(
                    path or "(root)", f"expected at least {min_items} items, got {len(value)}"
                )
            )
        max_items = schema.get("maxItems")
        if max_items is not None and len(value) > max_items:
            errors.append(
                SchemaError(
                    path or "(root)", f"expected at most {max_items} items, got {len(value)}"
                )
            )
        items = schema.get("items")
        if isinstance(items, Mapping):
            for i, entry in enumerate(value):
                _check(entry, items, _index(path, i), errors)
    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(SchemaError(_join(path, str(key)), f"missing required field {key!r}"))
        properties = schema.get("properties", {})
        if isinstance(properties, Mapping):
            for key, subschema in properties.items():
                if key in value and isinstance(subschema, Mapping):
                    _check(value[key], subschema, _join(path, str(key)), errors)
        additional = schema.get("additionalProperties", True)
        if additional is False and isinstance(properties, Mapping):
            for key in value:
                if key not in properties:
                    errors.append(SchemaError(_join(path, str(key)), f"unknown field {key!r}"))
        elif isinstance(additional, Mapping):
            for key in value:
                if key not in properties:
                    _check(value[key], additional, _join(path, str(key)), errors)


def _typename(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def load_schema(schema_id: str = SCHEMA_ID) -> dict[str, Any]:
    """Read the row schema for ``schema_id``, resolved relative to the repo root
    and also findable from an installed wheel."""
    relative = SCHEMA_FILES.get(schema_id, SCHEMA_RELATIVE_PATH)
    fallback_name = Path(relative).name
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        candidate = parent / relative
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
        candidate = parent / "schema" / fallback_name
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    try:
        from importlib.resources import files

        resource = files("verbatim_bench").joinpath(f"../../benchmarks/schema/{fallback_name}")
        if resource.is_file():
            return json.loads(resource.read_text(encoding="utf-8"))
    except (ImportError, ValueError, OSError):
        pass
    raise FileNotFoundError(f"could not locate {relative} from {here}")


def validate_for_schema(doc: Mapping[str, Any], schema_id: str) -> list[SchemaError]:
    """Validate ``doc`` against the schema named by ``schema_id``."""
    return validate(doc, load_schema(schema_id))


def validate(doc: Mapping[str, Any], schema: Mapping[str, Any] | None = None) -> list[SchemaError]:
    """Return every violation, not just the first. Empty list means valid."""
    resolved = schema if schema is not None else load_schema()
    errors: list[SchemaError] = []
    _check(dict(doc), resolved, "", errors)
    return errors

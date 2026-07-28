from __future__ import annotations

from typing import Any


def patch_gradio_schema() -> None:
    import gradio_client.utils as client_utils

    if getattr(client_utils, "_gpt_sovits_bool_schema_compat", False):
        return

    original = client_utils._json_schema_to_python_type

    def parse_schema(schema: Any, definitions: Any) -> str:
        if isinstance(schema, bool):
            return "Any"
        return original(schema, definitions)

    client_utils._json_schema_to_python_type = parse_schema
    client_utils._gpt_sovits_bool_schema_compat = True

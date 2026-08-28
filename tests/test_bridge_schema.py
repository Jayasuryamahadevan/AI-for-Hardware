"""Tool-schema emitters: one neutral ToolSpec, six vendor projections."""

from __future__ import annotations

from labbench.bridge.schema import ToolSpec, emit, sanitise_name


def make_spec(**overrides) -> ToolSpec:
    defaults = {
        "name": "device.invoke",
        "description": "Run a command.",
        "parameters": {
            "type": "object",
            "properties": {
                "device": {"type": "string"},
                "z_um": {"type": "number", "minimum": 0, "maximum": 200},
            },
            "required": ["device"],
        },
        "hazard": "motion", "destructive": True,
    }
    defaults.update(overrides)
    return ToolSpec(**defaults)


class TestSanitiseName:
    def test_dots_and_slashes_become_underscores(self):
        assert sanitise_name("device.invoke") == "device_invoke"
        assert sanitise_name("a/b.c") == "a_b_c"

    def test_truncated_to_64_chars(self):
        assert len(sanitise_name("x" * 100)) == 64


class TestAnthropic:
    def test_shape(self):
        [tool] = emit([make_spec()], "anthropic")
        assert tool["name"] == "device_invoke"
        assert tool["input_schema"]["properties"]["device"]["type"] == "string"
        assert "hazard: motion" in tool["description"]
        assert "IRREVERSIBLE" in tool["description"]


class TestOpenAI:
    def test_non_strict_preserves_optional_arguments(self):
        [tool] = emit([make_spec()], "openai")
        assert tool["function"]["parameters"]["required"] == ["device"]

    def test_strict_mode_makes_every_property_required_and_nullable(self):
        [tool] = emit([make_spec()], "openai", strict=True)
        params = tool["function"]["parameters"]
        assert set(params["required"]) == {"device", "z_um"}
        assert params["additionalProperties"] is False
        # z_um was optional; strict mode must widen it to admit null rather
        # than force the model to invent a value.
        z_type = params["properties"]["z_um"]["type"]
        assert "null" in z_type

    def test_openai_responses_flattens_by_one_level(self):
        [tool] = emit([make_spec()], "openai-responses")
        assert tool["type"] == "function"
        assert tool["parameters"]["properties"]["device"]["type"] == "string"


class TestGemini:
    def test_drops_unsupported_keywords_into_prose(self):
        [wrapped] = emit([make_spec()], "gemini")
        [decl] = wrapped["function_declarations"]
        z_schema = decl["parameters"]["properties"]["z_um"]
        assert "minimum" not in z_schema
        assert "0 to 200" in z_schema["description"]


class TestJsonSchema:
    def test_keeps_labbench_metadata_as_structured_fields(self):
        [tool] = emit([make_spec()], "jsonschema")
        assert tool["hazard"] == "motion"
        assert tool["annotations"]["destructive"] is True


class TestOpenApi:
    def test_produces_one_path_per_tool_with_hazard_extensions(self):
        doc = emit([make_spec()], "openapi")
        assert doc["openapi"].startswith("3.1")
        [(path, item)] = doc["paths"].items()
        assert path == "/tools/device_invoke"
        assert item["post"]["x-hazard"] == "motion"
        assert item["post"]["x-irreversible"] is True


class TestFullDescription:
    def test_read_only_note_is_appended(self):
        spec = make_spec(read_only=True, destructive=False, hazard="none")
        assert "read-only" in spec.full_description()

    def test_plain_tool_gets_no_bracketed_notes(self):
        spec = make_spec(hazard=None, destructive=False, read_only=False)
        assert "[" not in spec.full_description()

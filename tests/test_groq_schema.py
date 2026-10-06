from tracemeet.llm.groq_provider import strict_schema


def test_strict_schema_preserves_nullable_fields_and_original():
    original = {
        "type": "object",
        "properties": {
            "owner": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
            }
        },
        "$defs": {
            "Evidence": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string", "minLength": 1}
                },
            }
        },
    }

    result = strict_schema(original)

    assert result["additionalProperties"] is False
    assert result["required"] == ["owner"]
    assert result["properties"]["owner"]["anyOf"] == [
        {"type": "string"}, {"type": "null"}
    ]
    assert "default" not in result["properties"]["owner"]
    assert result["$defs"]["Evidence"]["required"] == ["quote"]
    assert result["$defs"]["Evidence"]["additionalProperties"] is False
    assert result["$defs"]["Evidence"]["properties"]["quote"]["minLength"] == 1

    # The source schema was not mutated.
    assert "additionalProperties" not in original
    assert original["properties"]["owner"]["default"] is None
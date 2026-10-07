import pytest

from tracemeet.stages.minutes_group import (
    DiscussionGroup,
    GroupPlan,
    GroupingError,
    materialize_groups,
    validate_group_plan,
)


def make_plan(*groups):
    return GroupPlan(
        groups=[
            DiscussionGroup(
                label=f"Discussion {index}",
                observation_numbers=numbers,
            )
            for index, numbers in enumerate(groups, start=1)
        ]
    )


def test_related_observations_can_span_distant_positions():
    plan = make_plan([1, 4], [2, 3])
    validate_group_plan(plan, 4)


@pytest.mark.parametrize(
    "groups",
    [
        ([1], [2]),          # Missing 3.
        ([1, 2], [2, 3]),   # Duplicate 2.
        ([1, 2, 3, 4],),    # Unexpected 4.
        ([0, 1, 2, 3],),    # Invalid zero.
    ],
)
def test_invalid_partition_is_rejected(groups):
    with pytest.raises(GroupingError):
        validate_group_plan(make_plan(*groups), 3)


def test_original_ids_are_restored_in_source_order():
    observations = [
        {"observation_id": "chunk_001_note_001"},
        {"observation_id": "chunk_002_note_001"},
        {"observation_id": "chunk_003_note_001"},
    ]
    result = materialize_groups(
        make_plan([2], [3, 1]),
        observations,
    )

    assert result[0]["observation_ids"] == [
        "chunk_001_note_001",
        "chunk_003_note_001",
    ]
    assert result[1]["observation_ids"] == [
        "chunk_002_note_001",
    ]


from pydantic import ValidationError

from tracemeet.stages.minutes_group import (
    assignment_schema,
    assignments_to_plan,
)


def test_assignment_schema_requires_every_observation():
    schema = assignment_schema(3)

    with pytest.raises(ValidationError):
        schema.model_validate(
            {
                "assignments": {
                    "obs_0001": "Deployment",
                    "obs_0003": "Deployment",
                }
            }
        )


def test_assignment_schema_rejects_unknown_observation():
    schema = assignment_schema(1)

    with pytest.raises(ValidationError):
        schema.model_validate(
            {
                "assignments": {
                    "obs_0001": "Deployment",
                    "obs_0099": "Other",
                }
            }
        )


def test_python_groups_distant_observations_without_losing_any():
    schema = assignment_schema(3)
    response = schema.model_validate(
        {
            "assignments": {
                "obs_0001": "Deployment",
                "obs_0002": "Hiring",
                "obs_0003": "Deployment",
            }
        }
    )

    plan = assignments_to_plan(response, 3)

    assert plan.groups[0].observation_numbers == [1, 3]
    assert plan.groups[1].observation_numbers == [2]
    validate_group_plan(plan, 3)


def test_assignment_label_cannot_be_blank():
    schema = assignment_schema(1)

    with pytest.raises(ValidationError):
        schema.model_validate(
            {"assignments": {"obs_0001": "   "}}
        )


def test_compact_schema_keeps_required_fields_and_constraints():
    schema = assignment_schema(3).model_json_schema()
    assignments = schema["$defs"]["ObservationAssignments"]

    assert assignments["required"] == [
        "obs_0001", "obs_0002", "obs_0003"
    ]
    assert assignments["additionalProperties"] is False

    field = assignments["properties"]["obs_0001"]
    assert "title" not in field
    assert field["type"] == "string"
    assert field["minLength"] == 1
    assert field["maxLength"] == 80
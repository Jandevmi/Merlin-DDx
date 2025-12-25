import logging
import re

from pydantic import ValidationError, BaseModel

from src.pipeline.pydantic_schemas import DiagnosesModel, ICDsModel
from src.pipeline.verifier_args import VerifierArgs


def extract_values_from_json(data):
    values = []

    def recurse(node):
        if isinstance(node, dict):
            if "value" in node and isinstance(node["value"], (int, float)):
                values.append(node["value"])
            else:
                for val in node.values():
                    recurse(val)
        elif isinstance(node, list):
            for item in node:
                recurse(item)

    recurse(data)
    return values


def convert_symptom_dict_to_symptom_vector(symptoms: list) -> list:
    return [[extract_values_from_json(choice) for choice in choices] for choices in symptoms]


def truncate_reasons(diagnoses, max_length=1000):
    """Truncate the 'reason' field in each diagnosis if it exceeds max_length."""
    if not isinstance(diagnoses, list):
        return diagnoses
    for item in diagnoses:
        reason = item.get('reason')
        if isinstance(reason, str) and len(reason) > max_length:
            # Cut at the last space before max_length
            if len(reason) > max_length:
                # logging.warning(f'Truncate reason: {reason}')
                truncated = reason[:max_length].rsplit(' ', 1)[0]
                item['reason'] = truncated + '...'
    return diagnoses


def get_regex_pattern(schema) -> str:
    """Get regex pattern for extracting JSON-like strings based on the schema type."""
    if schema.__name__ == "ManifestationsModel":
        return r'(\{\s*".*?"\s*:\s*{.*}\s*\})'  # dict of dicts
    elif schema == DiagnosesModel or schema == ICDsModel:
        return r'(\[\s*{.*?}\s*\])'  # list of dicts
    else:
        raise NotImplementedError(f"Unknown scheme type: {schema}")


def convert_string_to_verified_json(json_strings: list, schema: BaseModel) -> dict | None:
    """Convert JSON-like strings to validated JSON objects using the provided schema."""
    for match in reversed(json_strings):
        try:
            json_obj = schema.model_validate_json(match).model_dump()
            return truncate_reasons(json_obj)
        except ValidationError as e:
            # print(e)
            continue


def extract_jsons(schema: BaseModel, gen_output: str) -> dict | None:
    """Extract and validate JSON-like data from LLM outputs using dynamic schemas."""
    if not gen_output:
        return None

    pattern = get_regex_pattern(schema)
    matches = re.findall(pattern, gen_output, re.DOTALL)
    return convert_string_to_verified_json(matches, schema)


def extract_jsons_from_responses(v_args: VerifierArgs, responses: list[list[str]]) -> list:
    """Extract validated JSON objects from LLM responses."""
    extracted_jsons = [[extract_jsons(v_args.pydantic_scheme, choice)
                        for choice in choices] for choices in responses]
    not_none_count = sum(choice is not None for choices in extracted_jsons for choice in choices)
    total_count = sum(len(choices) for choices in extracted_jsons)
    logging.info(f'Extracted {not_none_count} valid JSONs from {total_count} total choices.')
    return extracted_jsons

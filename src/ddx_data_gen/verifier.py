import logging

import pandas as pd
import torch

from torch import tensor

from src.ddx_data_gen.json_extraction import get_symptom_vectors, DiagnosesModel, ICDsModel
from src.ddx_data_gen.prompt_args import PromptArgs
from src.utils import convert_codes_to_short_codes


from sentence_transformers import SentenceTransformer, util
DEVICE = 'cuda'


def get_extraction_score_torch(v1: list, v2: list) -> float:
    if len(v1) == 0 or len(v2) == 0:
        return 0.0

    try:
        v1_tensor = tensor(v1, dtype=torch.int, device=DEVICE)
        v2_tensor = tensor(v2, dtype=torch.int, device=DEVICE)
        acc = (v1_tensor == v2_tensor).float().mean()
        return round(acc.item(), 2)
    except Exception as e:
        logging.error(f"Error in extraction score: {e}")
        return 0.0


def normalized_dot_product_torch(prediction, label):
    prediction = tensor(prediction, dtype=torch.float32, device=DEVICE)
    label = tensor(label, dtype=torch.float32, device=DEVICE)

    if prediction.shape != label.shape:
        return 0.0

    dot = torch.dot(prediction, label)
    k = (label != 0).sum()

    if k == 0:
        return 0.5

    return round(((dot + k) / (2 * k)).item(), 3)


def get_rank_score(y_pred: list[str], y_true: str):
    """1 if main diagnose on first place -0.1 for every rank behind"""
    y_pred = [pred.lower() for pred in y_pred]
    if not isinstance(y_pred, list):
        return -1
    try:
        return round(1 - (y_pred[:10].index(y_true.lower()) / 10), 1)
    except ValueError:
        return 0


def extract_and_map_diagnose_name(model, diagnose_json: list[dict], labels: list[str], label_embeddings) -> list[str]:
    if diagnose_json is None:
        return []

    pred_names = []
    for pred_dict in diagnose_json:
        pred = pred_dict['name']
        if pred in labels:
            pred_names.append(pred)
            continue

        pred_embedding = model.encode(pred, convert_to_tensor=True,
                                      normalize_embeddings=True,
                                      show_progress_bar=False)
        cosine_scores = util.cos_sim(pred_embedding, label_embeddings)
        best_idx = torch.argmax(cosine_scores).item()
        matched_name = labels[best_idx]
        pred_names.append(matched_name)

    return pred_names


def extract_diagnose_names_from_choices(
        p_args: PromptArgs,
        patients: pd.DataFrame,
        choices: list[list[list[dict]] | None],
        model: SentenceTransformer
) -> list[list[list[str]] | None]:
    """Extracts name from dict name/reason format.
    Maps predicted name to a potential diagnosis based on the primary complaint."""
    work_df = patients.copy()
    work_df['choices'] = choices
    matched_names = []

    # Iterate over all primary complaints
    for primary_complaint, potential_diagnoses in p_args.potential_diagnoses.items():
        sub_df = work_df[work_df['Chief Complaint'] == primary_complaint]
        label_embeddings = model.encode(
            potential_diagnoses,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        for choices in sub_df['choices'].tolist():
            if choices is not None:
                matched_names.append(
                    [extract_and_map_diagnose_name(model, choice, potential_diagnoses, label_embeddings)
                     for choice in choices]
                )
            else:
                matched_names.append([])

    return matched_names


def extract_icd_names(icd_dicts: list, potential_codes: set) -> list:
    if not icd_dicts:
        return []

    result = []
    for choices in icd_dicts:
        if not choices:
            result.append([])
            continue

        inner_result = []
        for choice in choices:
            if not choice:
                inner_result.append([])
                continue

            valid_codes = [
                code for code in convert_codes_to_short_codes(icd['icd_code'] for icd in choice)
                if code in potential_codes
            ]
            inner_result.append(valid_codes)

        result.append(inner_result)

    return result


def verifier_1_scores(preds, labels) -> list[list[dict]]:
    return [[
        {
            'extraction_acc': get_extraction_score_torch(preds[i][j], labels[i]),
            'verifier_score': normalized_dot_product_torch(preds[i][j], labels[i])
        } for j in range(len(preds[i]))] for i in range(len(preds))]


def verifier_2_3_scores(preds: list[list[list[str]]], labels: list[str]) -> list[list[dict]]:
    all_scores = []

    for pred_choices, label_text in zip(preds, labels):
        score_row = [
            {"verifier_score": get_rank_score(pred_set, label_text)}
            for pred_set in pred_choices
        ]
        all_scores.append(score_row)

    return all_scores


def get_icd_f1_score(pred: list, true: list) -> float:
    intersection = set(pred) & set(true)
    precision = len(intersection) / len(pred) if pred else 0.0
    recall = len(intersection) / len(true) if true else 0.0
    return round(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0, 3)


def verifier_4_scores(preds, labels) -> list[list[dict]]:
    return [[
        {
            'verifier_score': get_icd_f1_score(preds[i][j], labels[i]),
        } for j in range(len(preds[i]))] for i in range(len(preds))]


def calculate_scores(p_args: PromptArgs, work_df: pd.DataFrame, extracted_choices: list[list[list[dict]]],
                     mapping_model) -> (list[list[dict]], list):
    """Extracts labels and calculates scores for the given verifier step."""
    scheme_name = p_args.pydantic_scheme.__name__
    logging.info(f'Calculate scores for {scheme_name}')
    if scheme_name == 'ManifestationsModel':
        predictions = get_symptom_vectors(extracted_choices)
        return verifier_1_scores(predictions, work_df['disease_vector'].tolist()), predictions

    elif p_args.pydantic_scheme == DiagnosesModel:
        predictions = extract_diagnose_names_from_choices(p_args, work_df, extracted_choices, mapping_model)
        return verifier_2_3_scores(predictions, work_df['disease'].tolist()), predictions

    elif p_args.pydantic_scheme == ICDsModel:
        potential_codes = set(p_args.potential_icds)
        predictions = extract_icd_names(extracted_choices, potential_codes)
        labels = [convert_codes_to_short_codes(codes) for codes in work_df['ICD_CODES'].tolist()]
        return verifier_4_scores(predictions, labels), predictions

    else:
        raise NotImplementedError(f'Unknown scheme: {p_args.pydantic_scheme}')

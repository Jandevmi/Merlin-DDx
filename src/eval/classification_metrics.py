import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score

from src.eval.pipeline import extract_icd_names
from src.utils import detect_device, convert_codes_to_short_codes


def calculate_verifier_metrics(dataset_df: pd.DataFrame) -> dict:
    return {
        'v1_score': dataset_df['v1_score'].mean(),
        'v2_score': dataset_df['v2_score'].mean(),
        'v3_score': dataset_df['v3_score'].mean(),
        'v4_score': dataset_df['v4_score'].mean()
    }


def cosine_similarity_torch(prediction, label):
    prediction = torch.tensor(prediction, dtype=torch.float32, device='mps')
    label = torch.tensor(label, dtype=torch.float32, device='mps')

    if prediction.shape != label.shape:
        return 0.0

    pred_norm = torch.norm(prediction)
    label_norm = torch.norm(label)

    if pred_norm == 0 or label_norm == 0:
        return 0.0  # or define a fallback, e.g. 0.5

    cos_sim = torch.dot(prediction, label) / (pred_norm * label_norm)
    return round(cos_sim.item(), 3)


def get_mrr(y_true: str, y_pred: list) -> float:
    if y_true not in y_pred:
        return 0
    else:
        return 1 / (y_pred.index(y_true) + 1)


def get_recall_1(y_true: str, y_pred: list) -> float:
    if not y_pred:
        return 0
    return 1 if y_true == y_pred[0] else 0


def calculate_disease_metrics(y_pred, y_true) -> float:

    # Compute metrics using list comprehensions
    mrr_scores = [get_mrr(t, p) for t, p in zip(y_true, y_pred)]
    return sum(mrr_scores) / len(mrr_scores)
    # recall1_scores = [get_recall_1(t, p) for t, p in zip(y_true, y_pred)]
    #
    # # Return average metrics
    # return {
    #     'Diagnose MRR': sum(mrr_scores) / len(mrr_scores),
    #     # 'Diagnose Recall@1': sum(recall1_scores) / len(recall1_scores)
    # }


def summarize_cv_results(results: pd.DataFrame, experiment_name: str, add_std: bool):
    df = pd.DataFrame(results)
    mean = df.mean()
    std = df.std()

    # Interleave columns as metric_mean / metric_std
    summary = {}
    for i, metric in enumerate(df.columns):
        summary[f"{metric}"] = mean[metric]
        if add_std:
            summary[f"{i}_std"] = std[metric]
        # summary[f"{metric}_std"] = f"{std[metric]:.1e}"

    return pd.DataFrame(summary, index=[experiment_name])


def calculate_in_domain_score(results: pd.DataFrame) -> float:
    short_codes = results['ICD_CODES'].map(convert_codes_to_short_codes).tolist()
    unique_labels = {label for labels in short_codes for label in labels}

    def calculate_score(icds: list, uniques: set) -> float:
        return len(set(icds).intersection(uniques)) / len(icds)

    return ((results['v4_json']
             .map(extract_icd_names)
            .pipe(lambda x: x[x.map(bool)]))  # Filter None
            .apply(lambda x: calculate_score(x, unique_labels)).mean())


def calculate_icd_metrics(y_pred: list[list[str]], y_true: list[list[str]]) -> dict:
    device = detect_device()
    mlb = MultiLabelBinarizer()
    mlb.fit(y_pred + y_true)
    y_pred = mlb.transform(y_pred)
    y_true = mlb.transform(y_true)

    y_pred_bin = torch.tensor(y_pred, dtype=torch.float32, device=device)
    y_true_bin = torch.tensor(y_true, dtype=torch.float32, device=device)
    num_labels = len(mlb.classes_)

    precision_micro = MultilabelPrecision(num_labels=num_labels, average='micro').to(device)
    recall_micro = MultilabelRecall(num_labels=num_labels, average='micro').to(device)
    f1_micro = MultilabelF1Score(num_labels=num_labels, average='micro').to(device)

    precision_macro = MultilabelPrecision(num_labels=num_labels, average='macro').to(device)
    recall_macro = MultilabelRecall(num_labels=num_labels, average='macro').to(device)
    f1_macro = MultilabelF1Score(num_labels=num_labels, average='macro').to(device)

    return {
        # 'recall_micro': round(recall_micro(y_pred_bin, y_true_bin).item(), 4),
        # 'recall_macro': round(recall_macro(y_pred_bin, y_true_bin).item(), 4),
        # 'precision_micro': round(precision_micro(y_pred_bin, y_true_bin).item(), 4),
        # 'precision_macro': round(precision_macro(y_pred_bin, y_true_bin).item(), 4),
        'ICD F1 Micro': round(f1_micro(y_pred_bin, y_true_bin).item(), 4),
        'ICD F1 Macro': round(f1_macro(y_pred_bin, y_true_bin).item(), 4)
    }


def get_valid_json_pct(df: pd.DataFrame) -> float:
    invalids = 0
    for i in [2, 4]:
        invalids += sum(df[f'v{i}_json'].apply(lambda x: x == 'None' or not x))
    return round(1 - (invalids / (3 * len(df))), 4)

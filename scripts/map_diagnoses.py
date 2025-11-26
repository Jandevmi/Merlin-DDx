
import pandas as pd

from src.exp_args import ExpArgs
from src.preprocessing.wikidoc_mapping import add_mapped_primary_diagnoses


if __name__ == "__main__":
    args = ExpArgs()
    add_mapped_primary_diagnoses(args, 'patients_10.pq')
    df_10 = pd.read_parquet('data/reasoning/abdominal_pain/patients_10.pq')
    df_10.rename(columns={'disease': 'disease_old'}, inplace=True)  # Use something more meaningful
    df_10['disease'] = df_10['wikidoc_diagnoses'].apply(lambda x: x[0])
    df_10.to_csv('data/reasoning/abdominal_pain/patients_10.pq', index=False)

    add_mapped_primary_diagnoses(args, 'patients_63.pq')
    df_63 = pd.read_parquet('data/reasoning/abdominal_pain/patients_63.pq')
    df_63.rename(columns={'disease': 'disease_old'}, inplace=True)  # Use something more meaningful
    df_63['disease'] = df_63['wikidoc_diagnoses'].apply(lambda x: x[0])
    df_63.to_csv('data/reasoning/abdominal_pain/patients_63.pq', index=False)


import os

import wandb

from src.wandb.run import init_wandb

wandb_run = init_wandb('Upload Patients Dataset', eval_mode=False)

# artifact = wandb.Artifact('patients_10', type='dataset')
# artifact.add_file('data/reasoning/abdominal_pain/patients_10.pq')
# wandb_run.log_artifact(artifact)
#
# artifact = wandb.Artifact('patients_63', type='dataset')
# artifact.add_file('data/reasoning/abdominal_pain/patients_63.pq')
# wandb_run.log_artifact(artifact)
#
# artifact = wandb.Artifact('patients_1411', type='dataset')
# artifact.add_file('data/reasoning/abdominal_pain/patients_1411.pq')
# wandb_run.log_artifact(artifact)

artifact = wandb.Artifact('patients_3055', type='dataset')
artifact.add_file('data/reasoning/abdominal_pain/patients_3055.pq')
wandb_run.log_artifact(artifact)

artifact = wandb.Artifact('test_dataset', type='dataset')
artifact.add_file('data/reasoning/abdominal_pain/test_dataset.pq')
wandb_run.log_artifact(artifact)

artifact = wandb.Artifact('patients_ood_700', type='dataset')
artifact.add_file('data/reasoning/patients_ood_700.pq')
wandb_run.log_artifact(artifact)

artifact = wandb.Artifact('abdominal_pain_diagnoses', type='schemes')
artifact.add_file('data/medical_schemes/diagnoses/abdominal_pain.csv')
wandb_run.log_artifact(artifact)

artifact = wandb.Artifact('abdominal_pain_symptoms', type='schemes')
artifact.add_file('data/medical_schemes/symptoms/abdominal_pain.yaml')
wandb_run.log_artifact(artifact)

wandb_run.finish()

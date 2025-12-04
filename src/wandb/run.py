import json
import logging
import os

import wandb
from wandb.sdk.wandb_run import Run

from src.exp_args import ExpArgs
from src.pipeline.verifier_args import VerifierArgs


def init_wandb(run_name: str, eval_mode=False, tags=None) -> Run:
    project_name = 'merlin-eval' if eval_mode else 'trump'

    key = os.getenv("WANDB_API_KEY") or json.load(open("config.json"))["WANDB_API_KEY"]

    wandb.login(key=key)
    return wandb.init(
        project=project_name,
        entity='datexis-phd',
        name=run_name,
        tags=tags,
    )


def update_wandb_name_tags(run, exp_args: ExpArgs, v_args: VerifierArgs):
    """Update the name of an existing WandB run."""
    run.name = exp_args.short_llm_name
    run.tags += (exp_args.short_llm_name, exp_args.hardware)

    if exp_args.ood_eval:
        run.name += '-OOD'
        run.tags += ('OOD', )
    if exp_args.config_str['Model'].get('lora'):
        run.name += f'-lora-{exp_args.config_str["name"][-6:]}'
        run.tags += ('lora', )
    if exp_args.config_str['Model'].get('thinking'):
        run.name += '-think'
    if not exp_args.config_str['Client_Job'].get('merlin_mode'):
        run.name += '-no-merlin'
    if not exp_args.config_str['Client_Job'].get('think_about_labs'):
        run.name += '-no-labs'
    elif exp_args.guided_decoding:
        run.name += '-guid_decoding'
        run.tags += ('gui_dec', )

    exp_args.run_name = run.name

    if exp_args.eval_mode and exp_args.merlin_mode:
        run.tags += ('merlin_eval', )
    elif exp_args.eval_mode and not exp_args.merlin_mode:
        run.tags += ('mimic_eval', )
    else:
        run.tags += (f'data_gen', str(exp_args.num_samples))

    logging.info(f'Updated WandB run name to: {run.name}')

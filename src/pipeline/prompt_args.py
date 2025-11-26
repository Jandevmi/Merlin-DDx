import logging
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel

from src.pipeline.prompts import PROMPT_TEMPLATES_GEN, PROMPT_TEMPLATES_EVAL_MERLIN, \
    PROMPT_TEMPLATES_EVAL_MIMIC, PROMPT_TEMPLATES_EVAL_NO_LAB
from src.exp_args import ExpArgs


@dataclass
class PromptArgs:
    chief_complaint: str = 'abdominal_pain'
    api_config: dict = None
    prompt: str = None
    manifestations_yaml: any = None
    guided_decoding: bool = True
    manifestation_extraction_schema: any = None
    pydantic_scheme: type[BaseModel] = None
    potential_diagnoses: dict = None
    potential_icds: list[str] = None
    verifier_thresholds: dict = None
    num_choices: int = 10
    num_samples: int = 10
    batch_size: int = 4
    concurrency: int = 32
    temperature: float = 0.2
    max_tokens: int = 2500
    budget: int = 5
    current_budget: int = -1
    current_verifier_step: int = 1
    init_time: datetime = datetime.now()

    def get_run_name(self, llm_name=None) -> str:
        if llm_name:
            return f'{self.num_samples}_{llm_name}_{str(self.init_time)[:-7]}'
        else:
            return f'{self.num_samples}_{str(self.init_time)[:-7]}'

    def __str__(self):
        return (f"Batchsize: {self.batch_size}, Max. concurrency: {self.concurrency}, "
                f"Temperature: {self.temperature}, Max. tokens: {self.max_tokens}, "
                f"Num Choices: {self.num_choices}")

    def set_verifier_args(
            self,
            exp_args: ExpArgs,
            v_args: dict,
            v_step: int,
    ):
        """ Set prompt arguments for the current verifier step. """
        idx = v_step - 1
        if exp_args.eval_mode and exp_args.merlin_mode:
            templates = PROMPT_TEMPLATES_EVAL_MERLIN
        elif exp_args.eval_mode and exp_args.think_about_labs:
            templates = PROMPT_TEMPLATES_EVAL_NO_LAB
        elif exp_args.eval_mode and not exp_args.merlin_mode:
            templates = PROMPT_TEMPLATES_EVAL_MIMIC
        else:
            templates = PROMPT_TEMPLATES_GEN

        templates = templates
        self.prompt = templates[v_args['templates'][idx]]
        self.budget = v_args['budget'][v_step-1]
        self.temperature = v_args['temperatures'][v_step-1]
        self.max_tokens = v_args['max_tokens'][v_step-1]
        self.pydantic_scheme = v_args['schema'][v_step - 1] or self.pydantic_scheme

        # Check if current budget is given from checkpoint, set only once
        current_budget = v_args.get('current_budget', None)
        if self.current_budget == -1 and current_budget:
            self.current_budget = v_args.get('current_budget', self.budget)
        else:
            self.current_budget = self.budget

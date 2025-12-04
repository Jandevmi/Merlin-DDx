import asyncio
import logging
import os
import time
import traceback
from urllib.error import HTTPError

import httpx
import requests
import wandb
from aiohttp import ClientSession, ClientTimeout
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from src.pipeline.prompt_builder import extract_admission_note_from_prompt
from src.exp_args import ExpArgs
from src.pipeline.verifier_args import VerifierArgs
from src.utils import is_model_chat_based

logger = logging.getLogger()


def get_api_config(service_name='vllm-server', namespace='clinibench', local=False) -> dict:
    if local:
        api_base = "http://localhost:8000/v1"
    else:
        api_base = f"http://{service_name}.{namespace}.svc.cluster.local/v1"
    return {
        "service_name": service_name,
        "openai_api_key": os.getenv("OPENAI_API_KEY", 'openai-abc-key'),
        'openai_api_base': api_base,
        'openai_api_health_url': api_base[:-2] + "health",
    }


def check_connection(api_config: dict) -> bool:
    backoff_time = 1  # Start with 1 second
    num_tries = 0
    max_tries = 100
    while num_tries <= max_tries:
        try:
            response = requests.get(api_config['openai_api_health_url'])
            if response.status_code == 200:
                return True
        except (requests.exceptions.RequestException, HTTPError):
            logging.info(f"Connect {num_tries} to {api_config['openai_api_base']}, "
                         f"retrying in {backoff_time}s")
            time.sleep(backoff_time)
            backoff_time = min(backoff_time * 2, 60)  # Exponential backoff (capped at 60s)
            num_tries += 1
    raise RuntimeError(f"Could not connect to vLLM server after {max_tries} attempts")


def get_endpoint(exp_args: ExpArgs, v_args: VerifierArgs) -> str:
    base = exp_args.api_config["openai_api_base"]
    if is_model_chat_based(exp_args.llm_name):
        if v_args.batch_size > 1:
            raise ValueError("Batch Size must be 1 for Chat Based Models")
        return f"{base}/chat/completions"
    return f"{base}/completions"


async def get_model(api_config: dict) -> str:
    check_connection(api_config)
    client = AsyncOpenAI(
        api_key=api_config['openai_api_key'],
        base_url=api_config['openai_api_base'],
        timeout=httpx.Timeout(1000000),
    )
    models = await client.models.list()
    # Log all available model IDs
    for model in models.data:
        logging.info(f"Available model: {model.id}")
    return models.data[0].id


def build_payload(exp_args: ExpArgs, v_args: VerifierArgs, prompts=None) -> dict:
    payload = {
        "model": "lora_module" if exp_args.lora else exp_args.llm_name,
        "temperature": v_args.temperature,
        "n": v_args.num_choices,
        "max_tokens": v_args.max_tokens if prompts else 256,
        "stream": False,
        "echo": False,
    }

    # Add prompts to payload
    system_prompt = "You are a helpful medical assistant."
    prompts = prompts or ["Warmup test prompt"]
    if is_model_chat_based(exp_args.llm_name):
        # For chat-based models, wrap each prompt separately
        payload["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompts[0]},
        ]
    else:
        # For completion-based models, pass all prompts directly
        payload["prompt"] = [f"{system_prompt}\n{p}" for p in prompts]

    # print(v_args.pydantic_scheme.model_json_schema())

    # Add response format if guided decoding is enabled
    if exp_args.guided_decoding:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": v_args.pydantic_scheme.__name__,
                "schema": v_args.pydantic_scheme.model_json_schema(),
                "strict": True
            }
        }
    return payload


def build_coroutine(session, exp_args: ExpArgs, v_args: VerifierArgs, prompts=None):
    prompts = prompts or ["Warmup test prompt"]
    url = get_endpoint(exp_args, v_args)
    payload = build_payload(exp_args, v_args, prompts)
    headers = {"Content-Type": "application/json"}

    async def request_coro():

        try:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except Exception:
            logger.error(f"Request failed for prompts: {prompts}\n{traceback.format_exc()}")
            return None

    return request_coro, prompts


async def gather_with_concurrency(n, *coros_with_prompts):
    semaphore = asyncio.Semaphore(n)

    async def sem_wrapper(coro, prompt):

        async with semaphore:
            try:
                return await asyncio.wait_for(coro(), timeout=2000)
            except Exception:
                error_note = extract_admission_note_from_prompt(prompt)
                logger.error(f"Task failed for note: {error_note}")
                logger.error(traceback.format_exc())
                return None

    wrapped = [sem_wrapper(coro, prompt) for coro, prompt in coros_with_prompts]
    return await tqdm_asyncio.gather(*wrapped)


async def warm_up(exp_args, v_args: VerifierArgs, session: ClientSession, llm_name) -> None:
    """Warm up and build the outlines grammar on vllm side"""
    warmup_requests = [build_coroutine(session, exp_args, v_args, llm_name) for _ in range(3)]
    warmup_coros = [coro() for coro, _ in warmup_requests]
    await asyncio.gather(*warmup_coros)


async def send_prompts(exp_args, v_args: VerifierArgs, prompts: list, session) -> list:
    batches = [prompts[i:i + v_args.batch_size] for i in range(0, len(prompts), v_args.batch_size)]
    coroutines = [build_coroutine(session, exp_args, v_args, batch) for batch in batches]
    return await gather_with_concurrency(v_args.concurrency, *coroutines)


def extract_text_from_responses(responses: list[dir], num_choices: int, model: str) -> list:
    responses = [response["choices"] for response in responses]

    # Remove batch structure
    responses = [response for batch in responses for response in batch]
    responses = [responses[i:i + num_choices] for i in range(0, len(responses), num_choices)]

    # Extract text based on model type
    if is_model_chat_based(model):
        content = [[choice['message']["content"] for choice in response] for response in responses]
        # thinking = [[choice['message'].get("content") for choice in response] for response in responses]
        return content
    else:
        return [[choice['text'] for choice in response] for response in responses]


async def query_prompts(exp_args: ExpArgs, v_args: VerifierArgs, prompts: list) -> list:
    check_connection(exp_args.api_config)
    if exp_args.llm_name is None:
        exp_args.llm_name = await get_model(exp_args.api_config)
        wandb.log({'model': exp_args.llm_name})

    async with ClientSession(timeout=ClientTimeout(total=1000000)) as session:

        await warm_up(exp_args, v_args, session, exp_args.llm_name)
        responses = await send_prompts(exp_args, v_args, prompts, session)
        return extract_text_from_responses(responses, v_args.num_choices, exp_args.llm_name)

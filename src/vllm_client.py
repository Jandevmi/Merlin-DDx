import asyncio
import logging
import os
import time
import traceback
from urllib.error import HTTPError

import aiohttp
import httpx
import requests
import wandb
from aiohttp import ClientSession, ClientTimeout
from openai import AsyncOpenAI

from src.ddx_data_gen.json_extraction import get_json_schema
from src.exp_args import ExpArgs
from src.ddx_data_gen.prompt_builder import PromptArgs

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
    # logging.info(f'Connecting to {api_config["openai_api_base"]}')
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


def filter_prompt_text(text: str) -> str:
    idx = text.find("### Admission Note")
    if idx != -1:
        return text[idx:]
    else:
        return text  # fallback if not found


def get_request(session, prompt_args: PromptArgs, model: str, prompts=None):
    prompts = prompts or ["Warmup test prompt"]
    # ToDo: Add system prompt to prompt args
    system_prompt = "You are a helpful medical assistant."
    url = f"{prompt_args.api_config['openai_api_base']}/completions"

    # Todo write function
    chat_based = any(name in model.lower() for name in ["chat", "mistral", 'qwen', 'Qwen', 'medreason'])
    if chat_based:
        url = f"{prompt_args.api_config['openai_api_base']}/chat/completions"

        if prompt_args.batch_size > 1:
            raise ValueError("Batch Size must be 1 for Chat Based Models")

    payload = {
        "model": model,
        "temperature": prompt_args.temperature,
        "n": prompt_args.num_choices,
        "max_tokens": prompt_args.max_tokens if prompts else 256,
        "stream": False,
        "echo": False,
    }

    if chat_based:
        # For chat-based models, wrap each prompt separately
        payload["messages"] = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": prompts[0]}]}
        ]
    else:
        # For completion-based models, pass all prompts directly
        payload["prompt"] = [f"{system_prompt}\n{p}" for p in prompts]

    # --- 🔧 Add guided decoding via JSON schema ---
    if prompt_args.guided_decoding:
        # payload['guided_json'] = prompt_args.pydantic_scheme.model_json_schema()
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": prompt_args.pydantic_scheme.__name__,
                "schema": get_json_schema(prompt_args.pydantic_scheme, prompt_args.guided_reasoning)
                # "schema": flatten_pydantic_schema(prompt_args.pydantic_scheme)
            }
        }

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


async def gather_with_concurrency(n, *coros):
    semaphore = asyncio.Semaphore(n)
    total = len(coros)
    completed = 0
    results = []

    async def sem_task(coro_with_prompt):
        nonlocal completed
        coro, prompt = coro_with_prompt
        async with semaphore:
            try:
                # Timeout each request to avoid hangs
                response = await asyncio.wait_for(coro(), timeout=2000)
                return response
            except aiohttp.ClientResponseError as e:
                error_note = filter_prompt_text(prompt)
                logger.error(f"Task failed for prompt: {error_note}\n{traceback.format_exc()}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.error(f"Bad Request: {e.status} {e.message}\n")
                return None
            finally:
                completed += 1
                if completed % 10 == 0 or completed == total:
                    logger.info(f"Progress: {completed}/{total} requests complete")

    tasks = [asyncio.create_task(sem_task(c)) for c in coros]
    for fut in asyncio.as_completed(tasks):
        res = await fut
        results.append(res)

    return results


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


async def warm_up(prompt_args: PromptArgs, session: ClientSession, llm_name) -> None:
    """Warm up and build the outlines grammar on vllm side"""
    warmup_requests = [get_request(session, prompt_args, llm_name) for _ in range(3)]
    warmup_coros = [coro() for coro, _ in warmup_requests]

    await asyncio.gather(*warmup_coros)


async def send_prompts(prompt_args: PromptArgs, prompts: list, session: ClientSession,
                       llm_name: str) -> list:
    batches = [prompts[i:i + prompt_args.batch_size]
               for i in range(0, len(prompts), prompt_args.batch_size)]

    request_tuples = [get_request(session, prompt_args, llm_name, batch) for batch in batches]

    return await gather_with_concurrency(prompt_args.concurrency, *request_tuples)


def extract_text_from_responses(responses: list, num_choices: int, model: str) -> list:
    # Extract text from responses
    # responses = [(await response.json()) for response in responses]
    responses = [response["choices"] for response in responses]

    # Remove batch structure
    responses = [response for batch in responses for response in batch]
    responses = [responses[i:i + num_choices] for i in range(0, len(responses), num_choices)]
    # Return [[choice1, choice2, ...], [choice1, choice2, ...]]
    if any(name in model.lower() for name in ["chat", "mistral", 'qwen', 'Qwen', 'medreason']):
        # [[print(type(choice["message"]["content"])) for choice in response] for response in responses]
        # return [
        #     (msg.get("reasoning_content", "") or "") + (msg.get("content", "") or "")
        #     for response in responses
        #     for choice in response
        #     for msg in [choice["message"]]
        # ]
        content = [[choice['message']["content"] for choice in response] for response in responses]
        # thinking = [[choice['message'].get("content", None) for choice in response] for response in responses]
        return content
    else:
        return [[choice['text'] for choice in response] for response in responses]


async def query_prompts(exp_args: ExpArgs, prompt_args: PromptArgs, prompts: list) -> list:
    """Results have format [text][choices] if num_choice >1 else [text]"""
    check_connection(prompt_args.api_config)

    if exp_args.llm_name is None:
        exp_args.llm_name = await get_model(prompt_args.api_config)
        wandb.log({'model': exp_args.llm_name})

    model = "medreason" if exp_args.lora else exp_args.llm_name

    async with ClientSession(timeout=ClientTimeout(total=1000000)) as session:
        await warm_up(prompt_args, session, exp_args.llm_name)

        max_retries = 3
        for attempt in range(max_retries):

            responses = await send_prompts(prompt_args, prompts, session, model)

            if responses and all(r is not None for r in responses):
                break  # Success
            logging.warning(f"Attempt {attempt+1}: Got bad responses, retrying after delay...")
            await asyncio.sleep(10)
        else:
            logging.error("Max retries reached. Some responses may be None.")

        responses = extract_text_from_responses(responses, prompt_args.num_choices, exp_args.llm_name)

        # Sleep to free CUDA memory
        await asyncio.sleep(30)

        return responses

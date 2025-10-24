import logging
import yaml
import subprocess
import time
from jinja2 import Template

from src.kubernetes.client_template import client_template
from src.kubernetes.server_template import server_template


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def calculate_hardware_requirements(cfg) -> dict:
    parallel_size = cfg['Hardware'].get('parallel_size', 1)
    cfg['memory_limit_server'] = parallel_size * 256
    cfg['memory_request_server'] = parallel_size * 16
    cfg['cpu_limit_server'] = parallel_size * 32
    cfg['cpu_request_server'] = parallel_size * 2
    return cfg


def convert_lists_to_str(cfg):
    # Preprocess lists so Jinja only sees ready-to-use strings
    temps = ", ".join(str(x) for x in cfg["Client_Job"]["temperatures"])
    budget = ", ".join(str(x) for x in cfg["Client_Job"]["budget"])
    max_tokens = ", ".join(str(x) for x in cfg["Client_Job"]["max_tokens"])
    thresholds = ", ".join(str(x) for x in cfg["Client_Job"]["thresholds"])

    cfg = cfg.copy()  # shallow copy to avoid mutating original
    cfg["Client_Job"]["budget_str"] = budget
    cfg["Client_Job"]["temperatures_str"] = temps
    cfg["Client_Job"]["max_tokens_str"] = max_tokens
    cfg["Client_Job"]["thresholds_str"] = thresholds
    return cfg


def render_server(cfg):
    tmpl = Template(server_template)
    return tmpl.render(cfg=cfg)


def render_client(cfg):
    num_gpus = cfg['Hardware'].get('parallel_size', '')
    gpu_type = cfg['Hardware'].get('server_gpu', '')
    cfg['hardware_string'] = str(num_gpus) + "x" + gpu_type
    tmpl = Template(client_template)
    return tmpl.render(cfg=cfg)


def apply_yaml(yaml_str):
    proc = subprocess.run(["kubectl", "apply", "-f", '-'], input=yaml_str.encode(), check=True)
    return proc


def wait_for_job(job_name, namespace="default"):
    while True:
        res = subprocess.run(
            ["kubectl", "get", "job", job_name, "-o", "jsonpath={.status.succeeded}", "-n", namespace],
            capture_output=True, text=True
        )
        if res.stdout.strip() == "1":
            print(f"✅ Job {job_name} finished successfully.")
            return True
        time.sleep(10)


def deploy_server(server_yaml: str, namespace: str):
    ns_flag = ["-n", namespace] if namespace else []
    logging.info("🚀 Deploying server...")
    subprocess.run(["kubectl", "apply", "-f", server_yaml] + ns_flag, check=True)
    # Wait until the pod is Ready
    while True:
        result = subprocess.run(
            ["kubectl", "get", "deploy", "-o", "jsonpath={.items[0].status.readyReplicas}"] + ns_flag,
            capture_output=True,
            text=True,
            )
        if result.stdout.strip() == "1":
            logging.info("✅ Server is ready.")
            break
        logging.info("⏳ Waiting for server to become ready...")
        time.sleep(10)


def shutdown_server(server_yaml: str, namespace: str, server_name: str):
    ns_flag = ["-n", namespace] if namespace else []
    logging.info("🛑 Shutting down server...")
    subprocess.run(["kubectl", "delete", "-f", server_yaml] + ns_flag, check=False)
    subprocess.run(["kubectl", "delete", "service", server_name] + ns_flag, check=False)



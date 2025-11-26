import subprocess
import yaml
from pathlib import Path


def shutdown_from_config(cfg_path: str):
    # Load config
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    name = cfg["name"]

    # Resource names
    client_job = f"vllm-client-{name}"
    server_deployment = f"vllm-server-{name}"
    server_service = f"vllm-server-{name}"

    resources = [
        ("job", client_job),
        ("pod", server_deployment),
        ("service", server_service),
    ]

    for kind, res in resources:
        print(f"Deleting {kind}/{res} ...")
        result = subprocess.run(
            ["kubectl", "delete", kind, res, "-n", cfg['namespace']],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(result.stderr.strip())


def shutdown(type: str, name: str):
    print(f"Shutting down {name} {type}")
    if type == ' client':
        args = ["kubectl", "delete", 'job', name]
    else:
        args = ["kubectl", "delete", 'pod', name]
    result = subprocess.run(
        args,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print(result.stderr.strip())


if __name__ == "__main__":
    # cfg = load_config('scripts/server_client_config.yaml')
    # name = cfg["name"]

    shutdown_from_config('scripts/server_client_config.yaml')

import argparse
import subprocess

from src.kubernetes.yaml_spawner import load_config, render_client, apply_yaml, \
    calculate_hardware_requirements, convert_lists_to_str, wait_for_job

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    cfg = load_config('scripts/server_client_config.yaml')
    cfg = convert_lists_to_str(cfg)
    cfg = calculate_hardware_requirements(cfg)

    client_name = f"vllm-client-{cfg["name"]}"
    client_yaml = render_client(cfg)

    num_runs = 1
    for _ in range(num_runs):
        print(f"Deleting {client_name} ...")
        result = subprocess.run(
            ["kubectl", "delete", 'job', client_name, "-n", cfg['namespace']],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(result.stderr.strip())

        print("🚀 Deploying client job...")
        apply_yaml(client_yaml)
        wait_for_job(f"vllm-client-{cfg['name']}", cfg.get('namespace', "clinibench"))

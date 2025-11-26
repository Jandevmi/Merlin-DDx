import argparse
import subprocess

from src.kubernetes.yaml_spawner import load_config, render_server, render_client, apply_yaml, \
    wait_for_job, calculate_hardware_requirements, convert_lists_to_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    cfg = load_config('scripts/server_client_config.yaml')
    cfg = convert_lists_to_str(cfg)
    cfg = calculate_hardware_requirements(cfg)

    server_yaml = render_server(cfg)
    client_yaml = render_client(cfg)

    print("🚀 Deploying server...")
    apply_yaml(server_yaml)
    print("🚀 Deploying client job...")
    apply_yaml(client_yaml)

    for i in range(3):
        client_name = f"vllm-client-{cfg['name']}"
        print(f'Start Job {i} {client_name}')
        wait_for_job(client_name, cfg.get('namespace', "clinibench"))
        apply_yaml(client_yaml)
        result = subprocess.run(
            ["kubectl", "delete", 'job', client_name, "-n", cfg['namespace']],
            capture_output=True,
            text=True
        )
    #
    if args.shutdown:
        print("🛑 Shutting down server...")
        subprocess.run(["kubectl", "delete", "-f", "-"], input=server_yaml.encode())

import subprocess
import sys

USER = "jfrick"
IMAGE_NAME = "vllm_client"
# IMAGE_NAME = "vllm_server"
DOCKERFILE = f"Dockerfile.{IMAGE_NAME}"
IMAGE_PATH = f"registry.datexis.com/{USER}/{IMAGE_NAME}:latest"

# Build Docker image
print(f"Building Docker image for {IMAGE_NAME} using {DOCKERFILE}...")
build = subprocess.run([
    "docker", "build", "--platform", "linux/amd64",
    "-f", f"k8s/deployment/{DOCKERFILE}",
    "-t", IMAGE_PATH, "."
])

if build.returncode != 0:
    print("Docker build failed!")
    sys.exit(1)

# Push Docker image
print(f"Pushing Docker image to {IMAGE_PATH}...")
push = subprocess.run(["docker", "push", IMAGE_PATH])

if push.returncode != 0:
    print("Docker push failed!")
    sys.exit(1)

print(f"Docker image {IMAGE_PATH} built and pushed successfully!")

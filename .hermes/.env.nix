# Cloud (build.nvidia.com)
hermes chat --provider nvidia --model nvidia/nemotron-3-super-120b-a12b
# Requires: NVIDIA_API_KEY in ~/.hermes/.env

# Local NIM endpoint — override base URL
NVIDIA_BASE_URL=https://127.0.0.1:8085/v1 hermes chat --provider nvidia --model nvidia/nemotron-3-super-120b-a12b

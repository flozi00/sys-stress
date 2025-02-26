docker run --gpus all \
    --shm-size 32g \
    --env-file .env \
    --pull always \
    --rm \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.3-70B-Instruct \
    --host 0.0.0.0 --port 8000 --torchao-config fp8wo \
    --tensor-parallel-size 2 --attention-backend flashinfer --sampling-backend flashinfer \
    --grammar-backend xgrammar --enable-p2p-check --mem-fraction-static 0.2 --context-length 32000

```bash

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    lmsysorg/sglang:latest python3 -m sglang.launch_server --model neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8 --tp 8 --enable-mixed-chunk --context-length 10000 --mem-fraction-static 0.86 --stream-interval 6 --kv-cache-dtype fp8_e5m2 --disable-cuda-graph --host 0.0.0.0 --port 8000

```

```bash

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest --model neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 --max-model-len 8192 --cpu-offload-gb 0 --enable-prefix-caching --num-scheduler-steps 12 --tensor-parallel-size 4 --quantization compressed-tensors --port 8000 --enforce-eager --swap-space 4 --gpu-memory-utilization 0.95

```

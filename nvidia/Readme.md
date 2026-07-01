```bash

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    lmsysorg/sglang:latest python3 -m sglang.launch_server --model neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8 --tp 8 --enable-mixed-chunk --context-length 10000 --mem-fraction-static 0.86 --stream-interval 6 --kv-cache-dtype fp8_e5m2 --disable-cuda-graph --host 0.0.0.0 --port 8000

```

## vLLM single-GPU benchmark

Set `HF_TOKEN` to a Hugging Face token with Llama 3.3 access. The named Docker
volume keeps downloaded model weights across runs.
This uses a pre-quantized 4-bit 70B checkpoint to stress one large data-center
GPU. Lower the token lengths first if a smaller-memory card runs out of memory.

```bash
docker run --rm --runtime nvidia --gpus '"device=0"' \
    --ipc=host \
    --env "HF_TOKEN=$HF_TOKEN" \
    -v vllm-hf-cache:/root/.cache/huggingface \
    --entrypoint vllm \
    vllm/vllm-openai:latest \
    bench throughput \
    --model unsloth/Llama-3.3-70B-Instruct-bnb-4bit \
    --backend vllm \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 256 \
    --num-prompts 512 \
    --max-model-len 4096 \
    --dtype bfloat16
```

Due to inkompitability issues in some cases the grub params needs to be edited.

```bash
sudo nano /etc/default/grub
```

then put to the default linux params:

```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash nokaslr"
```

followed by:
```bash
sudo update-grub
sudo reboot now
```

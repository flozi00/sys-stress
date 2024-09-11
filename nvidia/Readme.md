```bash

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    lmsysorg/sglang:latest python3 -m sglang.launch_server --model neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8 --tp 4 --enable-mixed-chunk --context-length 8000 --mem-fraction-static 0.86 --stream-interval 6 --kv-cache-dtype fp8_e5m2 --disable-cuda-graph --host 0.0.0.0 --port 8000

```

Alternative Driver+Cuda install
```
wget https://developer.download.nvidia.com/compute/cuda/12.6.1/local_installers/cuda_12.6.1_560.35.03_linux.run
sudo sh cuda_12.6.1_560.35.03_linux.run
```

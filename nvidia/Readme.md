```bash

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    lmsysorg/sglang:latest python3 -m sglang.launch_server --model neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8 --tp 8 --enable-mixed-chunk --context-length 10000 --mem-fraction-static 0.86 --stream-interval 6 --kv-cache-dtype fp8_e5m2 --disable-cuda-graph --host 0.0.0.0 --port 8000

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

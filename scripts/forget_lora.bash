CUDA_VISIBLE_DEVICES=0,1 DS_SKIP_CUDA_CHECK=1 accelerate launch \
    --config_file config/accelerate_config.yaml \
     --main_process_port 2216 \
    ./forget.py --config-name forget_lora.yaml \

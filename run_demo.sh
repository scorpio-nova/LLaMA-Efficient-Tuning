LLAMA2_PATH=/data/cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
LORA_PATH=/root/code/LLaMA-Efficient-Tuning/saves/LLaMA2-7B/lora/2023-10-07-16-09-03

python src/web_demo.py \
    --model_name_or_path $LLAMA2_PATH \
    --template default \
    # --finetuning_type lora \
    # --checkpoint_dir $LORA_PATH
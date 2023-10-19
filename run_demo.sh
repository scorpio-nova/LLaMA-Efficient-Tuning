# LLAMA2_PATH=/data/cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
# full finetuned model
# LLAMA2_PATH=/data/xukp/models/llama/llama-2-7b-sft-10-07-18-34
# lora finetuned model
# LORA_PATH=/data/xukp/models/llama/llama-2-7b-lora-10-07-18-51
# solving model
LLAMA2_PATH=/data/xukp/models/llama/llama-2-7b-all_solving_only_steps-10-18-20-25

# LORA_PATH=/root/code/LLaMA-Efficient-Tuning/saves/LLaMA2-7B/lora/2023-10-07-16-09-03
TEMPLATE=vanilla

# python src/cli_demo.py \
python src/web_demo.py \
    --model_name_or_path $LLAMA2_PATH \
    --template $TEMPLATE \
    # --finetuning_type lora \
    # --checkpoint_dir $LORA_PATH
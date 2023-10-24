# LLAMA2_PATH=/data/cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
# full finetuned model
# LLAMA2_PATH=/data/xukp/models/llama/llama-2-7b-sft-10-07-18-34
# lora finetuned model
# LORA_PATH=/data/xukp/models/llama/llama-2-7b-lora-10-07-18-51
# solving model
# LLAMA2_PATH=/root/models/llama-tuned/codellama-34b-lora-solving_only_steps-10-19-10-10
LLAMA2_PATH=/lustre/cache/huggingface/models--codellama--CodeLlama-34b-hf/snapshots/fda69408949a7c6689a3cf7e93e632b8e70bb8ad

# LORA_PATH=/root/code/LLaMA-Efficient-Tuning/saves/LLaMA2-7B/lora/2023-10-07-16-09-03
LORA_PATH=/root/models/llama-tuned/codellama-34b-lora-solving_os_wprefix-10-24-14-51
TEMPLATE=vanilla

# python src/cli_demo.py \
python src/web_demo.py \
    --model_name_or_path $LLAMA2_PATH \
    --template $TEMPLATE \
    --finetuning_type lora \
    --checkpoint_dir $LORA_PATH
TIME=$(date "+%m-%d-%H-%M")
DATASET=solving_only_steps
# set HF_HOME env
# export HF_HOME=/lustre/cache/huggingface
# OUTPUT_DIR=~/models/llama-tuned/llama-2-7b-$DATASET-$TIME
OUTPUT_DIR=~/models/llama-tuned/codellama-34b-$DATASET-$TIME
# origina model 7B
MODEL_NAME_OR_PATH=/data/cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
# code llama
# MODEL_NAME_OR_PATH=/lustre/cache/huggingface/models--codellama--CodeLlama-34b-hf/snapshots/fda69408949a7c6689a3cf7e93e632b8e70bb8ad
# MODEL_NAME_OR_PATH="codellama/CodeLlama-34b-hf"
# model added special tokens
# MODEL_NAME_OR_PATH=/data/xukp/models/llama/llama2-7b-added_special_tokens   # added special tokens
TEMPLATE=vanilla
VAL_SIZE=0.05
NUM_GPUS=8
EPOCHS=3
# accelerate launch src/train_bash.py \

# deepspeed --num_gpus $NUM_GPUS --master_port=9901 src/train_bash.py \
deepspeed --hostfile hostfile.txt src/train_bash.py \
    --deepspeed "ds_config_stage2_off_optim&param.json" \
    --stage sft \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --do_train True \
    --overwrite_cache False \
    --finetuning_type full \
    --template $TEMPLATE \
    --dataset_dir data \
    --dataset $DATASET \
    --cutoff_len 8192 \
    --learning_rate 5e-05 \
    --num_train_epochs $EPOCHS \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 200 \
    --warmup_steps 0 \
    --flash_attn False \
    --lora_rank 8 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --resume_lora_training True \
    --output_dir $OUTPUT_DIR \
    --fp16 True \
    --plot_loss True \
    --val_size $VAL_SIZE \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --load_best_model_at_end True
TIME=$(date "+%m-%d-%H-%M")
DATASET=hint
TEMPLATE=hint

# wandb
export WANDB_PROJECT=xukp20-generate-hint

# set HF_HOME env
# export HF_HOME=/lustre/cache/huggingface
OUTPUT_DIR=~/models/llama-tuned/mistral-7b-generate-hint-$TEMPLATE-$DATASET-$TIME
# OUTPUT_DIR=~/models/llama-tuned/llama-7b-$DATASET-$TIME

# Mistral 7B
MODEL_NAME_OR_PATH="/data/cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/5e9c98b96d071dce59368012254c55b0ec6f8658"
# llama 7B
# MODEL_NAME_OR_PATH=/data/cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9

VAL_SIZE=0.01
NUM_GPUS=8
# LR=5e-5
LR=1e-5
EPOCHS=2
CUTOFF_LEN=8192

# accelerate launch src/train_bash.py \
# deepspeed --hostfile hostfile.txt src/train_bash.py \
deepspeed --num_gpus $NUM_GPUS --master_port=9901 src/train_bash.py \
    --deepspeed "/root/code/LLaMA-Efficient-Tuning/ds_config.json" \
    --stage sft \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --do_train True \
    --overwrite_cache False \
    --finetuning_type full \
    --template $TEMPLATE \
    --dataset_dir data \
    --dataset $DATASET \
    --cutoff_len $CUTOFF_LEN \
    --learning_rate $LR \
    --num_train_epochs $EPOCHS \
    --max_samples 100000 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 200 \
    --warmup_steps 0 \
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
    --report_to wandb 
    # --load_best_model_at_end True
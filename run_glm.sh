TIME=$(date "+%m-%d-%H-%M")
DATASET=solving_nve_nin_nups
TEMPLATE=alpaca

# wandb
export WANDB_PROJECT=xukp20-chatglm-sft

# set HF_HOME env
# export HF_HOME=/lustre/cache/huggingface
OUTPUT_DIR=~/models/llama-tuned/chatglm3-6b-$TEMPLATE-$DATASET-$TIME
# OUTPUT_DIR=~/models/llama-tuned/codellama-34b-$DATASET-$TIME

# GLM 6B
MODEL_NAME_OR_PATH="/lustre/cache/huggingface/hub/models--THUDM--chatglm3-6b-base/snapshots/c6561772567b9d13567e5372225ee2ec22379a9b"

VAL_SIZE=0.01
NUM_GPUS=8
# LR=5e-5
LR=2e-5
EPOCHS=3
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
    --fp16 \
    --plot_loss True \
    --val_size $VAL_SIZE \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --report_to wandb 
    # --load_best_model_at_end True

HINT_MODEL=/root/models/llama-tuned/llama-7b-hint-10-31-00-09
TEMPLATE=hint

# python src/cli_demo.py \
python src/web_demo.py \
    --model_name_or_path $HINT_MODEL \
    --template $TEMPLATE
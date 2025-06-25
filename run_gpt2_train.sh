#Instruction Fine-Tune params:

ignore_index=-100
num_epochs=5
trainable_layers=None
top_k=5
temp=1
max_new_tokens=100
context_length=1024
val_split=0.05
train_split=0.85
batch_size=8
seed=123
pre_save_model=None
load_weights=True
peft_type=None
mask_instruction=True
python gpt_trainingPipeline.py \
  --experiment_name 'IFT_Exp' \
  --base_modelName 'gpt2_355M' \
  --data_path 'instruction-data-NT.json' \
  --training_type 'IFT' \
  --peft_type $peft_type \
  --load_weights $load_weights \
  --pre_save_model $pre_save_model \
  --model_name 'gpt2_355M_instruct_FineTuned' \
  --tokenizer 'tiktoken' \
  --seed $seed \
  --batch_size $batch_size \
  --train_split $train_split \
  --val_split $val_split \
  --context_length $context_length \
  --max_new_tokens $max_new_tokens \
  --temp $temp \
  --top_k $top_k \
  --trainable_layers $trainable_layers \
  --num_epochs $num_epochs \
  --max_training_length 'longest_training_example' \
  --prompt_style 'alpaca' \
  --ignore_index $ignore_index \
  --mask_instruction $mask_instruction \
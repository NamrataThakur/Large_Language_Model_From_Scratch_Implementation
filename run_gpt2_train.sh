#Instruction Fine-Tune params:
#If you have to use any variable with None value, then skip that variable in the script below. 
#The Default value will be used in that case. Else None will go as 'None' and error will rise in the code.

ignore_index=-100
num_epochs=2
trainable_layers=None
#top_k=5
temp=0.0
max_new_tokens=256
context_length=1024
val_split=0.05
train_split=0.85
batch_size=8
seed=123
pre_save_model=None
load_weights=True
peft_type=None
mask_instruction=False
dropout_rate=0.1
eos_id=50256
python gpt_trainingPipeline.py \
  --experiment_name 'IFT_Exp_NonMaskedInst' \
  --base_modelName 'gpt2_355M' \
  --data_path 'instruction-data-NT.json' \
  --training_type 'IFT' \
  --load_weights $load_weights \
  --model_name 'gpt2_355M_nonMaskedInstruct_FineTuned' \
  --tokenizer 'tiktoken' \
  --seed $seed \
  --batch_size $batch_size \
  --train_split $train_split \
  --val_split $val_split \
  --context_length $context_length \
  --max_new_tokens $max_new_tokens \
  --temp $temp \
  --dropout_rate $dropout_rate \
  --num_epochs $num_epochs \
  --max_training_length 'longest_training_example' \
  --prompt_style 'alpaca' \
  --ignore_index $ignore_index \
  --mask_instruction $mask_instruction \
  --eos_id $eos_id
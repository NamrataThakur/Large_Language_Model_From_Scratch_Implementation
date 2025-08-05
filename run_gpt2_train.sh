#If you have to use any variable with None value, then skip that variable in the script below. 
#The Default value will be used in that case. Else None will go as 'None' and error will rise in the code.

#Uncomment the argument block for the corresponding training type:

#Instruction Fine-Tune (IFT) params:

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
pre_save_model="gpt2_355M_MaskedInstruct_FineTuned_v1.pth"
load_weights=True
peft_type="lora"
mask_instruction=True
dropout_rate=0.1
eos_id=50256
use_warmup=True
use_gradient_clip=True
warmup_steps=20
initial_lr=1e-5
min_lr=1e-5
lora_rank=16
lora_alpha=16
python gpt_trainingPipeline.py \
  --experiment_name 'IFT_Exp_MaskedInst_v1_LoRA' \
  --base_modelName 'gpt2_355M' \
  --data_path 'instruction-data-NT.json' \
  --training_type 'IFT' \
  --load_weights $load_weights \
  --model_name 'gpt2_355M_MaskedInstruct_FineTuned_v1_LoRA' \
  --tokenizer 'tiktoken' \
  --seed $seed \
  --batch_size $batch_size \
  --train_split $train_split \
  --val_split $val_split \
  --context_length $context_length \
  --max_new_tokens $max_new_tokens \
  --pre_save_model $pre_save_model \
  --temp $temp \
  --peft_type $peft_type \
  --dropout_rate $dropout_rate \
  --num_epochs $num_epochs \
  --max_training_length 'longest_training_example' \
  --eos_id $eos_id \
  --prompt_style 'alpaca' \
  --ignore_index $ignore_index \
  --use_warmup $use_warmup \
  --mask_instruction $mask_instruction \
  --use_gradient_clip $use_gradient_clip \
  --warmup_steps $warmup_steps \
  --initial_lr $initial_lr \
  --min_lr $min_lr \
  --lora_rank $lora_rank \
  --lora_alpha $lora_alpha 


# Classification Supervised Fine-Tune (SFT) params:

# ignore_index=-100
# num_epochs=5
# trainable_layers=last_block
# top_k=5
# temp=0.0
# max_new_tokens=256
# context_length=1024
# val_split=0.1
# train_split=0.7
# batch_size=8
# seed=123
# pre_save_model="gpt2_124M_SFT_Spam_v2.pth"
# load_weights=True
# peft_type="lora"
# mask_instruction=False
# dropout_rate=0.0
# eos_id=50256
# use_warmup=True
# use_gradient_clip=True
# warmup_steps=20
# initial_lr=1e-5
# min_lr=1e-5
# lora_rank=16
# lora_alpha=16
# python gpt_trainingPipeline.py \
#   --experiment_name 'SFT_Exp_ALL_v2_LoRA_noGC' \
#   --base_modelName 'gpt2_124M' \
#   --data_path 'sms_spam_collection.zip' \
#   --training_type 'SFT' \
#   --load_weights $load_weights \
#   --model_name 'gpt2_124M_SFT_Spam_v2_LoRA_noGC' \
#   --tokenizer 'tiktoken' \
#   --seed $seed \
#   --batch_size $batch_size \
#   --pre_save_model $pre_save_model \
#   --peft_type $peft_type \
#   --train_split $train_split \
#   --val_split $val_split \
#   --context_length $context_length \
#   --max_new_tokens $max_new_tokens \
#   --temp $temp \
#   --trainable_layers $trainable_layers \
#   --dropout_rate $dropout_rate \
#   --num_epochs $num_epochs \
#   --max_training_length 'longest_training_example' \
#   --top_k $top_k \
#   --eos_id $eos_id  \
#   --warmup_steps $warmup_steps \
#   --initial_lr $initial_lr \
#   --min_lr $min_lr \
#   --lora_rank $lora_rank \
#   --lora_alpha $lora_alpha \


#Preference Fine-Tune (PFT) params:

# ignore_index=-100
# num_epochs=1
# trainable_layers=None
# #top_k=5
# temp=0.0
# max_new_tokens=40
# context_length=1024
# val_split=0.05
# train_split=0.85
# batch_size=8
# seed=123
# pre_save_model='gpt2_355M_MaskedInstruct_FineTuned_v1.pth'
# load_weights=True
# peft_type=None
# mask_instruction=True
# dropout_rate=0.0
# eos_id=50256
# use_warmup=False
# use_gradient_clip=False
# warmup_steps=20
# initial_lr=1e-5
# min_lr=1e-5
# beta=0.1
# python gpt_trainingPipeline.py \
#   --experiment_name 'PFT_Exp_MaskedInst_v2' \
#   --base_modelName 'gpt2_355M' \
#   --data_path 'instruction-data-with-preference.json' \
#   --training_type 'PFT' \
#   --load_weights $load_weights \
#   --model_name 'gpt2_355M_MaskedInstruct_PFT_v2' \
#   --tokenizer 'tiktoken' \
#   --pre_save_model $pre_save_model \
#   --seed $seed \
#   --batch_size $batch_size \
#   --train_split $train_split \
#   --val_split $val_split \
#   --context_length $context_length \
#   --max_new_tokens $max_new_tokens \
#   --temp $temp \
#   --dropout_rate $dropout_rate \
#   --num_epochs $num_epochs \
#   --max_training_length 'model_context_length' \
#   --eos_id $eos_id \
#   --prompt_style 'alpaca' \
#   --ignore_index $ignore_index \
#   --mask_instruction $mask_instruction \
#   --warmup_steps $warmup_steps \
#   --initial_lr $initial_lr \
#   --min_lr $min_lr \
#   --beta $beta

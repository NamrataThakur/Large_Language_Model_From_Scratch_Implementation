#If you have to use any variable with None value, then skip that variable in the script below. 
#The Default value will be used in that case. Else None will go as 'None' and error will rise in the code.

#Pre-Training params:

num_epochs=1
top_k=3
temp=0.0
max_new_tokens=100
context_length=512
vocab_size=50257
embedding_dimension=512
num_heads=8
num_layers=8
qkv_bias=False
eval_batchSize=256
eval_freq=64
weight_decay=0.1
beta1=0.9
beta2=0.95
val_split=0.05
train_split=0.85
batch_size=16
target_batch_size=1024 
seed=123
dropout_rate=0.3
eos_id=50256
use_warmup=True
use_gradient_clip=True
warmup_steps=0.05
initial_lr=3e-4
min_lr=1e-06
learning_rate=8e-4
rms_eps=1e-6
rms_bias=True
theta_base=10000.0
num_experts=4
num_active_experts=2
num_kv_groups=1
ff_hidden_dim=400
arch_type='original'
kv_cache=True
moe_noise=True
python gpt_pretrainingPipeline.py \
  --experiment_name 'Pre-Train_Exp_CustomConfig_ORGarch_S_v2' \
  --data_path 'tinystories' \
  --model_type 'custom' \
  --arch_type $arch_type \
  --model_name 'gpt2_ORG_preTrain_S_v2' \
  --tokenizer 'tiktoken' \
  --seed $seed \
  --batch_size $batch_size \
  --eval_batchSize $eval_batchSize \
  --eval_freq $eval_freq \
  --target_batch_size $target_batch_size \
  --train_split $train_split \
  --val_split $val_split \
  --optimizer 'AdamW' \
  --context_length $context_length \
  --vocab_size $vocab_size \
  --embedding_dimension $embedding_dimension \
  --max_training_length 'model_context_length' \
  --num_heads $num_heads \
  --num_layers $num_layers \
  --weight_decay $weight_decay \
  --beta1 $beta1 \
  --beta2 $beta2 \
  --rms_eps $rms_eps \
  --rms_bias $rms_bias \
  --theta_base $theta_base \
  --num_experts $num_experts \
  --num_active_experts $num_active_experts \
  --num_kv_groups $num_kv_groups \
  --ff_hidden_dim $ff_hidden_dim \
  --max_new_tokens $max_new_tokens \
  --temp $temp \
  --top_k $top_k \
  --dropout_rate $dropout_rate \
  --num_epochs $num_epochs \
  --eos_id $eos_id \
  --initial_lr $initial_lr \
  --warmup_steps $warmup_steps \
  --min_lr $min_lr \
  --learning_rate $learning_rate \
  --use_gradient_clip $use_gradient_clip \
  --use_warmup $use_warmup \
  --moe_noise $moe_noise \
   --kv_cache $kv_cache \
  --qkv_bias $qkv_bias \


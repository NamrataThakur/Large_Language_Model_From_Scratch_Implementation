import torch
from .linear_lora import LinearWithLORA
from .classWise_linear_lora import ClassLinearWithLORA
from .lora import LORA


def lora_parameterization(model, rank = 16, alpha = 16):

    lora_rank = rank
    lora_alpha = alpha

    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):

            #Replace the "Linear" layers with the LoRA layers (LoRa layers --> Linear + LoRa)
            setattr(model, name, LinearWithLORA(linear=module, rank=lora_rank, alpha=lora_alpha))
        else:
            lora_parameterization(model=module, rank = lora_rank, alpha = lora_alpha)



def freeze_model(model):
    params_orig = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for p in model.parameters():
        p.requires_grad = False
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total trainable parameters after model freeze : ', params)

    return params_orig



def classWise_lora_parameterizaton(model, rank, alpha, class_names):

    lora_rank = rank
    lora_alpha = alpha
    class_names = class_names

    for name, module in model.named_children():

        if isinstance(module, torch.nn.Linear):

            adapter_names = {
                k : LORA(feature_in=module.in_features, feature_out=module.out_features, rank=lora_alpha, alpha=lora_alpha)
                for k in class_names
            }

            #Replace the "Linear" layers with the new class-wise LoRA layers (LoRA layers --> Linear + classWiseLoRA)
            setattr(model, name, ClassLinearWithLORA(linear=module, adapter_names=adapter_names))

        else:
            classWise_lora_parameterizaton(model=module, rank=lora_rank, alpha=lora_alpha, class_names=class_names)
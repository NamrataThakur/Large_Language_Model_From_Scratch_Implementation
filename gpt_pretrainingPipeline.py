import argparse
import logging
import os
import time
from datetime import datetime
import warnings
import pprint
import sys
from tqdm import tqdm
import tiktoken
import torch
import urllib.request
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from importlib.metadata import version
import numpy as np
import zipfile
from pathlib import Path
import pandas as pd
import json
from datasets import load_dataset

import configparser
warnings.filterwarnings("ignore")

#Load the class for model config
from gpt_ClassificationFT.gpt2_model_customConfig import GPT2_CustomConfig
from gpt_ClassificationFT.gpt2_model_config import GPT2_ModelConfig

#Load the Model Architecture
from transformer_blocks.gpt2 import GPT2
from transformer_blocks.gpt2_gqa import GQAGPT2
from transformer_blocks.gpt2_moe import MoEGPT2

#Load the pre-training class
from gpt_Pretraining.gpt2_pretrain import GPT2_PreTrain
from gpt_Pretraining.apply_weights import apply_weights

#Dataloaders for Pre-Training
from dataloader.Pretraining.customGPT2_pretrainDataloader import GPTCustomPretrainDataloader

#Load the class for plotting the graphs
from gpt_Pretraining.plot_metrics import Plots


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    #Add the parsing arguments:
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='dummy_Exp',
        help=('name of the experiment and its corresponding log file')
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default='tinystories',
        help=('The name of the training data file. This file is present under the folder "data". ' \
                                                    'Extension accepted: .txt, .csv, .tsv, .zip, .json ' \
                'OR the dataset name to be downloaded from internet. Example: tinystories, tinystories_instruct, etc'  )  
    )

    parser.add_argument(
        '--model_type',
        type=str,
        default='original',
        help=('Whether to use original GPT2, or Custom GPT2 Architecture. ' \
                                'Options: original or custom')
    )

    parser.add_argument(
        '--base_modelName',
        type=str,
        default='gpt2_124M',
        help=('base GPT2 model. To be used ONLY if model_type is "original". ' \
                        'Options: gpt2_124M, gpt2_355M, gpt2_774M, gpt2_1558M')
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default='gpt2',
        help=('name of the trained model to be saved. ')
    )

    parser.add_argument(
        '--pre_save_model',
        type=str,
        default=None,
        help=('name of the pre-saved model that will be loaded for further training. '
                        'Options: None (if None, then the model will trained from scratch) ' \
                                ' Or name of the previously trained model)')
    )

    parser.add_argument(
        '--tokenizer',
        type=str,
        default='tiktoken',
        help=('Name of the tokenizer to be used. Options: tiktoken (use openai tiktoken package), ' \
                                                        'customBPE')
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help=('random seed')
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help=('batch_size per gpu')
    )

    parser.add_argument(
        '--target_batch_size',
        type=int,
        default=16,
        help=('To be used for gradient accumulation step. Effective batch_size per gpu. Preferrably double of the batch_size value.')
    )

    parser.add_argument(
        '--train_split',
        type=float,
        default=0.70,
        help=('percentage of the data to be considered for training. ' \
                'To be used if the training dataset doesnt contain train-valid split')
    )

    parser.add_argument(
        '--val_split',
        type=float,
        default=0.10,
        help=('percentage of the data to be considered for validation'\
                'To be used if the training dataset doesnt contain train-valid split')
    )

    parser.add_argument(
        '--optimizer',
        type=str,
        default='AdamW',
        help=('Name of the optimizer to be used. Options: AdamW (recommended), ' \
                                                        'SGD')
    )

    parser.add_argument(
        '--context_length',
        type=int,
        default=1024,
        help=('context length of the attention block to be considered.')
    )

    parser.add_argument(
        '--vocab_size',
        type=int,
        default=50257,
        help=('Total Vocab Size that model will be trained on.')
    )

    parser.add_argument(
        '--embedding_dimension',
        type=int,
        default=768,
        help=('embedding dimension in the tokenization layer.')
    )

    parser.add_argument(
        '--num_heads',
        type=int,
        default=12,
        help=('number of heads of the attention block to be considered.')
    )

    parser.add_argument(
        '--num_layers',
        type=int,
        default=12,
        help=('number of layers of the attention block to be considered.')
    )

    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.0,
        help=('Dropout rate value.')
    )

    parser.add_argument(
        '--qkv_bias',
        type=bool,
        default=False,
        help=('Bias to be used or not.')
    )

    parser.add_argument(
        '--ff_hidden_dim',
        type=int,
        default=768,
        help=('The hidden dimension of the Feedforward block.')
    )

    parser.add_argument(
        '--eval_batchSize',
        type=int,
        default=5,
        help=('Total batch size used for evaluation.')
    )

    parser.add_argument(
        '--eval_freq',
        type=int,
        default=5,
        help=('The interval at which evaluation will occur.')
    )

    parser.add_argument(
        '--kv_cache',
        type=bool,
        default=False,
        help=('KV Caching to be used during inference')
    )
    

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-1,
        help=('Weight decay param of optimizer.')
    )

    parser.add_argument(
        '--beta1',
        type=float,
        default=0.9,
        help=('Hyper-param if optimizer is AdamW')
    )

    parser.add_argument(
        '--beta2',
        type=float,
        default=0.95,
        help=('Hyper-param if optimizer is AdamW')
    )

    parser.add_argument(
        '--rms_eps',
        type=float,
        default=1e-6,
        help=('Epsilon used in RMS normalization')
    )

    parser.add_argument(
        '--rms_bias',
        type=bool,
        default=False,
        help=('Shift used in RMS normalization')
    )

    parser.add_argument(
        '--theta_base',
        type=float,
        default=10000.0,
        help=('Theta used in RoPE')
    )

    parser.add_argument(
        '--num_kv_groups',
        type=int,
        default=0,
        help=('Key-Value groups for attention')
    )

    parser.add_argument(
        '--num_experts',
        type=int,
        default=8,
        help=('Theta used in RoPE')
    )

    parser.add_argument(
        '--num_active_experts',
        type=int,
        default=2,
        help=('Theta used in RoPE')
    )

    parser.add_argument(
        "--max_training_length",
        type=str,
        default="model_context_size",
        help=("The context length of the data inputs. Options: model_context_size or custom integer.")
    )

    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=100,
        help=('Maximum number of tokens generated')
    )

    parser.add_argument(
        '--temp',
        type=float,
        default=0.0,
        help=('Temperature for generation. Options: 0 - 2, Closer to 0, more deterministic generation, closer to 2, more creative generation.')
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=None,
        help=('Top k tokens considered for next token prediction. Options: None (It will consider all the tokens in the final generation)')
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help=("Number of training epochs.")
    )

    parser.add_argument(
        "--eos_id",
        type=int,
        default=50256,
        help=("Token id for end of string.")
    )

    parser.add_argument(
        "--arch_type",
        type=str,
        default='original',
        help=("Model architecture to be used for pre-training. " \
                            "Options: original (gpt2 architecture with smaller size)," \
                                    " GQA (custom architecture with GQA and FF block)," \
                                    " MOE (custom architecture with GQA and MOE block)")
    )

    parser.add_argument(
        "--moe_noise",
        type=bool,
        default=True,
        help=("Whether to use gaussian noise to the MoE Router to add stability. Options: True (Recommended), False")
    )

    parser.add_argument(
        "--use_warmup",
        type=bool,
        default=False,
        help=("Whether to use initial learning rate warmup. Options: True , False")
    )


    parser.add_argument(
        "--use_gradient_clip",
        type=bool,
        default=False,
        help=("Whether to clip gradients to avoid exploding gradient problem. Options: True , False")
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=float,
        default=0.01,
        help=("Prcentage of total steps to be taken for warmup. Generally 0.01")
    )

    parser.add_argument(
        "--initial_lr",
        type=float,
        default=3e-05,
        help=("Intial LR value to start the warmup from.")
    )   

    # Calculation minimum LR as 0.1 * max learning rate (according to Karpathy)
    # parser.add_argument(
    #     "--min_lr",
    #     type=float,
    #     default=1e-6,
    #     help=("Minimum LR value to achieve using cosine annealing (decay.)")
    # )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help=("Maximum LR to reach before cosine annealing will trigger.")
    )


    args = parser.parse_args()
    torch.manual_seed(args.seed)

    try:
        config = configparser.ConfigParser()
        config.read("config.ini")

        DATA_FOLDER = config["PATHS"]["DATA_FOLDER"]
        LOG_FOLDER = config["PATHS"]["LOG_FOLDER"]
        MODEL_ROOT_FOLDER = config['PATHS']['MODEL_ROOT_FOLDER']

        #Check if model root folder is present, else create it:
        if not os.path.exists(MODEL_ROOT_FOLDER):
            os.mkdir(MODEL_ROOT_FOLDER)

        #Add the logging functionality:
        if not os.path.exists(LOG_FOLDER):
            os.mkdir(LOG_FOLDER)

        logging_path = os.path.join(LOG_FOLDER,args.experiment_name+'_'+str(datetime.now().timestamp())+'.log')
        print(logging_path)
        
        logging.basicConfig(
            filename=logging_path,
            level=logging.INFO,
            format='[%(asctime)s.%(msecs)03d] %(message)s',
            filemode='w'
        )

        logger = logging.getLogger()
        logger.info(str(args))
    
    except Exception as e:
        print("Exception while reading config file : ", e)

    #Load the tokenizer:
    tokenizer = args.tokenizer
    if tokenizer == 'tiktoken':
        tokenizer = tiktoken.get_encoding("gpt2")

    start_time = time.time()

    #Load the model config:
    try:
        if args.model_type == 'original':
            logger.info('Original GPT2 configuration to be used for pre-training..!')

            m_config = GPT2_ModelConfig()
            gpt2_config = m_config.load_model_config(model_name=args.base_modelName, drop_rate=args.dropout_rate,
                                                    context_length=args.context_length)
            gpt2_baseInst = GPT2(gpt2_config)
            gpt2_baseInst.eval()
            logger.info(f'Configuration of the original {args.base_modelName} base model loaded..!')

        else:

            logger.info('Custom GPT2 configuration to be used for pre-training..!')

            config_dict = {
                'vocab_size' :  args.vocab_size,
                'embedding_dimension':args.embedding_dimension,
                'num_heads':args.num_heads,
                'context_length':args.context_length,
                'dropout':args.dropout_rate,
                'qkv_bias':args.qkv_bias,
                'num_layers':args.num_layers,
                'ff_hidden_dim':args.ff_hidden_dim,
                'rms_eps':args.rms_eps,
                'rms_bias':args.rms_bias,
                'theta_base':args.theta_base,
                'num_kv_groups':args.num_kv_groups,
                'num_experts':args.num_experts,
                'num_active_experts':args.num_active_experts,
                'moe_noise':args.moe_noise,

                'weight_decay' : args.weight_decay,
                'beta1' : args.beta1,
                'beta2' : args.beta2
            }

            gpt2_config = GPT2_CustomConfig(config_dict)

            logger.info(f'The custom config of the model to be trained : {gpt2_config.config} ')

            if args.arch_type == 'original':
                gpt2_baseInst = GPT2(gpt2_config.config)
            elif args.arch_type == 'GQA':
                gpt2_baseInst = GQAGPT2(gpt2_config.config)
            else:
                gpt2_baseInst = MoEGPT2(gpt2_config.config)

            logger.info(f'Architecture Type :{args.arch_type}')
            gpt2_baseInst.eval()
            logger.info(f'Configuration of the custom GPT2 model loaded..!')
        
        

    except Exception as e:
        logger.error(f"Error in loading the model config class {e}")
        raise Exception(f"Error in loading the model config class {e}")
    

    try:
        
        if args.data_path == 'tinystories':
            data = load_dataset("roneneldan/TinyStories")
            train_df = data['train']
            val_df = data['validation']
            logger.info(f'Total records in Train Data: {len(train_df)} , Validation Data : {len(val_df)}')
        
        else:
            print('TO DO')

        logger.info('------------- Data Ingestion Completed. -------------')

        logger.info("Loading the dataset class for pre-training...")       
        
        if args.max_training_length == 'model_context_length':
            max_seq_length = gpt2_config.config['context_length']

        else:
            max_seq_length = int(args.max_training_length)
            
        
        assert max_seq_length <= gpt2_config.config['context_length'], (
                                    f"Max training sequence ({args.max_training_length}) cannot be more"
                                    f" than model context length of {gpt2_config.config['context_length']}" 
                                    )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f'Device Available: {device} .. ! ')
        
            
        train_dataLoader, train_total_tokens = GPTCustomPretrainDataloader(train_df, device=device, 
                                                       pad_token = args.eos_id,
                                                       max_seq_length=max_seq_length,
                                                       batch_size=args.batch_size,
                                                       tokenizer = args.tokenizer,
                                                       shuffle=True, drop_last=True,num_workers=0)
        
        val_dataLoader, val_total_tokens = GPTCustomPretrainDataloader(val_df, device=device, 
                                                    pad_token = args.eos_id,
                                                    max_seq_length=max_seq_length,
                                                    batch_size=args.batch_size,
                                                    tokenizer = args.tokenizer,
                                                    shuffle=True, drop_last=True,num_workers=0)
    
        
        
            
        #Print the dataloader contents to confirm correct format:
        logger.info('************** TRAIN DATALOADER ****************************')
        logger.info(f'Length of Train Dataloader (number of batches): {len(train_dataLoader)}')
        logger.info(f'Total Train Tokens : {train_total_tokens}')
        for x,y in train_dataLoader:
            logger.info(f'{x.shape}, {y.shape}')
            break
            

        logger.info('************** VAL DATALOADER ****************************')
        logger.info(f'Length of Val Dataloader (number of batches): {len(val_dataLoader)}')
        logger.info(f'Total Validation Tokens : {val_total_tokens}')
        for x,y in val_dataLoader:
            logger.info(f'{x.shape}, {y.shape}')
            break

        logger.info('Dataloaders created successfully for pre-training task..!')
        logger.info('---------------------------------------------------------')
        
    except Exception as e:
        logger.error(f'Error in loading file and creating dataloader:: {e}')
        raise Exception(f'Error in loading file and creating dataloader:: {e}')
    

    if args.pre_save_model is None:
        
        logger.info(f'Model will be trained from scratch..!')

        logger.info(f'Loading the model with random weights for training..!')

        gpt2_baseInst = apply_weights(gpt2_baseInst)
        
        logger.info(f'Model loaded with random weights for training..!')

    else:

        try:
            logger.info(f'Loading the weights of the model : {args.pre_save_model}..!')
            model_path = os.path.join(MODEL_ROOT_FOLDER,args.pre_save_model)
            print(model_path)
            logger.info(f'Model present in the path: {model_path}')

            #Model and Optimizer are saved in the path. Loading only the model for fine-tuning:
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
            gpt2_baseInst.load_state_dict(checkpoint['model'] )

            gpt2_baseInst.eval()

            logger.info('Model weights loaded successfully..!')

        except Exception as e:
                logger.error(f'Error in loading model weights : {e}')
                raise Exception(f'Error in loading model weights : {e}')
        

    try:
        params = sum(p.numel() for p in gpt2_baseInst.parameters())/1e6
        logger.info(f'Model Size : {params} millions (M)')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f'Device Available: {device} .. ! ')
        
        gpt2_baseInst.to(device)
        logger.info(f'Training Stage : Model sent to {device} for fine-tuning..!')

        torch.manual_seed(args.seed)

        logger.info(f'Training Stage : Training of the model started ..!')

        start_context = "Once upon a time,"

        #Maximum LR value is given in this optimizer 'lr' param. 
        optimizer = torch.optim.AdamW(gpt2_baseInst.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                                      betas = (args.beta1, args.beta2), eps=1e-8)

        epochs = args.num_epochs

        min_lr = args.learning_rate * 0.1
        logger.info(f"Minimum LR : {min_lr}")

        #Note: If gradient accumulation has to happen at every step, then target_batch_size == batch_size
        gradient_accumulation_steps = int(args.target_batch_size // args.batch_size)
        logger.info(f'Gradient Accumulation Steps : {gradient_accumulation_steps}')

        #Pre-Train Model Save Path:
        save_model_name = args.model_name + '.pth'
        save_model_path = os.path.join(MODEL_ROOT_FOLDER,save_model_name)

        gpt2_trainer = GPT2_PreTrain(model=gpt2_baseInst, 
                            optimizer=optimizer,
                            train_dataLoader=train_dataLoader,
                            test_dataLoader=val_dataLoader,
                            num_epochs=epochs,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            global_batch_size=args.target_batch_size,
                            eval_batchSize=args.eval_batchSize, 
                            eval_freq=args.eval_freq,
                            device=device,
                            start_context=start_context,
                            max_new_tokens=args.max_new_tokens,
                            log_path=logger,                        #Pass the logger object instead of logging path
                            warmup_steps=args.warmup_steps,
                            initial_lr=args.initial_lr,
                            min_lr=min_lr,
                            use_warmup=args.use_warmup,
                            use_gradient_clip=args.use_gradient_clip,
                            kv_cache=args.kv_cache,
                            arch_type=args.arch_type
                            ) 

        train_losses, test_losses, track_tokens_seen, track_lr, total_steps = gpt2_trainer.train(model_save_path=save_model_path, temp=args.temp, top_k=args.top_k,  
                                                                            eos_id = args.eos_id)

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"Training completed in {execution_time_minutes:.2f} minutes.")
        logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")
    
        logger.info(f"BEST Pre-trained custom model saved in {save_model_path}..!")
    
    except Exception as e:
        logger.error(f'Error in fine-tuning stage : {e}')
        raise Exception(f'Error in fine-tuning stage : {e}')
    
    try:
            logger.info(f'Saving the plots of the metrics tracked ..!')
            max_step = max(total_steps)
            epochs_tensor = torch.linspace(0, max_step, len(train_losses))
            plt = Plots(track_tokens_seen, epochs_tensor, train_losses, test_losses)
            plt.plots('Loss', args.experiment_name,  xlabel="Steps")

            if args.use_warmup:
                plt.plot_lrs(track_lr, label='Learning Rate', type=args.experiment_name)

            end_time = time.time()
            execution_time_minutes = (end_time - start_time) / 60
            print(f"Pipeline completed in {execution_time_minutes:.2f} minutes.")
            logger.info(f"Pipeline completed in {execution_time_minutes:.2f} minutes.")

    except Exception as e:
        logger.error(f'Error in model evaluation stage : {e}')
        raise Exception(f'Error in model evaluation stage : {e}')
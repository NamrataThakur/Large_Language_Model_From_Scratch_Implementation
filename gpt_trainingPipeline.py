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
from imblearn.over_sampling import SMOTE, ADASYN
import math
from tensorboardX import SummaryWriter
import json
import configparser
warnings.filterwarnings("ignore")


#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from functools import partial

#Load the Model Architecture
from transformer_blocks.gpt2 import GPT2

#Load the text generation class
from gpt_Pretraining.text_generation import Text_Generation

#Load the class for plotting the graphs
from gpt_Pretraining.plot_metrics import Plots

#Load the class containing metrics functions (loss, accuracy etc)
from gpt_Pretraining.metrics import Metrics

#Load the pre-training class
from gpt_Pretraining.gpt2_pretrain import GPT2_PreTrain

#Load the classification (supervised) finetuning class
from gpt_ClassificationFT.gpt2_classificationFT import GPT2_ClassificationFineTune

#Load the class for model config
from gpt_ClassificationFT.gpt2_model_config import GPT2_ModelConfig
from gpt_PreferenceFT.gpt2_preferenceFT import GPT2_PreferenceFineTune

#Utilities for model weight loading
from model_utils.gpt_download import download_and_load_gpt2
from model_utils.load_model_weights import *

#Dataloaders for Pre-Training, SFT, Instruct FT, Preference FT
from dataloader.Pretraining.gpt2_datasetSplit import dataset_split
from dataloader.Pretraining.gpt2_pretrainDataloader import GPTCustomDataloader
from dataloader.Classification_finetuning.gpt2_classificationDataloader import GPTCustomSFTDataloader
from dataloader.Instruction_finetuning.gpt2_instructDataloader import GPTCustomInstructDataloader
from dataloader.Instruction_finetuning.gpt2_instructDataFormat import *
from dataloader.Preference_finetuning.gpt2_preferenceDataloader import GPTCustomPreferenceDataloader
from dataloader.Preference_finetuning.gpt2_preferenceDataFormat import analyse_preferenceTuning
from gpt_ClassificationFT.data_preprocessing import csv_preproccessing

#LoRA classes and functions:
from parameter_efficient_training.apply_lora import *
from parameter_efficient_training.linear_lora import LinearWithLORA
from parameter_efficient_training.lora import LORA



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
        '--base_modelName',
        type=str,
        default='gpt2_124M',
        help=('base GPT2 model. Options: gpt2_124M, gpt2_355M, gpt2_774M, gpt2_1558M')
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default='sms_spam_collection.zip',
        help=('The name of the training data file. This file is present under the folder "data". Extension accepted: .txt, .csv, .tsv, .zip, .json ')
    )

    parser.add_argument(
        '--training_type',
        type=str,
        default='SFT',
        help=("the pipeline to be run for what Type of training. "
                "Options: pre-train (Pre-Training Original GPT2 Arch), SFT (supervised classfication fine-tune)," \
                           " IFT (instruction fine-tune), PFT (preference fine-tune)")
    )

    parser.add_argument(
        '--peft_type',
        type=str,
        default=None,
        help=("whether any parameter efficient techniques to be used for training. Options: None (full training), lora, qlora")
    )

    parser.add_argument(
        '--load_weights',
        type=bool,
        default=True,
        help=('whether to load weights from saved model or use random weights. Options: True (load pre-saved weights), False (use random weights)')
    )

    parser.add_argument(
        '--pre_save_model',
        type=str,
        default=None,
        help=('name of the pre-saved model that will be loaded for further tasks. '
                        'Options: None (if None, then load weights of base gpt2 model if load_weights=True' \
                                ' Or name of the previously trained model)')
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default='gpt2',
        help=('name of the trained model to be saved. ')
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
        '--train_split',
        type=float,
        default=0.70,
        help=('percentage of the data to be considered for training')
    )

    parser.add_argument(
        '--val_split',
        type=float,
        default=0.10,
        help=('percentage of the data to be considered for validation')
    )

    parser.add_argument(
        '--context_length',
        type=int,
        default=1024,
        help=('context length of the attention block to be considered.')
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
        '--dropout_rate',
        type=float,
        default=0.0,
        help=('Dropout rate value.')
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=None,
        help=('Top k tokens considered for next token prediction. Options: None (It will consider all the tokens in the final generation)')
    )

    parser.add_argument(
        "--trainable_layers",
        type=str,
        default=None,
        help=("Which layers to train. Used for SFT (primarily). Options: 'all', 'last_block', 'last_two_blocks', 'None'.")
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
        "--max_training_length",
        type=str,
        default="longest_training_example",
        help=("The context length of the data inputs. Options: longest_training_example, model_context_size or custom integer.")
    )

    parser.add_argument(
        "--prompt_style",
        type=str,
        default="alpaca",
        help=("The prompty style used for instruction fine-tune and preference fine-tune process. Options: alpaca, phi3")
    )

    parser.add_argument(
        "--ignore_index",
        type=int,
        default=-100,
        help=("The value used for masking the padding tokens in the input and target tensors so that those indices are not used during loss calc. "
                                            "Options: -100 (preferred), " \
                                            "custom integer (not recommended)")
    )

    parser.add_argument(
        "--mask_instruction",
        type=bool,
        default=False,
        help=("Whether to mask the instruction tokens or not in the target tensor. Used for IFT, PFT. Options: True , False")
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
        type=int,
        default=20,
        help=("Number of iterations (step) for LR to warmup.")
    )

    parser.add_argument(
        "--initial_lr",
        type=float,
        default=3e-05,
        help=("Intial LR value to start the warmup from.")
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help=("Minimum LR value to achieve using cosine annealing (decay.)")
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help=("Weightage parameter to be used in DPO loss for Preference fine-tune.")
    )

    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help=("The rank of LoRA matrices.")
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help=("LoRA hyper-parameter.")
    )


    args = parser.parse_args()
    torch.manual_seed(args.seed)

    try:
        config = configparser.ConfigParser()
        config.read("config.ini")

        DATA_FOLDER = config["PATHS"]["DATA_FOLDER"]
        LOG_FOLDER = config["PATHS"]["LOG_FOLDER"]
        MODEL_ROOT_FOLDER = config['PATHS']['MODEL_ROOT_FOLDER']

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

    #Load the model config:
    try:
        m_config = GPT2_ModelConfig()
        gpt2_config = m_config.load_model_config(model_name=args.base_modelName, drop_rate=args.dropout_rate,
                                                 context_length=args.context_length)
        gpt2_baseInst = GPT2(gpt2_config)
        gpt2_baseInst.eval()
        logger.info(f'Configuration of the {args.base_modelName} base model loaded..!')

    except Exception as e:
        logger.error(f"Error in loading the model config class {e}")
        raise Exception(f"Error in loading the model config class {e}")

    #Load the data and dataloader:
    try:

        #Read the data:
        data_path = os.path.join(DATA_FOLDER,args.data_path)
        extension = data_path.split('.')[-1]
        logger.info(f'Extention detected for the training file is "{extension}".')

        #Basic Data Preprocessing:
        if extension == 'txt':
            with open(data_path, 'r', encoding='utf-8') as file:
                data = file.read()
                print('Total characters present in the training file: ', len(data))
                logger.info(f'Total characters present in the training file: {len(data)}')

            #Create the train, val, test files:
            train_df, val_df, test_df = dataset_split(data=data, train_split=args.train_split, val_split=args.val_split, 
                                                        classify=False)
            logger.info(f'Training, Validation and Test Data created from the training file. Train data: {len(train_df)}, Val Data: {len(val_df)}, Test Data: {len(test_df)}')
            logger.info(f'Train data tokens: {len(tokenizer.encode(train_df))}, Val data tokens: {len(tokenizer.encode(val_df))}, Test data tokens: {len(tokenizer.encode(test_df))} and model context length : {gpt2_config["context_length"]}')

            # Sanity check
            if len(tokenizer.encode(train_df))  < gpt2_config["context_length"]:
                logger.info(f'Train data tokens: {len(tokenizer.encode(train_df))} and model context length : {gpt2_config["context_length"]}')
                logger.error("Not enough tokens for the training loader. "
                    "Try to lower the `GPT_CONFIG_124M['context_length']` or "
                    "increase the `train_split`")
                raise ValueError("Not enough tokens for the training loader. "
                    "Try to lower the `GPT_CONFIG_124M['context_length']` or "
                    "increase the `train_split`")
                
            if len(tokenizer.encode(val_df))  < gpt2_config["context_length"]:
                logger.info(f'Val data tokens: {len(tokenizer.encode(val_df))} and model context length : {gpt2_config["context_length"]}')
                logger.error("Not enough tokens for the validation loader. "
                    "Try to lower the `GPT_CONFIG_124M['context_length']` or "
                    "increase the `val_split`")
                raise ValueError("Not enough tokens for the validation loader. "
                    "Try to lower the `GPT_CONFIG_124M['context_length']` or "
                    "increase the `val_split`")
                
            if len(tokenizer.encode(test_df))  < gpt2_config["context_length"]:
                logger.info(f'Test data tokens: {len(tokenizer.encode(test_df))} and model context length : {gpt2_config["context_length"]}')
                logger.error("Not enough tokens for the test loader. "
                    "Try to lower the `GPT_CONFIG_124M['context_length']` or "
                    "decrease the `val_split`")
                raise ValueError("Not enough tokens for the test loader. "
                    "Try to lower the `GPT_CONFIG_124M['context_length']` or "
                    "decrease the `val_split`")

        
        elif extension == 'csv':
            balanced_df = csv_preproccessing(data_path, logger)

            #Create the train.csv, val.csv, test.csv
            train_df, val_df, test_df = dataset_split(data=balanced_df, train_split=args.train_split, val_split=args.val_split, 
                                                        classify=True)
            logger.info(f'Training, Validation and Test Data created from the training file. Train data: {train_df.shape}, Val Data: {val_df.shape}, Test Data: {test_df.shape}')

        
        elif extension == 'zip':
            
            print("Unzipping the file")
            logging.info('Unzipping the file')
            extracted_path =os.path.join(DATA_FOLDER,args.data_path.split('.')[0])

            with zipfile.ZipFile(data_path, "r") as zip_ref:
                zip_ref.extractall(extracted_path)

            fileName = [f for root,dir,file in os.walk(extracted_path) for f in file if f.lower() != 'readme']
            fileName = fileName[0].replace('.File','.tsv')

            # Add .tsv file extension
            data_file_path = Path(extracted_path) / (fileName+'.tsv')

            if not os.path.exists(data_file_path):
            
                original_file_path = Path(extracted_path) / fileName
                os.rename(original_file_path, data_file_path)
                print(f"File unzipped and saved as {data_file_path}")
            
            logger.info(f"File unzipped and saved at: {data_file_path}")

            balanced_df = csv_preproccessing(data_file_path, logger)

            #Create the train, val, test files:
            train_df, val_df, test_df = dataset_split(data=balanced_df, train_split=args.train_split, val_split=args.val_split, 
                                                        classify=True)
            logger.info(f'Training, Validation and Test Data created from the training file. Train data: {train_df.shape}, Val Data: {val_df.shape}, Test Data: {test_df.shape}')
        
        else:

            print('Reading for .json files..!')
            logger.info('Reading for .json files..!')
            with open(data_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            logger.info(f"Number of entries : {len(data)}. ")
            logger.info(f"Example of data for Instruct Fine-Tune :: \n {data[1000]}")
            pprint.pp(data[1000])

            #Create the train, val, test files:
            train_df, val_df, test_df = dataset_split(data=data, train_split=args.train_split, val_split=args.val_split, 
                                                        classify=False)
            logger.info(f'Training, Validation and Test Data created from the training file. Train data: {len(train_df)}, Val Data: {len(val_df)}, Test Data: {len(test_df)}')
            # logger.info(f'Train data tokens: {len(tokenizer.encode(train_df))}, Val data tokens: {len(tokenizer.encode(val_df))}, Test data tokens: {len(tokenizer.encode(test_df))} '
            #             f'and model context length : {gpt2_config["context_length"]}')
        
        #Prepare the dataloaders:
        if args.training_type == 'pre-train':
            logger.info('---------------------------------------------------------')
            logger.info("Loading the dataset class for pre-training...")
                
            train_dataLoader = GPTCustomDataloader(train_df, context_length=gpt2_config['context_length'], 
                                                   stride=gpt2_config['context_length'], batch_size=args.batch_size,
                                                   tokenizer = args.tokenizer,
                                                    shuffle=True, drop_last=True,num_workers=0)
            
            val_dataLoader = GPTCustomDataloader(val_df, context_length=gpt2_config['context_length'], 
                                                 stride=gpt2_config['context_length'], batch_size=args.batch_size,
                                                 tokenizer = args.tokenizer,
                                                 shuffle=True, drop_last=True,num_workers=0)
            

            test_dataLoader = GPTCustomDataloader(test_df, context_length=gpt2_config['context_length'], 
                                                  stride=gpt2_config['context_length'], batch_size=args.batch_size,
                                                   tokenizer = args.tokenizer,
                                                    shuffle=True, drop_last=True,num_workers=0)
            
            
                
            #Print the dataloader contents to confirm correct format:
            logger.info('************** TRAIN DATALOADER ****************************')
            logger.info(f'Length of Train Dataloader (number of batches): {len(train_dataLoader)}')
            for x,y in train_dataLoader:
                logger.info(f'{x.shape}, {y.shape}')
                break
                

            logger.info('************** VAL DATALOADER ****************************')
            logger.info(f'Length of Val Dataloader (number of batches): {len(val_dataLoader)}')
            for x,y in val_dataLoader:
                logger.info(f'{x.shape}, {y.shape}')
                break

            logger.info('************** TEST DATALOADER ****************************')
            logger.info(f'Length of Test Dataloader (number of batches): {len(test_dataLoader)}')
            for x,y in test_dataLoader:
                logger.info(f'{x.shape}, {y.shape}')
                break
                
                
            logger.info('Dataloaders created successfully for pre-training task..!')
            logger.info('---------------------------------------------------------')

        elif args.training_type == 'SFT':
            logger.info('---------------------------------------------------------')
            logger.info("Loading the dataset class for supervised classification fine-tuning task...")
            train_dataLoader = None

            if args.max_training_length == "longest_training_example":

                train_dataLoader = GPTCustomSFTDataloader(train_df, batch_size = args.batch_size, max_seq_length = None,
                                                    tokenizer = args.tokenizer,shuffle=True, drop_last=True,num_workers=0)
                
                #Print the dataloader contents to confirm correct format:
                logger.info('************** TRAIN DATALOADER ****************************')
                logger.info(f'Length of Train Dataloader (number of batches): {len(train_dataLoader)}')
                for x,y in train_dataLoader:
                    logger.info(f'{x.shape}, {y.shape}')
                    break
                train_max_length = x.shape[1]
                logger.info(f'Longest Training Example Length : {train_max_length}.')
            
            elif args.max_training_length == 'model_context_length':
                train_max_length = gpt2_config['context_length']

            else:
                train_max_length = int(args.max_training_length)
                

            assert train_max_length <= gpt2_config['context_length'], (
                                    f"Max training sequence ({args.max_training_length}) cannot be more"
                                    f" than base model context length of {gpt2_config['context_length']}" 
                                    )
            
            if train_dataLoader is None:
                train_dataLoader = GPTCustomSFTDataloader(train_df, batch_size = args.batch_size,max_seq_length = train_max_length,
                                                    tokenizer = args.tokenizer,shuffle=True, drop_last=True,num_workers=0)
                
                #Print the dataloader contents to confirm correct format:
                logger.info('************** TRAIN DATALOADER ****************************')
                logger.info(f'Length of Train Dataloader (number of batches): {len(train_dataLoader)}')
                for x,y in train_dataLoader:
                    logger.info(f'{x.shape}, {y.shape}')
                    break

            val_dataLoader = GPTCustomSFTDataloader(val_df, batch_size = args.batch_size, max_seq_length = train_max_length,
                                                   tokenizer = args.tokenizer,
                                                    shuffle=True, drop_last=True,num_workers=0)
            
            test_dataLoader = GPTCustomSFTDataloader(test_df, batch_size = args.batch_size,  max_seq_length = train_max_length,
                                                   tokenizer = args.tokenizer,
                                                    shuffle=True, drop_last=True,num_workers=0)
            
            logger.info('************** VAL DATALOADER ****************************')
            logger.info(f'Length of Val Dataloader (number of batches): {len(val_dataLoader)}')
            for x,y in val_dataLoader:
                logger.info(f'{x.shape}, {y.shape}')
                break
            
            logger.info('************** TEST DATALOADER ****************************')
            logger.info(f'Length of Test Dataloader (number of batches): {len(test_dataLoader)}')
            for x,y in test_dataLoader:
                logger.info(f'{x.shape}, {y.shape}')
                break
                
                
            logger.info('Dataloaders created successfully for classification fine-tuning task..!')
            logger.info('---------------------------------------------------------')


        elif args.training_type == 'IFT':
            logger.info("Loading the dataset class for instruction fine-tuning task...")

            if args.max_training_length == "longest_training_example":
                max_seq_length = None
            
            elif args.max_training_length == 'model_context_length':
                max_seq_length = gpt2_config['context_length']

            else:
                max_seq_length = int(args.max_training_length)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print('Device Available: ', device)

            train_dataLoader = GPTCustomInstructDataloader(train_df, device=device, max_seq_length=max_seq_length, batch_size=args.batch_size,
                                                           tokenizer = args.tokenizer,prompt_style=args.prompt_style,  ignore_index = args.ignore_index, 
                                                           shuffle=True, drop_last=True,num_workers=0, mask_instruction = args.mask_instruction
                                                           )
            

            val_dataLoader = GPTCustomInstructDataloader(val_df, device=device, max_seq_length=max_seq_length, batch_size=args.batch_size,
                                                           tokenizer = args.tokenizer,prompt_style=args.prompt_style, ignore_index = args.ignore_index, 
                                                           shuffle=True, drop_last=True,num_workers=0, mask_instruction = args.mask_instruction
                                                           )
            

            test_dataLoader = GPTCustomInstructDataloader(test_df, device=device, max_seq_length=max_seq_length, batch_size=args.batch_size,
                                                           tokenizer = args.tokenizer,prompt_style=args.prompt_style, ignore_index = args.ignore_index, 
                                                           shuffle=True, drop_last=True,num_workers=0, mask_instruction = args.mask_instruction
                                                           )
            

            #Print the dataloader contents to confirm correct format:
            logger.info('************** TRAIN DATALOADER ****************************')
            logger.info(f'Length of Train Dataloader (number of batches): {len(train_dataLoader)}')

            i=0
            for x,y in train_dataLoader:
                logger.info(f'{x.shape}, {y.shape}')
                if i > 3:
                    break
                else:
                    i = i + 1
                

            logger.info('************** VAL DATALOADER ****************************')
            logger.info(f'Length of Val Dataloader (number of batches): {len(val_dataLoader)}')
            i=0
            for x,y in val_dataLoader:
                logger.info(f'{x.shape}, {y.shape}')
                if i > 3:
                    break
                else:
                    i = i + 1

            logger.info('************** TEST DATALOADER ****************************')
            logger.info(f'Length of Test Dataloader (number of batches): {len(test_dataLoader)}')
            i=0
            for x,y in test_dataLoader:
                logger.info(f'{x.shape}, {y.shape}')
                if i > 3:
                    break
                else:
                    i = i + 1
                
            logger.info('Dataloaders created successfully for fine-tuning task..!')
            logger.info('---------------------------------------------------------')


        else:
            logger.info("Loading the class for preference fine-tuning task (PFT) ...")

            if args.max_training_length == "longest_training_example":
                max_seq_length = None
            
            elif args.max_training_length == 'model_context_length':
                max_seq_length = gpt2_config['context_length']

            else:
                max_seq_length = int(args.max_training_length)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print('Device Available: ', device)

            train_dataLoader = GPTCustomPreferenceDataloader(train_df, device=device, max_seq_length=max_seq_length, batch_size=args.batch_size,
                                                           tokenizer = args.tokenizer,prompt_style=args.prompt_style,  seed = args.seed,
                                                           shuffle=True, drop_last=True,num_workers=0, mask_instruction = args.mask_instruction
                                                           )
            

            val_dataLoader = GPTCustomPreferenceDataloader(val_df, device=device, max_seq_length=max_seq_length, batch_size=args.batch_size,
                                                           tokenizer = args.tokenizer,prompt_style=args.prompt_style, seed = args.seed,
                                                           shuffle=True, drop_last=True,num_workers=0, mask_instruction = args.mask_instruction
                                                           )
            

            test_dataLoader = GPTCustomPreferenceDataloader(test_df, device=device, max_seq_length=max_seq_length, batch_size=args.batch_size,
                                                           tokenizer = args.tokenizer,prompt_style=args.prompt_style, seed = args.seed,
                                                           shuffle=True, drop_last=True,num_workers=0, mask_instruction = args.mask_instruction
                                                           )
            
            #Print the dataloader contents to confirm correct format:
            logger.info('************** TRAIN DATALOADER ****************************')
            logger.info(f'Length of Train Dataloader (number of batches): {len(train_dataLoader)}')

            i=0
            for row in train_dataLoader:
                logger.info(f"{row['correct_response'].shape}, {row['wrong_response'].shape}")
                if i > 3:
                    break
                else:
                    i = i + 1

            logger.info(f'Info (keys) present in the loader for PFT : {row.keys()}')
                

            logger.info('************** VAL DATALOADER ****************************')
            logger.info(f'Length of Val Dataloader (number of batches): {len(val_dataLoader)}')
            i=0
            for row in val_dataLoader:
                logger.info(f"{row['correct_response'].shape}, {row['wrong_response'].shape}")
                if i > 3:
                    break
                else:
                    i = i + 1

            logger.info('************** TEST DATALOADER ****************************')
            logger.info(f'Length of Test Dataloader (number of batches): {len(test_dataLoader)}')
            i=0
            for row in test_dataLoader:
                logger.info(f"{row['correct_response'].shape}, {row['wrong_response'].shape}")
                if i > 3:
                    break
                else:
                    i = i + 1
                
            logger.info('Dataloaders created successfully for fine-tuning task..!')
            logger.info('---------------------------------------------------------')


    except Exception as e:
        logger.error(f'Error in loading file and creating dataloader:: {e}')
        raise Exception(f'Error in loading file and creating dataloader:: {e}')

    #Load the model weights:
    if args.load_weights:

        #Check if model root folder is present, else create it:
        if not os.path.exists(MODEL_ROOT_FOLDER):
            os.mkdir(MODEL_ROOT_FOLDER)

        if args.pre_save_model is None:

            try:
                logger.info(f'Loading the weights of the base model : {args.base_modelName}..!')

                # Using the function "download_and_load_gpt2" as is given in the book "Build LLM From Scratch":
                modelSize = args.base_modelName.split('_')[-1]
                modelDir= args.base_modelName.split('_')[0]

                model_path = os.path.join(MODEL_ROOT_FOLDER,modelDir)
                print(model_path)
                logger.info(f'Model present in the path: {model_path}')

                settings, params = download_and_load_gpt2(model_size=modelSize, models_dir=model_path)

                #Load the weights from OpenAI GPT2 to our instance:
                gpt2_loadedWeights(gpt2_baseInst, params)
                #gpt2_baseInst.to(device)

                logger.info('Model weights loaded successfully..!')

                #Generate a text to check if loading is successful:
                logger.info('Generate a text to check if loading is successful..!')
                generate = Text_Generation(model=gpt2_baseInst, device='cpu', tokenizer_model='gpt2')
                output_text = generate.text_generation(input_text = "Once upon a time,", max_new_tokens=args.max_new_tokens, 
                                                        temp=args.temp, top_k= args.top_k, eos_id = args.eos_id)
                logger.info(f'Generating a text :: \n{output_text}')
            
            except Exception as e:
                logger.error(f'Error in loading model weights : {e}')
                raise Exception(f'Error in loading model weights : {e}')

        else:
            try:
                logger.info(f'Loading the weights of the model : {args.pre_save_model}..!')
                model_path = os.path.join(MODEL_ROOT_FOLDER,args.pre_save_model)
                print(model_path)
                logger.info(f'Model present in the path: {model_path}')
                
                if args.training_type == 'SFT':

                    logger.info('Updating the model head of the base model to load saved weights successfully..!')
                    in_features = gpt2_config['embedding_dimension']
                    out_features = len(train_df['Label'].value_counts().index.tolist())

                    #Add the classification layer:
                    gpt2_baseInst.final_projection = torch.nn.Linear(in_features=in_features, out_features= out_features)

                #Model and Optimizer are saved in the path. Loading only the model for fine-tuning:
                checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
                gpt2_baseInst.load_state_dict(checkpoint['model'] )

                gpt2_baseInst.eval()

                logger.info('Model weights loaded successfully..!')

                if args.training_type == 'PFT' or args.training_type == 'IFT':

                    #Generate a text to check if loading is successful:
                    logger.info('Generate a text to check if loading is successful..!')
                    _, input_text = format_input_response(val_df[10], prompt_style = args.prompt_style, inference=True)

                    prompt = """Below is an instruction that describes a task. Write a response
                                that appropriately completes the request.

                                ### Instruction:
                                Convert the active sentence to passive: 'The chef cooks the meal every day.'
                            """
                    
                    generate = Text_Generation(model=gpt2_baseInst, device='cpu', tokenizer_model='gpt2')
                    output_text = generate.text_generation(input_text = input_text, max_new_tokens=args.max_new_tokens, 
                                                            temp=args.temp, top_k= args.top_k, eos_id = args.eos_id)
                    
                    #response = (output_text[len(prompt)-1:]).replace("### Response:", " ").replace('Response:', '').strip()
                    logger.info(f'Generating a text :: \n{output_text}')

                
                if args.training_type == 'PFT':
                    logger.info(f'Preference Fine-Tuning requires 2 models --> Policy (Trainable) and Reference (Frozen) ..! '
                                f'Weights for the policy model is loaded above..! '
                                f'Weights for reference model will be loaded NOW .. ! ')
                    
                    gpt2_reference = GPT2(gpt2_config)
                    gpt2_reference.load_state_dict(checkpoint['model'] )
                    gpt2_reference.eval()

                    logger.info('Model weights for REFERENCE model is loaded successfully..!')
            
            except Exception as e:
                logger.error(f'Error in loading model weights : {e}')
                raise Exception(f'Error in loading model weights : {e}')

    else:
        logger.info(f'Loading the model with random weights for training..!')

    if args.training_type == 'SFT':
        if args.peft_type is None:

            try:
                logger.info(f'Training the full model as no paramater efficient mechanisms are given..!')

                for params in gpt2_baseInst.parameters():
                    params.requires_grad = False

                logger.info(f'Training Stage : Frozen the original paramters of the model..!')

                in_features = gpt2_config['embedding_dimension']
                out_features = len(train_df['Label'].value_counts().index.tolist())

                #Add the classification layer:
                gpt2_baseInst.final_projection = torch.nn.Linear(in_features=in_features, out_features= out_features)
                
                logger.info(f'Training Stage : Added the NEW classification head..!')

                logger.info('************* Verifying the NEW output head of the model *************')
                input_text = "Once upon a time"
                input_encoded = tokenizer.encode(input_text,allowed_special='all')
                input_encoded = torch.tensor(input_encoded).unsqueeze(0)
                with torch.no_grad():
                    output_tensor = gpt2_baseInst(input_encoded) #Need to convert the encoded token list to torch tensor and add the batch dimension through unsqueeze
                
                logger.info(f'Output Dimension:: {output_tensor.shape} .')
                print('Output Dimension:: ', output_tensor.shape)
                assert output_tensor.shape[2] == out_features, (
                                f"Output Dimension is not matching with the number of classes in the data. Please verify...!" 
                                )

                logger.info('************* Verifying the NEW output head of the model : Successfull *************')


                if args.trainable_layers == "last_block" or args.trainable_layers == "last_two_blocks":
                    logger.info(f'Training Stage : Unfreezing the weights of last block of the model for fine-tuning..!')

                    torch.manual_seed(args.seed)

                    #Unfreeze the final layer normalization block parameters for fine-tuning:
                    for params in gpt2_baseInst.final_layerNorm.parameters():
                        params.requires_grad = True

                    #Unfreeze the last transformer block parameters for fine-tuning:
                    for params in gpt2_baseInst.transformer_block[-1].parameters():
                        params.requires_grad = True

                    
                    if args.trainable_layers == "last_two_blocks":
                        logger.info(f'Training Stage : Unfreezing the weights of second to last block of the model too for fine-tuning..!')
                        
                        #Unfreeze the seond to last transformer block parameters for fine-tuning:
                        for params in gpt2_baseInst.transformer_block[-2].parameters():
                            params.requires_grad = True

                elif args.trainable_layers == "all":
                    for params in gpt2_baseInst.parameters():
                        params.requires_grad = True
                    
                    logger.info(f'Training Stage : Unfreezing all layer weights for fine-tuning..!')
            
            except Exception as e:
                logger.error(f'Error in weight unfreezing stage : {e}')
                raise Exception(f'Error in weight unfreezing stage : {e}')

        elif args.peft_type == 'lora':
            logger.info(f'Paramater efficient mechanisms given is {args.peft_type}..!')

            params_orig = freeze_model(gpt2_baseInst)
            logger.info(f'Total trainable paramters in the original model: {params_orig}.')
            lora_parameterization(model=gpt2_baseInst, rank = args.lora_rank, alpha = args.lora_alpha)

            params_with_lora = sum(p.numel() for p in gpt2_baseInst.parameters() if p.requires_grad)

            logger.info(f"Total parameters in the model after LORA addition: {params_orig + params_with_lora}" )
            logger.info(f"Total trainable parameters with LORA (%): {round((params_with_lora / params_orig)*100, 2)} .")


            logger.info(f'Training Stage : LoRA Layers Added ..!')

        else:
            logger.info(f'Paramater efficient mechanisms given is {args.peft_type}..!')


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device Available: ', device)

        try:
            gpt2_baseInst.to(device)
            logger.info(f'Training Stage : Model sent to {device} for fine-tuning..!')
            logger.info(f'Trainable Parameters : {sum(p.numel() for p in gpt2_baseInst.parameters() if p.requires_grad)}..! ')
            
            start_time = time.time()
            torch.manual_seed(args.seed)

            logger.info(f'Training Stage : Fine-tuning of the model started ..!')

            optimizer = torch.optim.AdamW(gpt2_baseInst.parameters(), lr=5e-4, weight_decay=0.1)

            epochs = args.num_epochs

            #Save the SFT Model:
            save_model_name = args.model_name + '.pth'
            save_model_path = os.path.join(MODEL_ROOT_FOLDER,save_model_name)

            gpt2_trainer = GPT2_ClassificationFineTune(model=gpt2_baseInst, 
                                optimizer=optimizer,
                                train_dataLoader=train_dataLoader,
                                test_dataLoader=val_dataLoader,
                                num_epochs=epochs,
                                eval_batchSize=5, 
                                eval_freq=50,
                                device=device,
                                log_path=logger,     #Pass the logger object instead of logging path
                                warmup_steps=args.warmup_steps,
                                initial_lr=args.initial_lr,
                                min_lr=args.min_lr,
                                use_warmup=args.use_warmup,
                                use_gradient_clip=args.use_gradient_clip
                                ) 

            train_losses, test_losses, train_accuracy, val_accuracy, num_samples, track_lr = gpt2_trainer.train(save_model_path)

            end_time = time.time()
            execution_time_minutes = (end_time - start_time) / 60
            print(f"Training completed in {execution_time_minutes:.2f} minutes.")
            logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")

            logger.info(f"SFT Fine-Tuned model saved in {save_model_path}..!")
        
        except Exception as e:
            logger.error(f'Error in fine-tuning stage : {e}')
            raise Exception(f'Error in fine-tuning stage : {e}')

        try:
            logger.info(f'Saving the plots of the metrics tracked ..!')
            x = torch.linspace(0, epochs, len(train_losses))
            samples = torch.linspace(0, num_samples, len(train_losses))
            plt = Plots(samples, x, train_losses, test_losses, label = 'Loss')
            plt.plots('Loss', args.experiment_name)

            x = torch.linspace(0, epochs, len(train_accuracy))
            samples = torch.linspace(0, num_samples, len(train_accuracy))
            plt = Plots(samples, x,train_accuracy, val_accuracy, label = 'Accuracy')
            plt.plots('Accuracy', args.experiment_name)

            if args.use_warmup:
                plt.plot_lrs(track_lr, label='Learning Rate', type=args.experiment_name)

            metrics = Metrics(gpt2_baseInst, device)

            torch.manual_seed(args.seed)
            train_accuracy = metrics.accuracy_loader(train_dataLoader, num_batches=10)
            val_accuracy = metrics.accuracy_loader(val_dataLoader, num_batches=10)
            test_accuracy = metrics.accuracy_loader(test_dataLoader, num_batches=10)

            print(f'Supervised Fine-Tuned Model for {epochs} epochs:')
            print(f"Training accuracy: {train_accuracy*100:.2f}%")
            print(f"Validation accuracy: {val_accuracy*100:.2f}%")
            print(f"Test accuracy: {test_accuracy*100:.2f}%")

            logger.info(f"Training accuracy: {train_accuracy*100:.2f}%")
            logger.info(f"Validation accuracy: {val_accuracy*100:.2f}%")
            logger.info(f"Test accuracy: {test_accuracy*100:.2f}%")


            logger.info(f'Saving the model response for the test dataset ..!')
            response_save_path = os.path.join(DATA_FOLDER, args.model_name+'_testdata_response.csv')
            pred_label_list = []
            generate = Text_Generation(model=gpt2_baseInst, device=device, tokenizer_model='gpt2')
            for text in test_df['Text'].values:
                pred_label = generate.classify_text(text, max_length=train_max_length)
                pred_label_list.append(pred_label)

            test_df['Pred_Label'] = pred_label_list
            test_df.to_csv(response_save_path, index=None)
            logger.info(f'Model response for the test dataset saved in {response_save_path}..!')

            end_time = time.time()
            execution_time_minutes = (end_time - start_time) / 60
            print(f"Pipeline completed in {execution_time_minutes:.2f} minutes.")
            logger.info(f"Pipeline completed in {execution_time_minutes:.2f} minutes.")


        except Exception as e:
            logger.error(f'Error in model evaluation stage : {e}')
            raise Exception(f'Error in model evaluation stage : {e}')
        
    elif args.training_type == 'IFT':

        logger.info(f'Instruction Fine-tuning the base model: {args.base_modelName} ..!')

        if args.peft_type is None:

            logger.info(f'Training the full model as no paramater efficient mechanisms are given..!')

        elif args.peft_type == 'lora':

            logger.info(f'Paramater efficient mechanisms given is {args.peft_type}..!')

            params_orig = freeze_model(gpt2_baseInst)
            logger.info(f'Total trainable paramters in the original model: {params_orig}.')

            lora_parameterization(model=gpt2_baseInst, rank = args.lora_rank, alpha = args.lora_alpha)

            params_with_lora = sum(p.numel() for p in gpt2_baseInst.parameters() if p.requires_grad)

            logger.info(f"Total parameters in the model after LORA addition: {params_orig + params_with_lora}" )
            logger.info(f"Total trainable parameters with LORA (%): {round((params_with_lora / params_orig)*100, 2)} .")

            logger.info(f'Training Stage : LoRA Layers Added ..!')

        else:
            logger.info(f'Paramater efficient mechanisms given is {args.peft_type}..!') 

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device Available: ', device)

        try:
            gpt2_baseInst.to(device)
            logger.info(f'Training Stage : Model sent to {device} for fine-tuning..!')

            i, start_context = format_input_response(val_df[0], prompt_style = args.prompt_style, inference=True) 

            start_time = time.time()
            torch.manual_seed(args.seed)

            logger.info(f'Training Stage : Fine-tuning of the model started ..!')

            optimizer = torch.optim.AdamW(gpt2_baseInst.parameters(), lr=0.00005, weight_decay=0.1)

            epochs = args.num_epochs

            #IFT Model Save Path:
            save_model_name = args.model_name + '.pth'
            save_model_path = os.path.join(MODEL_ROOT_FOLDER,save_model_name)

            gpt2_trainer = GPT2_PreTrain(model=gpt2_baseInst, 
                                optimizer=optimizer,
                                train_dataLoader=train_dataLoader,
                                test_dataLoader=val_dataLoader,
                                num_epochs=epochs,
                                eval_batchSize=5, 
                                eval_freq=5,
                                device=device,
                                start_context=start_context,
                                max_new_tokens=args.max_new_tokens,
                                log_path=logger,                        #Pass the logger object instead of logging path
                                warmup_steps=args.warmup_steps,
                                initial_lr=args.initial_lr,
                                min_lr=args.min_lr,
                                use_warmup=args.use_warmup,
                                use_gradient_clip=args.use_gradient_clip
                                ) 

            train_losses, test_losses, track_tokens_seen, track_lr = gpt2_trainer.train(model_save_path=save_model_path, temp=args.temp, top_k=args.top_k,  
                                                                              eos_id = args.eos_id)

            end_time = time.time()
            execution_time_minutes = (end_time - start_time) / 60
            print(f"Training completed in {execution_time_minutes:.2f} minutes.")
            logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")
       
            logger.info(f"BEST Instruction Fine-Tuned (IFT) model saved in {save_model_path}..!")
        
        except Exception as e:
            logger.error(f'Error in fine-tuning stage : {e}')
            raise Exception(f'Error in fine-tuning stage : {e}')
        

        try:
            logger.info(f'Saving the plots of the metrics tracked ..!')
            epochs_tensor = torch.linspace(0, epochs, len(train_losses))
            plt = Plots(track_tokens_seen, epochs_tensor, train_losses, test_losses)
            plt.plots('Loss', args.experiment_name)

            if args.use_warmup:
                plt.plot_lrs(track_lr, label='Learning Rate', type=args.experiment_name)

            logger.info(f'Saving the model response for the test dataset ..!')
            generate = Text_Generation(model=gpt2_baseInst, device=device, tokenizer_model='gpt2')
            test_data_response = save_model_response(data = test_df, generate = generate, 
                                                     temp = args.temp, top_k = args.top_k, eos_id = args.eos_id, 
                                                     max_new_tokens = args.max_new_tokens) 

            response_save_path = os.path.join(DATA_FOLDER, args.model_name+'_testdata_response.json')
            with open(response_save_path, "w") as file:
                json.dump(test_data_response, file, indent=4)
            
            logger.info(f'Model response for the test dataset saved in {response_save_path}..!')

            end_time = time.time()
            execution_time_minutes = (end_time - start_time) / 60
            print(f"Pipeline completed in {execution_time_minutes:.2f} minutes.")
            logger.info(f"Pipeline completed in {execution_time_minutes:.2f} minutes.")

        except Exception as e:
            logger.error(f'Error in model evaluation stage : {e}')
            raise Exception(f'Error in model evaluation stage : {e}')
    
    else:

        logger.info(f'Preference Fine-tuning the IFT model: {args.pre_save_model} ..!')

        if args.peft_type is None:

            logger.info(f'Training the full model as no paramater efficient mechanisms are given..!')

        elif args.peft_type == 'lora':
            
            logger.info(f'Paramater efficient mechanisms given is {args.peft_type}..!')

            params_orig = freeze_model(gpt2_baseInst)
            logger.info(f'Total trainable paramters in the original model: {params_orig}.')
            
            lora_parameterization(model=gpt2_baseInst, rank = args.lora_rank, alpha = args.lora_alpha)

            params_with_lora = sum(p.numel() for p in gpt2_baseInst.parameters() if p.requires_grad)

            print(f"Total parameters in the model after LORA addition: {params_orig + params_with_lora}" )
            print(f"Total trainable parameters with LORA (%): {round((params_with_lora / params_orig)*100, 2)} .")

            logger.info(f'Training Stage : LoRA Layers Added ..!')

        else:
            logger.info(f'Paramater efficient mechanisms given is {args.peft_type}..!') 

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device Available: ', device)

        try:
            gpt2_baseInst.to(device)
            gpt2_reference.to(device)

            logger.info(f'Training Stage : Policy and Reference Models sent to {device} for fine-tuning..!')

            i, start_context = format_input_response(val_df[10], prompt_style = args.prompt_style, inference=True) 

            start_time = time.time()
            torch.manual_seed(args.seed)

            logger.info(f'Training Stage : Fine-tuning of the model started ..!')

            optimizer = torch.optim.AdamW(gpt2_baseInst.parameters(), lr=5e-6, weight_decay=0.01)
            epochs = args.num_epochs

            #PFT Model Save Path:
            save_model_name = args.model_name + '.pth'
            save_model_path = os.path.join(MODEL_ROOT_FOLDER,save_model_name)

            gpt2_trainer = GPT2_PreferenceFineTune(policy_model=gpt2_baseInst, 
                                                   reference_model=gpt2_reference,
                                                    optimizer=optimizer,
                                                    train_dataLoader=train_dataLoader,
                                                    test_dataLoader=val_dataLoader,
                                                    num_epochs=epochs,
                                                    eval_batchSize=5, 
                                                    eval_freq=5,
                                                    device=device,
                                                    start_context=start_context,
                                                    max_new_tokens=args.max_new_tokens,
                                                    beta = args.beta,
                                                    log_path=logger,                        #Pass the logger object instead of logging path
                                                    warmup_steps=args.warmup_steps,
                                                    initial_lr=args.initial_lr,
                                                    min_lr=args.min_lr,
                                                    use_warmup=args.use_warmup,
                                                    use_gradient_clip=args.use_gradient_clip
                                                    ) 

            tracking_preferenceFT, track_lr = gpt2_trainer.train(model_save_path=save_model_path, temp = args.temp, 
                                                                 top_k = args.top_k,  
                                                                 eos_id = args.eos_id)

            end_time = time.time()
            execution_time_minutes =(end_time - start_time) / 60
            print(f"Training completed in {execution_time_minutes:.2f} minutes.")
            logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")
       
            logger.info(f"BEST Preference Fine-Tuned (PFT) model saved in {save_model_path}..!")

        
        except Exception as e:
            logger.error(f'Error in fine-tuning stage : {e}')
            raise Exception(f'Error in fine-tuning stage : {e}')
        

        try:
            logger.info(f'Saving the plots of the metrics tracked ..!')

            epochs_tensor = torch.linspace(0, epochs, len(tracking_preferenceFT['train_loss']))
            plt = Plots(tracking_preferenceFT['tokens_seen'], epochs_tensor, tracking_preferenceFT['train_loss'], tracking_preferenceFT['val_loss'])
            plt.plots('Loss', args.experiment_name)

            train_reward_margin = [x - y for x,y in zip(tracking_preferenceFT['train_rewards_correct'], tracking_preferenceFT['train_rewards_wrong'])]
            val_reward_margin = [x - y for x,y in zip(tracking_preferenceFT['val_rewards_correct'], tracking_preferenceFT['val_rewards_wrong'])]

            plt = Plots(tracking_preferenceFT['tokens_seen'], epochs_tensor, train_reward_margin, val_reward_margin)
            plt.plots('Reward', args.experiment_name)

            if args.use_warmup:
                plt.plot_lrs(track_lr, label='Learning Rate', type=args.experiment_name)

            
            logger.info(f'Analysing 10 test samples after preference tuning..!')

            generate_policy = Text_Generation(model=gpt2_baseInst, device=device, tokenizer_model='gpt2')
            generate_reference = Text_Generation(model=gpt2_reference, device=device, tokenizer_model='gpt2')

            analyse_preferenceTuning(data = test_df, generate_policy = generate_policy, generate_reference = generate_reference, logger = logger, 
                                     n_records = 10,
                                     temp = args.temp, top_k = args.top_k, eos_id = args.eos_id, 
                                     max_new_tokens = args.max_new_tokens)
            
            
            logger.info(f'Saving the model response for the test dataset ..!')
            
            test_data_response = save_model_response(data = test_df, generate = generate_policy, 
                                                     temp = args.temp, top_k = args.top_k, eos_id = args.eos_id, 
                                                     max_new_tokens = args.max_new_tokens) 

            response_save_path = os.path.join(DATA_FOLDER, args.model_name+'_testdata_response.json')
            with open(response_save_path, "w") as file:
                json.dump(test_data_response, file, indent=4)
            
            logger.info(f'Model response for the test dataset saved in {response_save_path}..!')

            end_time = time.time()
            execution_time_minutes =(end_time - start_time) / 60
            print(f"Pipeline completed in {execution_time_minutes:.2f} minutes.")
            logger.info(f"Pipeline completed in {execution_time_minutes:.2f} minutes.")

        except Exception as e:
            logger.error(f'Error in model evaluation stage : {e}')
            raise Exception(f'Error in model evaluation stage : {e}')


        



                




                
        





        
    
    

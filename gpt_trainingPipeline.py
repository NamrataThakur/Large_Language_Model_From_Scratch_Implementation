import argparse
import logging
import os
import time
from datetime import datetime
import warnings
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
#from imblearn.over_sampling import SMOTE, ADASYN
import math
from tensorboardX import SummaryWriter
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
# from gpt_ClassificationFT.gpt2_classificationFT import GPT2_ClassificationFineTune

#Load the class for model config
from gpt_ClassificationFT.gpt2_model_config import GPT2_ModelConfig
#from gpt_PreferenceFT.gpt2_classificationFT import GPT2_ClassificationFineTune

#Utilities for model weight loading
from model_utils.gpt_download import download_and_load_gpt2
from model_utils.load_model_weights import *

#Dataloaders for Pre-Training, SFT, Instruct FT, Preference FT
from dataloader.Pretraining.gpt2_datasetSplit import dataset_split
from dataloader.Pretraining.gpt2_pretrainDataloader import GPTCustomDataloader
from dataloader.Classification_finetuning.gpt2_classificationDataloader import GPTCustomSFTDataloader
from dataloader.Instruction_finetuning.gpt2_instructDataloader import GPTCustomInstructDataloader
from dataloader.Instruction_finetuning.gpt2_instructDataFormat import *
#from dataloader.Preference_finetuning import *



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
        default='the-verdict.txt',
        help=('The name of the training data file. This file is present under the folder "data". Extension accepted: .txt, .csv, .tsv, .zip ')
    )

    parser.add_argument(
        '--training_type',
        type=str,
        default='pre-train',
        help=("the pipeline to be run for what Type of training. "
                "Options: pre-train, SFT (supervised classfication fine-tune)," \
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
        '--pre-save-model',
        type=str,
        default=None,
        help=('name of the pre-saved model that will be loaded for further tasks. '
                        'Options: None (if None, then either load weights of base gpt2 model if load_weights=True' \
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
        default=1,
        help=('batch_size per gpu')
    )

    parser.add_argument(
        '--train_split',
        type=float,
        default=0.50,
        help=('percentage of the data to be considered for training')
    )

    parser.add_argument(
        '--val_split',
        type=float,
        default=0.25,
        help=('percentage of the data to be considered for validation')
    )

    args = parser.parse_args()

    try:
        config = configparser.ConfigParser()
        config.read("config.ini")

        DATA_FOLDER = config["PATHS"]["DATA_FOLDER"]
        LOG_FOLDER = config["PATHS"]["LOG_FOLDER"]

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

    tokenizer = args.tokenizer
    if tokenizer == 'tiktoken':
        tokenizer = tiktoken.get_encoding("gpt2")

    #Load the model config:
    try:
        m_config = GPT2_ModelConfig()
        gpt2_config = m_config.load_model_config(model_name=args.base_modelName, context_length=1024)
        logger.info(f'Configuration of the {args.base_modelName} base model loaded..!')

    except Exception as e:
        logger.error(f"Error in loading the model config class {e}")

    #Load the data and dataloader:
    try:

        data_path = os.path.join(DATA_FOLDER,args.data_path)
        extension = data_path.split('.')[-1]
        logger.info(f'Extention detected for the training file is "{extension}".')

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
                    
                if len(tokenizer.encode(val_df))  < gpt2_config["context_length"]:
                    logger.info(f'Val data tokens: {len(tokenizer.encode(val_df))} and model context length : {gpt2_config["context_length"]}')
                    logger.error("Not enough tokens for the validation loader. "
                        "Try to lower the `GPT_CONFIG_124M['context_length']` or "
                        "increase the `val_split`")
                    
                if len(tokenizer.encode(test_df))  < gpt2_config["context_length"]:
                    logger.info(f'Test data tokens: {len(tokenizer.encode(test_df))} and model context length : {gpt2_config["context_length"]}')
                    logger.error("Not enough tokens for the validation loader. "
                        "Try to lower the `GPT_CONFIG_124M['context_length']` or "
                        "decrease the `val_split`")

        elif extension == 'csv':
            data = pd.read_csv(data_path)
            print('Total records present in the training file: ', data.shape)
            logger.info(f'Total records present in the training file: {data.shape}')

            #Create the train.csv, val.csv, test.csv

        else:
            #TO DO for ZIP Format
            print("TO DO FOR ZIP FORMAT")


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
            logger.info("Loading the dataset class for supervised classification fine-tuning task...")
        elif args.training_type == 'IFT':
            logger.info("Loading the dataset class for instruction fine-tuning task...")
        else:
            logger.info("Loading the class for preference fine-tuning task...")
    
    except Exception as e:
        logger.error(f'Error in loading file and creating dataloader:: {e}')

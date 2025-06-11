import argparse
import logging
import os
import time
import datetime
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
        default='/the-verdict.txt',
        help=('The path to the training data file. This file is present under the folder "data" ')
    )

    parser.add_argument(
        '--training_type',
        type=str,
        default='SFT',
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
        help=('path to the pre-saved model that will be loaded for further tasks. '
                        'Options: None (if None, then either load weights of base gpt2 model if load_weights=True' \
                                ' Or the path to the previously trained model)')
    )

    parser.add_argument(
        '--model_root_path',
        type=str,
        default='/model/',
        help=('base path where trained models to be saved.')
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

    args = parser.parse_args()

    #Add the logging functionality:
    logs_path = r'logs'
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)

    logging_path = os.path.join(logs_path,args.experiment_name+'_'+str(datetime.datetime.now())+'.log')
    print('D:\LLM_Deeplearning.ai\SEBASTIAN_RASCHKA\Large_Language_Model_From_Scratch_Implementation\logs\\'+args.experiment_name+'.log')
    
    logging.basicConfig(
        filename=r'D:\LLM_Deeplearning.ai\SEBASTIAN_RASCHKA\Large_Language_Model_From_Scratch_Implementation\logs\\'+args.experiment_name+'.log',
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        filemode='w'
    )

    logger = logging.getLogger()
    logger.info(str(args))

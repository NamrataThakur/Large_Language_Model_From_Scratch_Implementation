import pandas as pd

def csv_preproccessing(data_file_path, logger, dataset_name = "spam_ham"):

    #ADD A ELSE LOGIC FOR ANY OTHER CUSTOM DATASET:
    #LOGIC ADDED FOR SPAM-HAM AND SINGAPORE REVIEWS DATASET:
    if dataset_name == 'spam_ham':
        logger.info(f"Preprocessing for dataset : {dataset_name}")
        data = pd.read_csv(data_file_path, sep='\t', header = None, names=['Label', 'Text'])
        logger.info(f'Total records present in the training file: {data.shape}')

        #Balancing Strategies:
        #Detect the minority class and the corresponding record count:
        class_count = data['Label'].value_counts().array.tolist()
        class_name = data['Label'].value_counts().index
        min_count = min(class_count)
        minority_name = class_name[class_count.index(min_count)]

        #Prepare the minority and majority class dataset:
        min_df = data[data['Label'] == minority_name]
        max_df = data[data['Label'] != minority_name].sample(min_count, random_state=123)
        balanced_df = pd.concat([max_df,min_df], ignore_index=True)
        balanced_df['Label'] = balanced_df['Label'].map({'spam':1, 'ham': 0}) #Create a custom part for this --> TO DO

        logger.info(f'After balancing : {balanced_df.shape}')

    else:
        logger.info(f"Preprocessing for dataset : {dataset_name}")
        logger.info(f"")

        #Make the Label class renamed as "Target"

    return balanced_df



def get_focal_weights(train_dataset, label, logger):

    weight_dict = {}
    n_samples = train_dataset.shape[0]
    n_classes = len(train_dataset.label.unique())

    for i in list(range(n_classes)):
        weights_dict[i] = round((n_samples / (n_classes * train_dataset[label].value_counts()[i])),3)

    weights_dict = {k : v / total for total in (sum(weights_dict.values(), 0.0),) for k, v in weights_dict.items()}

    logger.info(f"Normalized Focal Weights : {weights_dict}")

    return weight_dict



def getClassNames(df, minority = False):

    classNames = ['global_lora']

    '''
    4    47292
    3    22313
    2     6519
    1     1358
    0      800
    '''

    
    df['Target'].value_counts().index

    return classNames
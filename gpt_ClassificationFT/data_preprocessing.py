import pandas as pd

def csv_preproccessing(data_file_path, logger):
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

    return balanced_df
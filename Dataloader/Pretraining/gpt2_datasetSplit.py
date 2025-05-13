import torch

def dataset_split(data, train_split, val_split, classify = False):

    torch.manual_seed(123)
    
    if classify:
        #Shuffle the dataset for classification Fine Tune:
        data = data.sample(frac=1, random_state=123).reset_index(drop=True)
        row_count = data.shape[0]

    else:
        row_count = len(data)

    #Create the split indices:
    train_df = data[ : int(train_split * row_count)]
    val_df = data[int(train_split * row_count) : int(train_split * row_count) + int(val_split * row_count)]
    test_df = data[int(train_split * row_count) + int(val_split * row_count) : ]

    print('Train and Val Split Index :: ',int(train_split * row_count), int(train_split * row_count) + int(val_split * row_count))
    return train_df, val_df, test_df
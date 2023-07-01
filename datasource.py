import torch

def get_data():
    from glob import glob
    import pandas as pd
    paths = glob('/home/seokj/workspace/FSR-prediction/data/*/*/*')
    filedata = pd.DataFrame([path.split('/')[-3:] for path in paths], columns=['subject', 'pose', 'filename'])
    filedata['path'] = paths
    filedata = filedata.sort_values(['subject', 'pose']).reset_index(drop=True)
    filedata
    data = []
    for index, value in filedata.iterrows():
        df = pd.read_pickle(value['path'])
        df = df.rename_axis('time').reset_index()
        df['id'] = index
        cols = df.columns.to_list()
        df = df[cols[-1:] + cols[:-1]]
        data.append(df)
    data = pd.concat(data)
    data = data.reset_index(drop=True)
    return data


class FSRDataset(torch.utils.data.Dataset):
    def __init__(self, X_df, y_df, index):
        assert(len(X_df) == len(y_df))
        self.X_df = X_df
        self.y_df = y_df
        self.index = index

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        import numpy as np
        X = self.X_df.loc[self.index[idx]].to_numpy().astype(np.float32)
        y = self.y_df.loc[self.index[idx]].to_numpy().astype(np.float32)
        return X, y
    
    
def get_index_splited_by_time(data, test_size=None):
    from sklearn.model_selection import train_test_split
    train_indexes = []
    test_indexes = []
    for _, group in data.groupby('id'):
        train_index, test_index = train_test_split(group.index, test_size=test_size, shuffle=False)
        train_indexes.append(train_index)
        test_indexes.append(test_index)
    return train_indexes, test_indexes
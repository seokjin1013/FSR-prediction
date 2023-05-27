import torch
import pathlib
import numpy as np
import pandas as pd
import functools
from sklearn.model_selection import train_test_split
from typing import Sequence

__init__ = [
    'get_force_transfer_dataset',
    'get_coord_transfer_dataset',
    'get_force_time_cut_dataset',
    'get_coord_time_cut_dataset',
    'get_all_transfer_dataset',
    'get_all_time_cut_dataset',
]


DATA_PATH = pathlib.Path('./data')


class FSRDataset(torch.utils.data.Dataset):
    def __init__(self, X_df: Sequence[pd.DataFrame], y_df: Sequence[pd.DataFrame]):
        assert(len(X_df) == len(y_df))
        self.X_df = list(X_df)
        self.y_df = list(y_df)

    def __len__(self):
        return len(self.X_df)
    
    def __getitem__(self, idx):
        X = self.X_df[idx].to_numpy().astype(np.float32)
        y = self.y_df[idx].to_numpy().astype(np.float32)
        return X, y


def _get_transfer_df():
    people = list(DATA_PATH.glob('*'))

    type_id = [0 for _ in range(len(people))]
    type_id[0] = 1
    type_id[1] = 2

    train_people = []
    valid_people = []
    test_people = []
    for i, p in enumerate(people):
        if type_id[i] == 0:
            train_people.append(p)
        elif type_id[i] == 1:
            valid_people.append(p)
        elif type_id[i] == 2:
            test_people.append(p)
    
    train_paths = []
    valid_paths = []
    test_paths = []
    for p in train_people:
        train_paths.extend(p.glob('*/*.pickle'))
    for p in valid_people:
        valid_paths.extend(p.glob('*/*.pickle'))
    for p in test_people:
        test_paths.extend(p.glob('*/*.pickle'))
    
    train_df_list = [pd.read_pickle(p) for p in train_paths]
    valid_df_list = [pd.read_pickle(p) for p in valid_paths]
    test_df_list = [pd.read_pickle(p) for p in test_paths]
    return train_df_list, valid_df_list, test_df_list
    

def _get_time_cut_df():
    paths = DATA_PATH.glob('*/*/*.pickle')
    df_list = [pd.read_pickle(p) for p in paths]

    train_df_list = []
    valid_df_list = []
    test_df_list = []
    for df in df_list:
        train_valid_df, test_df = train_test_split(df, test_size=1/7)
        train_df, valid_df = train_test_split(train_valid_df, test_size=1/6)
        train_df_list.append(train_df)
        valid_df_list.append(valid_df)
        test_df_list.append(test_df)
    
    return train_df_list, valid_df_list, test_df_list


def _get_force_dataset(scaler, df):
    if scaler:
        df = [pd.DataFrame(scaler.fit_transform(d), index=d.index, columns=d.columns) for d in df]
    return FSRDataset([d[['FSR_for_force']] for d in df], [d[['force']] for d in df])


def _get_coord_dataset(imputer, scaler, df):
    assert imputer.get_params()['keep_empty_features'] == True, '"keep empty features" of imputer should be True.'
    df = [pd.DataFrame(imputer.fit_transform(d), index=d.index, columns=d.columns) for d in df]
    if scaler:
        df = [pd.DataFrame(scaler.fit_transform(d), index=d.index, columns=d.columns) for d in df]
    return FSRDataset([d[['FSR_for_coord']] for d in df], [d[['x_coord', 'y_coord']] for d in df])


def _get_all_dataset(imputer, scaler, df):
    assert imputer.get_params()['keep_empty_features'] == True, '"keep empty features" of imputer should be True.'
    df = [pd.DataFrame(imputer.fit_transform(d), index=d.index, columns=d.columns) for d in df]
    if scaler:
        df = [pd.DataFrame(scaler.fit_transform(d), index=d.index, columns=d.columns) for d in df]
    return FSRDataset([d[['FSR_for_force', 'FSR_for_coord']] for d in df], [d[['force', 'x_coord', 'y_coord']] for d in df])


def _get_dataset(train_test_split_policy, feature_selection_policy):
    train_df, valid_df, test_df = train_test_split_policy()
    train_dataset = feature_selection_policy(train_df)
    valid_dataset = feature_selection_policy(valid_df)
    test_dataset = feature_selection_policy(test_df)
    return train_dataset, valid_dataset, test_dataset


def get_force_transfer_dataset(scaler=None):
    return _get_dataset(_get_transfer_df, functools.partial(_get_force_dataset, scaler))


def get_coord_transfer_dataset(imputer, scaler=None):
    return _get_dataset(_get_transfer_df, functools.partial(_get_coord_dataset, imputer, scaler))


def get_all_transfer_dataset(imputer, scaler=None):
    return _get_dataset(_get_transfer_df, functools.partial(_get_all_dataset, imputer, scaler))


def get_force_time_cut_dataset(scaler=None):
    return _get_dataset(_get_time_cut_df, functools.partial(_get_force_dataset, scaler))


def get_coord_time_cut_dataset(imputer, scaler=None):
    return _get_dataset(_get_time_cut_df, functools.partial(_get_coord_dataset, imputer, scaler))


def get_all_time_cut_dataset(imputer, scaler=None):
    return _get_dataset(_get_time_cut_df, functools.partial(_get_all_dataset, imputer, scaler))


from data import *
from model import *
from train import *
from utils import *
import torch.utils.data
import optuna
import torch
import functools
import pickle
import os
import argparse
import time

def objective():
    train_dataset, valid_dataset, _ = get_force_transfer_dataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=None)

    @functools.wraps(wrapped=objective)
    def inner(trial:optuna.trial.Trial):
        seed_everything()
        input_size = next(iter(train_dataset))[0].shape[-1]
        output_size = next(iter(train_dataset))[1].shape[-1]

        num_epoch = 100

        model_name = trial.suggest_categorical('model', ['CNN-LSTM'])
        if model_name == 'LSTM':
            hidden_size = trial.suggest_categorical('lstm_hidden_size', [8, 16, 32, 64, 128])
            num_layer = trial.suggest_int('lstm_num_layer', 1, 4)
            model = LSTM(input_size, hidden_size, num_layer, output_size)
        elif model_name == 'CNN-LSTM':
            lstm_hidden_size = trial.suggest_categorical('lstm_hidden_size', [8, 16, 32, 64, 128])
            lstm_num_layer = trial.suggest_int('lstm_num_layer', 1, 4)
            cnn_hidden_size = trial.suggest_categorical('cnn_hidden_size', [8, 16, 32, 64, 128])
            cnn_num_layer = trial.suggest_int('cnn_num_layer', 1, 4)
            model = CNN_LSTM(input_size, cnn_hidden_size, lstm_hidden_size, cnn_num_layer, lstm_num_layer, output_size)
        
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RAdam', 'AdamW', 'NAdam'])#, 'Adagrad', 'SGD'])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1, log=True)
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'Adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'RAdam':
            optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'NAdam':
            optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
            
        scheduler_name = trial.suggest_categorical('scheduler',[ 'CosineAnnealingLR'])
        if scheduler_name == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch)
        
        criterion = torch.nn.MSELoss()
        
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(logdir=os.path.join('./runs', trial.study.study_name, time.ctime().replace(':', '-')))
        
        return train(trial.study.study_name, train_loader, valid_loader, num_epoch, model, criterion, optimizer, scheduler, writer, trial, './parameters')

    return inner
objective = objective()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', dest='name', type=str, help='name of study to identify', default=time.ctime().replace(':', '-'))
    parser.add_argument('--num', dest='num', type=int, help='num of trials', default=100)
    parser.add_argument('--load', dest='load', type=str, help='path of optuna study object.pkl to resume')
    args = parser.parse_args()
    if not args.load:
        study = optuna.create_study(study_name=args.name)
    else:
        study = pickle.load(open(args.load, 'rb'))
        study.study_name = args.name
    try:
        study.optimize(objective, n_trials=args.num)
    finally:
        if (study.trials_dataframe()['state'] == 'COMPLETE').any():
            print(study.trials_dataframe())
            print(study.best_params)
            print(study.best_value)
            pickle.dump(study, open(os.path.join('./optuna_result', args.name), 'wb'))
    
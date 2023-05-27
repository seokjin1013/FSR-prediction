from utils import *
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import optuna
import os

__init__ = [
    'train'
]

def train(name, train_loader, test_loader, num_epoch, model, criterion, optimizer, scheduler, writer=None, trial=None, best_parameter_path=None):
    seed_everything()
    best_rmse = float('inf')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    criterion = criterion.to(device)
    
    for epoch in range(num_epoch):
        model.train()
        criterion.train()
        for X, y in train_loader:
            pred = model(X.to(device))
            loss = criterion(pred, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        model.eval()
        criterion.eval()
        with torch.no_grad():
            mae = []
            mse = []
            num = []
            for X, y in test_loader:
                pred = model(X.to(device))
                try:
                    mae.append(mean_absolute_error(y, pred.cpu().detach()))
                    mse.append(mean_squared_error(y, pred.cpu().detach()))
                    num.append(len(y))
                except ValueError:
                    return best_rmse
            mae = np.array(mae)
            mse = np.array(mse)
            num = np.array(num)
            mae = (mae * num).sum() / sum(num)
            mse = (mse * num).sum() / sum(num)
            
            rmse = mse ** 0.5
            if rmse < best_rmse:
                best_rmse = rmse

                # model save
                if best_parameter_path and (not trial or [t for t in trial.study.trials if t.state == optuna.trial.TrialState.COMPLETE] and rmse < trial.study.best_value):
                    path = os.path.join(best_parameter_path, name)
                    torch.save({
                        'model':model,
                        'criterion':criterion,
                        'optimizer':optimizer,
                        'scheduler':scheduler,
                    }, open(path, 'wb'))

            # Tensorboard writer
            if writer:
                writer.add_scalar('metric/MAE', mae, epoch)
                writer.add_scalar('metric/RMSE', rmse, epoch)
                writer.flush()

            # Optuna trial
            if trial:
                trial.report(rmse, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
    return best_rmse
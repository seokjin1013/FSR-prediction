from typing import Dict
import ray.tune
import fsr_data
import torch
import sklearn.metrics
import numpy as np
import os
import ray.train.torch

class FSR_Trainable(ray.tune.Trainable):
    @staticmethod
    def _import_class(name:str):
        import importlib
        index = name.rfind('.')
        module_name = name[:index] if index != -1 else '__main__'
        class_name = name[index + 1:]
        return getattr(importlib.import_module(module_name), class_name)


    def setup(self, config):
        index_X = config['index_X']
        index_y = config['index_y']
        model = config['model']
        model_args = config.get('model_args', {})
        criterion = config['criterion']
        optimizer = config['optimizer']
        optimizer_args = config['optimizer_args']
        imputer = config.get('imputer')
        scaler = config.get('scaler')
        data_loader = config.get('data_loader')

        data, train_indexes, test_indexes = self._import_class(data_loader)()
        self.data = data.copy()
        concated_train_indexes = np.concatenate(train_indexes)
        if imputer:
            self.imputer = self._import_class(imputer)(**config['imputer_args'])
            index_Xy = (index_X if isinstance(index_X, list) else [index_X]) + (index_y if isinstance(index_y, list) else [index_y])
            self.imputer.fit(self.data.loc[concated_train_indexes, index_Xy])
            self.data.loc[:, index_Xy] = self.imputer.transform(self.data.loc[:, index_Xy])
        if scaler:
            self.scaler_X = self._import_class(scaler)()
            self.scaler_y = self._import_class(scaler)()
            self.scaler_X.fit(self.data.loc[concated_train_indexes, index_X])
            self.scaler_y.fit(self.data.loc[concated_train_indexes, index_y])
            self.data.loc[:, index_X] = self.scaler_X.transform(self.data.loc[:, index_X])
            self.data.loc[:, index_y] = self.scaler_y.transform(self.data.loc[:, index_y])
        train_dataset = fsr_data.FSRDataset(self.data.loc[:, index_X], self.data.loc[:, index_y], train_indexes)
        test_dataset = fsr_data.FSRDataset(self.data.loc[:, index_X], self.data.loc[:, index_y], test_indexes)

        self.model = self._import_class(model)(input_size=len(self.data.loc[:, index_X].columns), output_size=len(self.data.loc[:, index_y].columns), **model_args)
        self.criterion = self._import_class(criterion)()
        self.optimizer = self._import_class(optimizer)(self.model.parameters(), **optimizer_args)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None)
    

    def step(self):
        self.model.train()
        for X, y in self.train_loader:
            pred = self.model(X)
            loss = self.criterion(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.model.eval()
        result = {}
        with torch.inference_mode():
            mae, mse, mape, num = [], [], [], []
            tmae, tmse, tmape = [], [], []
            for X, y in self.test_loader:
                pred = self.model(X)
                tmae.append(sklearn.metrics.mean_absolute_error(y, pred, multioutput='raw_values'))
                tmse.append(sklearn.metrics.mean_squared_error(y, pred, multioutput='raw_values'))
                tmape.append(sklearn.metrics.mean_absolute_percentage_error(y, pred, multioutput='raw_values'))
                if self.config.get('scaler'):
                    pred = self.scaler_y.inverse_transform(pred)
                    y = self.scaler_y.inverse_transform(y)
                mae.append(sklearn.metrics.mean_absolute_error(y, pred, multioutput='raw_values'))
                mse.append(sklearn.metrics.mean_squared_error(y, pred, multioutput='raw_values'))
                mape.append(sklearn.metrics.mean_absolute_percentage_error(y, pred, multioutput='raw_values'))
                num.append(len(y))
            mae = np.array(mae)
            mse = np.array(mse)
            mape = np.array(mape)
            tmae = np.array(tmae)
            tmse = np.array(tmse)
            tmape = np.array(tmape)
            if len(mae[0]) == 6:
                result['tmae_force'] = np.average(np.average(tmae, axis=1), weights=num)
                result['trmse_force'] = np.average(np.average(tmse, axis=1), weights=num) ** 0.5
                result['tmape_force'] = np.average(np.average(tmape, axis=1), weights=num)
                result['mae_force'] = np.average(np.average(mae, axis=1), weights=num)
                result['rmse_force'] = np.average(np.average(mse, axis=1), weights=num) ** 0.5
                result['mape_force'] = np.average(np.average(mape, axis=1), weights=num)
                result['metric'] = result['trmse_force']
            elif len(mae[0]) == 12:
                result['tmae_coord'] = np.average(np.average(tmae, axis=1), weights=num)
                result['trmse_coord'] = np.average(np.average(tmse, axis=1), weights=num) ** 0.5
                result['tmape_coord'] = np.average(np.average(tmape, axis=1), weights=num)
                result['mae_coord'] = np.average(np.average(mae, axis=1), weights=num)
                result['rmse_coord'] = np.average(np.average(mse, axis=1), weights=num) ** 0.5
                result['mape_coord'] = np.average(np.average(mape, axis=1), weights=num)
                result['metric'] = result['trmse_coord']
            elif len(mae[0]) == 18:
                result['tmae_force'] = np.average(np.average(tmae[:, :6], axis=1), weights=num)
                result['trmse_force'] = np.average(np.average(tmse[:, :6], axis=1), weights=num) ** 0.5
                result['tmape_force'] = np.average(np.average(tmape[:, :6], axis=1), weights=num)
                result['tmae_coord'] = np.average(np.average(tmae[:, 6:], axis=1), weights=num)
                result['trmse_coord'] = np.average(np.average(tmse[:, 6:], axis=1), weights=num) ** 0.5
                result['tmape_coord'] = np.average(np.average(tmape[:, 6:], axis=1), weights=num)
                result['mae_force'] = np.average(np.average(mae[:, :6], axis=1), weights=num)
                result['rmse_force'] = np.average(np.average(mse[:, :6], axis=1), weights=num) ** 0.5
                result['mape_force'] = np.average(np.average(mape[:, :6], axis=1), weights=num)
                result['mae_coord'] = np.average(np.average(mae[:, 6:], axis=1), weights=num)
                result['rmse_coord'] = np.average(np.average(mse[:, 6:], axis=1), weights=num) ** 0.5
                result['mape_coord'] = np.average(np.average(mape[:, 6:], axis=1), weights=num)
                result['metric'] = (result['trmse_force'] + result['trmse_coord']) / 2
            else:
                assert 'output should be only 6 or 12 or 18'
        return result
    

    def eval(self):
        self.model.eval()
        with torch.inference_mode():
            preds = []
            ys = []
            for X, y in self.test_loader:
                pred = self.model(X).detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                if self.config.get('scaler'):
                    pred = self.scaler_y.inverse_transform(pred)
                    y = self.scaler_y.inverse_transform(y)
                preds.append(pred)
                ys.append(y)
        return pred, y


    def save_checkpoint(self, tmp_checkpoint_dir):
        model_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        optimizer_path = os.path.join(tmp_checkpoint_dir, "optimizer.pth")
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
        return tmp_checkpoint_dir
    

    def load_checkpoint(self, tmp_checkpoint_dir):
        model_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        else:
            ray.logger.warning('not found model state dict')
        optimizer_path = os.path.join(tmp_checkpoint_dir, "optimizer.pth")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path))
        else:
            ray.logger.warning('not found optimizer state dict')


    def reset_config(self, new_config: Dict):
        del self.imputer, self.scaler_X, self.scaler_y, self.model, self.criterion, self.optimizer
        index_X = new_config['index_X']
        index_y = new_config['index_y']
        model = new_config['model']
        model_args = new_config['model_args']
        criterion = new_config['criterion']
        optimizer = new_config['optimizer']
        optimizer_args = new_config['optimizer_args']
        imputer = new_config.get('imputer')
        scaler = new_config.get('scaler')
        data_loader = new_config.get('data_loader')

        data, train_indexes, test_indexes = self._import_class(data_loader)()
        self.data.loc[:] = data
        concated_train_indexes = np.concatenate(train_indexes)
        if imputer:
            self.imputer = self._import_class(imputer)(**new_config['imputer_args'])
            index_Xy = (index_X if isinstance(index_X, list) else [index_X]) + (index_y if isinstance(index_y, list) else [index_y])
            self.imputer.fit(self.data.loc[concated_train_indexes, index_Xy])
            self.data.loc[:, index_Xy] = self.imputer.transform(self.data.loc[:, index_Xy])
        if scaler:
            self.scaler_X = self._import_class(scaler)()
            self.scaler_y = self._import_class(scaler)()
            self.scaler_X.fit(self.data.loc[concated_train_indexes, index_X])
            self.scaler_y.fit(self.data.loc[concated_train_indexes, index_y])
            self.data.loc[:, index_X] = self.scaler_X.transform(self.data.loc[:, index_X])
            self.data.loc[:, index_y] = self.scaler_y.transform(self.data.loc[:, index_y])

        self.model = self._import_class(model)(input_size=len(self.data.loc[:, index_X].columns), output_size=len(self.data.loc[:, index_y].columns), **model_args)
        self.criterion = self._import_class(criterion)()
        self.optimizer = self._import_class(optimizer)(self.model.parameters(), **optimizer_args)
        return True
    

from typing import Literal, List
def define_searchspace(model_names:List[Literal['fsr_model.ANN', 'fsr_model.CNN_LSTM', 'fsr_model.LSTM']],
                       index_X:List[Literal['FSR_for_force', 'FSR_for_coord']],
                       index_Y:List[Literal['force', 'x_coord', 'y_coord']],
                       data_loader:Literal['fsr_data.get_index_splited_by_time', 'fsr_data.get_index_splited_by_subject']):
    def inner(trial):
        model = trial.suggest_categorical('model', model_names)
        if model == 'fsr_model.ANN':
            trial.suggest_categorical('model_args/hidden_size', [8, 16, 32, 64, 128])
            trial.suggest_int('model_args/num_layer', 1, 8)
        elif model == 'fsr_model.CNN_LSTM':
            trial.suggest_categorical('model_args/cnn_hidden_size', [8, 16, 32, 64, 128])
            trial.suggest_categorical('model_args/lstm_hidden_size', [8, 16, 32, 64, 128])
            trial.suggest_int('model_args/cnn_num_layer', 1, 8)
            trial.suggest_int('model_args/lstm_num_layer', 1, 8)
        elif model == 'fsr_model.LSTM':
            trial.suggest_categorical('model_args/hidden_size', [8, 16, 32, 64, 128])
            trial.suggest_int('model_args/num_layer', 1, 8)

        trial.suggest_categorical('criterion', ['torch.nn.MSELoss'])
        trial.suggest_categorical('optimizer', [
            'torch.optim.Adam',
            'torch.optim.NAdam',
            'torch.optim.Adagrad',
            'torch.optim.RAdam',
            'torch.optim.SGD',
        ])
        trial.suggest_float('optimizer_args/lr', 1e-5, 1e-1, log=True)
        imputer = trial.suggest_categorical('imputer', ['sklearn.impute.SimpleImputer'])
        if imputer == 'sklearn.impute.SimpleImputer':
            trial.suggest_categorical('imputer_args/strategy', [
                'mean',
                'median',
            ])
        trial.suggest_categorical('scaler', [ 
            'sklearn.preprocessing.StandardScaler',
            'sklearn.preprocessing.MinMaxScaler',
            'sklearn.preprocessing.RobustScaler',
        ])
        return {
            'index_X': index_X,
            'index_y': index_Y,
            'data_loader': data_loader
        }
    return inner


from typing import Literal, List
class SearchSpace:
    def __init__(self, model_names:List[Literal['fsr_model.ANN', 'fsr_model.CNN_LSTM', 'fsr_model.LSTM']],
                       index_X:List[Literal['FSR_for_force', 'FSR_for_coord']],
                       index_Y:List[Literal['force', 'x_coord', 'y_coord']],
                       data_loader:Literal['fsr_data.get_index_splited_by_time', 'fsr_data.get_index_splited_by_subject']):
        self.model_names = model_names
        self.index_X = index_X
        self.index_Y = index_Y
        self.data_loader = data_loader

    def __call__(self, trial):
        model = trial.suggest_categorical('model', self.model_names)
        if model == 'fsr_model.ANN':
            trial.suggest_categorical('model_args/hidden_size', [8, 16, 32, 64, 128])
            trial.suggest_int('model_args/num_layer', 1, 8)
        elif model == 'fsr_model.CNN_LSTM':
            trial.suggest_categorical('model_args/cnn_hidden_size', [8, 16, 32, 64, 128])
            trial.suggest_categorical('model_args/lstm_hidden_size', [8, 16, 32, 64, 128])
            trial.suggest_int('model_args/cnn_num_layer', 1, 8)
            trial.suggest_int('model_args/lstm_num_layer', 1, 8)
        elif model == 'fsr_model.LSTM':
            trial.suggest_categorical('model_args/hidden_size', [8, 16, 32, 64, 128])
            trial.suggest_int('model_args/num_layer', 1, 8)

        trial.suggest_categorical('criterion', ['torch.nn.MSELoss'])
        trial.suggest_categorical('optimizer', [
            'torch.optim.Adam',
            'torch.optim.NAdam',
            'torch.optim.Adagrad',
            'torch.optim.RAdam',
            'torch.optim.SGD',
        ])
        trial.suggest_float('optimizer_args/lr', 1e-5, 1e-1, log=True)
        imputer = trial.suggest_categorical('imputer', ['sklearn.impute.SimpleImputer'])
        if imputer == 'sklearn.impute.SimpleImputer':
            trial.suggest_categorical('imputer_args/strategy', [
                'mean',
                'median',
            ])
        trial.suggest_categorical('scaler', [ 
            'sklearn.preprocessing.StandardScaler',
            'sklearn.preprocessing.MinMaxScaler',
            'sklearn.preprocessing.RobustScaler',
        ])
        return {
            'index_X': self.index_X,
            'index_y': self.index_Y,
            'data_loader': self.data_loader
        }

import ray.tune
import ray.air
import ray.air.integrations.wandb
import ray.tune.schedulers
import ray.tune.search
import ray.tune.search.optuna

def get_tuner(searchspace, wandb_project, cpu_num=2):
    return ray.tune.Tuner(
        trainable=ray.tune.with_resources(
            FSR_Trainable, {'cpu':cpu_num},
        ),
        tune_config=ray.tune.TuneConfig(
            num_samples=100,
            scheduler=ray.tune.schedulers.ASHAScheduler(
                max_t=100,
                grace_period=1,
                reduction_factor=2,
                brackets=1,
                metric='metric',
                mode='min',
            ),
            search_alg=ray.tune.search.optuna.OptunaSearch(
                space=searchspace,
                metric='metric',
                mode='min',
            ),
        ), 
        run_config=ray.air.RunConfig(
            callbacks=[
                ray.air.integrations.wandb.WandbLoggerCallback(project=wandb_project),
            ],
            checkpoint_config=ray.air.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute='metric',
                checkpoint_score_order='min',
                checkpoint_frequency=5,
                checkpoint_at_end=True,
            ),
        ), 
    )

def get_tuner_PBT(searchspace, wandb_project, cpu_num=2):
    return ray.tune.Tuner(
        trainable=ray.tune.with_resources(
            FSR_Trainable, {'cpu':2},
        ),
        tune_config=ray.tune.TuneConfig(
            num_samples=4,
            reuse_actors=True,
            scheduler=ray.tune.schedulers.PopulationBasedTraining(
                time_attr='time_total_s',
                perturbation_interval=5,
                metric='metric',
                mode='min',
                hyperparam_mutations={
                    'model':['fsr_model.ANN'],
                    'model_args':{
                        'hidden_size':[8],
                        'num_layer':[1],
                    },
                    'criterion':['torch.nn.MSELoss'],
                    'optimizer':[
                        'torch.optim.NAdam',
                    ],
                    'optimizer_args':{
                        'lr':ray.tune.loguniform(1e-5, 1e-1),
                    },
                    'imputer':['sklearn.impute.SimpleImputer'],
                    'imputer_args':{
                        'strategy':['mean', 'median'],
                    },
                    'scaler':[ 
                        'sklearn.preprocessing.StandardScaler',
                        'sklearn.preprocessing.MinMaxScaler',
                        'sklearn.preprocessing.RobustScaler',
                    ],
                    'index_X': [['FSR_for_force', 'FSR_for_coord']],
                    'index_y': [['force', 'x_coord', 'y_coord']],
                    'data_loader': ['fsr_data.get_index_splited_by_time']
                },
            ),
        ), 
        run_config=ray.air.RunConfig(
            callbacks=[
                ray.air.integrations.wandb.WandbLoggerCallback(project='FSR-prediction'),
            ],
            checkpoint_config=ray.air.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute='metric',
                checkpoint_score_order='min',
                checkpoint_frequency=5,
                checkpoint_at_end=True,
            ),
        ), 
    )
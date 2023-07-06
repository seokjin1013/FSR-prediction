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
        model_args = config['model_args']
        criterion = config['criterion']
        optimizer = config['optimizer']
        optimizer_args = config['optimizer_args']
        imputer = config.get('imputer')
        scaler = config.get('scaler')
        data_loader = config.get('data_loader')

        data, train_indexes, test_indexes = self._import_class(data_loader)()
        data = data.copy()
        concated_train_indexes = np.concatenate(train_indexes)
        if imputer:
            self.imputer = self._import_class(imputer)(**config['imputer_args'])
            index_Xy = (index_X if isinstance(index_X, list) else [index_X]) + (index_y if isinstance(index_y, list) else [index_y])
            self.imputer.fit(data.loc[concated_train_indexes, index_Xy])
            data.loc[:, index_Xy] = self.imputer.transform(data.loc[:, index_Xy])
        if scaler:
            self.scaler_X = self._import_class(scaler)()
            self.scaler_y = self._import_class(scaler)()
            self.scaler_X.fit(data.loc[concated_train_indexes, index_X])
            self.scaler_y.fit(data.loc[concated_train_indexes, index_y])
            data.loc[:, index_X] = self.scaler_X.transform(data.loc[:, index_X])
            data.loc[:, index_y] = self.scaler_y.transform(data.loc[:, index_y])
        train_dataset = fsr_data.FSRDataset(data.loc[:, index_X], data.loc[:, index_y], train_indexes)
        test_dataset = fsr_data.FSRDataset(data.loc[:, index_X], data.loc[:, index_y], test_indexes)

        self.model = self._import_class(model)(input_size=len(data.loc[:, index_X].columns), output_size=len(data.loc[:, index_y].columns), **model_args)
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
        with torch.inference_mode():
            mae, mse, mape, num = [], [], [], []
            for X, y in self.test_loader:
                pred = self.model(X)
                if self.config.get('scaler'):
                    pred = self.scaler_y.inverse_transform(pred)
                    y = self.scaler_y.inverse_transform(y)
                mae.append(sklearn.metrics.mean_absolute_error(y, pred))
                mse.append(sklearn.metrics.mean_squared_error(y, pred))
                mape.append(sklearn.metrics.mean_absolute_percentage_error(y, pred))
                num.append(len(y))
            mae = np.average(mae, weights=num)
            mse = np.average(mse, weights=num)
            mape = np.average(mape, weights=num)
            rmse = mse ** 0.5
        return {'rmse': rmse, 'mae':mae, 'mape':mape}
    

    def eval(self):
        self.model.eval()
        with torch.inference_mode():
            results = []
            for X, y in self.test_loader:
                pred = self.model(X).detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                if self.config.get('scaler'):
                    pred = self.scaler_y.inverse_transform(pred)
                    y = self.scaler_y.inverse_transform(y)
                results.append((pred, y))
        return results


    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir
    

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))
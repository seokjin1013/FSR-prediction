import ray.tune
import datasource
import torch
import sklearn.metrics
import numpy as np
import os

class Trainable(ray.tune.Trainable):
    @staticmethod
    def _import_class(name:str):
        import importlib
        index = name.rfind('.')
        module_name = name[:index] if index != -1 else '__main__'
        class_name = name[index + 1:]
        return getattr(importlib.import_module(module_name), class_name)


    def setup(self, config, data):
        model = config['model']
        model_args = config['model_args']
        criterion = config['criterion']
        optimizer = config['optimizer']
        optimizer_args = config['optimizer_args']
        imputer = config['imputer']
        imputer_args = config['imputer_args']
        scaler = config['scaler']

        train_indexes, test_indexes = datasource.get_index_splited_by_time(data)
        self.imputer = self._import_class(imputer)(**imputer_args)
        self.scaler_X = self._import_class(scaler)()
        self.scaler_y = self._import_class(scaler)()

        index_X = 'FSR_for_coord'
        index_y = 'x_coord'
        concated_train_indexes = np.concatenate(train_indexes)
        self.imputer.fit(data.loc[concated_train_indexes, [index_X, index_y]])
        self.scaler_X.fit(data.loc[concated_train_indexes, index_X])
        self.scaler_y.fit(data.loc[concated_train_indexes, index_y])
        data.loc[:, [index_X, index_y]] = self.imputer.transform(data.loc[:, [index_X, index_y]])
        data.loc[:, index_X] = self.scaler_X.transform(data.loc[:, index_X])
        data.loc[:, index_y] = self.scaler_y.transform(data.loc[:, index_y])

        train_dataset = datasource.FSRDataset(data.loc[:, index_X], data.loc[:, index_y], train_indexes)
        test_dataset = datasource.FSRDataset(data.loc[:, index_X], data.loc[:, index_y], test_indexes)

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
        with torch.no_grad():
            mae, mse, mape, num = [], [], [], []
            for X, y in self.test_loader:
                pred = self.model(X)
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

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir
    
    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))
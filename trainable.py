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
        scaler = config['scaler']

        self.model = self._import_class(model)(**model_args)
        self.criterion = self._import_class(criterion)()
        self.optimizer = self._import_class(optimizer)(self.model.parameters(), **optimizer_args)

        train_index, test_index = datasource.get_index_splited_by_time(data)
        scaler = self._import_class(scaler)()
        data[:] = scaler.fit_transform(data)
        train_dataset = datasource.FSRDataset(data['FSR_for_force'], data['force'], train_index)
        test_dataset = datasource.FSRDataset(data['FSR_for_force'], data['force'], test_index)
        
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
                mae.append(sklearn.metrics.mean_absolute_error(y, pred.cpu().detach()))
                mse.append(sklearn.metrics.mean_squared_error(y, pred.cpu().detach()))
                mape.append(sklearn.metrics.mean_absolute_percentage_error(y, pred.cpu().detach()))
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
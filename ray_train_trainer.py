import datasource
import torch.utils.data
import sklearn.preprocessing
import numpy as np
from ray.air import session
from ray.train.torch import TorchCheckpoint, prepare_data_loader, prepare_model

def FSR_train_loop(config):
    def _import_class(name:str):
        import importlib
        index = name.rfind('.')
        module_name = name[:index] if index != -1 else '__main__'
        class_name = name[index + 1:]
        return getattr(importlib.import_module(module_name), class_name)
    
    model = config['model']
    criterion = config['criterion']
    optimizer = config['optimizer']
    imputer = config.get('imputer')
    scaler = config.get('scaler')
    index_X = config['index_X']
    index_y = config['index_y']

    data = datasource.get_data()
    train_indexes, test_indexes = datasource.get_index_splited_by_time(data)

    concated_train_indexes = np.concatenate(train_indexes)
    if imputer:
        imputer = _import_class(imputer)(**config['imputer_args'])
        imputer.fit(data.loc[concated_train_indexes, [index_X, index_y]])
    if scaler:
        scaler_X = _import_class(scaler)()
        scaler_y = _import_class(scaler)()
        scaler_X.fit(data.loc[concated_train_indexes, index_X])
        scaler_y.fit(data.loc[concated_train_indexes, index_y])
        data.loc[:, index_X] = scaler_X.transform(data.loc[:, index_X])
        data.loc[:, index_y] = scaler_y.transform(data.loc[:, index_y])
    train_dataset = datasource.FSRDataset(data.loc[:, index_X], data.loc[:, index_y], train_indexes)
    test_dataset = datasource.FSRDataset(data.loc[:, index_X], data.loc[:, index_y], test_indexes)

    model = _import_class(model)(input_size=len(data.loc[:, index_X].columns), output_size=len(data.loc[:, index_y].columns), **config['model_args'])
    model = prepare_model(model)
    criterion = _import_class(criterion)()
    optimizer = _import_class(optimizer)(model.parameters(), **config['optimizer_args'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None)
    train_loader = prepare_data_loader(train_loader)
    test_loader = prepare_data_loader(test_loader)

    while True:
        model.train()
        for X, y in train_loader:
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            mae, mse, mape, num = [], [], [], []
            for X, y in test_loader:
                pred = model(X)
                if scaler:
                    pred = scaler_y.inverse_transform(pred)
                    y = scaler_y.inverse_transform(y)
                mae.append(sklearn.metrics.mean_absolute_error(y, pred))
                mse.append(sklearn.metrics.mean_squared_error(y, pred))
                mape.append(sklearn.metrics.mean_absolute_percentage_error(y, pred))
                num.append(len(y))
            mae = np.average(mae, weights=num)
            mse = np.average(mse, weights=num)
            mape = np.average(mape, weights=num)
            rmse = mse ** 0.5
        session.report(
            dict(rmse=rmse, mae=mae, mape=mape),
            checkpoint=TorchCheckpoint.from_dict(
                dict(model=model.state_dict(), optimizer=optimizer.state_dict()),
            ),
        ),

from ray.air.config import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.tune.stopper import TrialPlateauStopper, ExperimentPlateauStopper, CombinedStopper

trainer = TorchTrainer(
    train_loop_per_worker=FSR_train_loop,
    scaling_config=ScalingConfig(
        num_workers=4,
        use_gpu=False,
    ),
    run_config=RunConfig(
        stop=CombinedStopper(
            # TrialPlateauStopper(metric='rmse'),
            # ExperimentPlateauStopper(metric='rmse'),
        ),
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute='rmse',
            checkpoint_score_order='min',
        ),
    ),
)
import typing
import yaml
import numpy as np

class Tasks:
    TASKS = yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)['experiment_id']

    def __call__(self, model:typing.Literal['ANN', 'CNN-LSTM', 'LSTM'], task:int):
        assert 1 <= task <= 6
        assert model in ['ANN', 'CNN-LSTM', 'LSTM']
        model_index = {'ANN':0, 'CNN-LSTM':1, 'LSTM':2}[model]
        return Tasks.TASKS[model_index + (task - 1) * 3]

    def __iter__(self):
        return iter(Tasks.TASKS)
    
    def __getitem__(self, index):
        return Tasks.TASKS[index]
    

TASKS = Tasks()


def get_GRF(data):
    data = np.array(data)
    return data.sum(axis=1)


def get_CoP(force, x_coord, y_coord):
    force = np.array(force)
    x_coord = np.array(x_coord)
    y_coord = np.array(y_coord)

    # strict check
    # assert ((force == 0) == np.isnan(x_coord)).all()
    # assert ((force == 0) == np.isnan(y_coord)).all()
    # valid_check = ((force == 0) == (np.isnan(x_coord))) & ((force == 0) == (np.isnan(x_coord)))
    # x, y = np.where(valid_check == False)
    # display(data.loc[x, [('force', 'D'), ('x_coord', 'D')]])

    # weak check
    assert not ((force != 0) & (np.isnan(x_coord))).any()
    assert not ((force != 0) & (np.isnan(y_coord))).any()

    x_nan = np.isnan(x_coord)
    y_nan = np.isnan(y_coord)
    x_coord[x_nan] = 0
    y_coord[y_nan] = 0

    x_force = force
    y_force = force.copy()
    x_force[x_nan] = 0
    y_force[y_nan] = 0
    
    with np.errstate(divide='ignore'):
        x = (x_force * x_coord).sum(axis=1) / x_force.sum(axis=1)
        y = (y_force * y_coord).sum(axis=1) / y_force.sum(axis=1)
    
    assert (np.isnan(x) == np.isnan(y)).all()
    return np.stack([x, y], axis=0)
def get_best_checkpoint(experiment_id, metric, mode):
    from fsr_trainable import FSR_Trainable
    import ray.tune
    path = '/home/seokj/ray_results/' + experiment_id
    result_grid = ray.tune.Tuner.restore(path, FSR_Trainable).get_results()
    if mode == 'max':
        mode = True
    elif mode == 'min':
        mode = False
    else:
        raise AttributeError('mode should be either max or min')

    target = float('inf') * (-1 if mode else 1)
    for result in result_grid:
        if result.best_checkpoints:
            checkpoint, metrics = result.best_checkpoints[-1]
            if target > metrics[metric] * (-1 if mode else 1):
                target = metrics[metric] * (-1 if mode else 1)
                best_checkpoint = checkpoint
                best_metrics = metrics
    
    return best_checkpoint, best_metrics


def get_best_prediction(checkpoint, metrics):
    from fsr_trainable import FSR_Trainable
    trainer = FSR_Trainable(config=metrics['config'])
    trainer.restore(checkpoint)
    return trainer.eval()
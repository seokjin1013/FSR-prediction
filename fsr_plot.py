def plot_list(data, *plot, axis_size=(2.5, 2.5), max_col=5):
    import matplotlib.pyplot as plt
    count = len(data)
    ncol = (count + max_col - 1) // max_col
    nrow = max_col
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * axis_size[0], ncol * axis_size[1]), sharex=True, sharey=True)
    axes = axes.flatten() if count > 1 else [axes]
    for i in range(count):
        for p in plot:
            p(axes[i], data[i])
    
    handles, labels = axes[0].get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, loc='outside right upper')
    fig.tight_layout()
    for i in range(count, len(axes)):
        fig.delaxes(axes[i])


def plot_coord(ax, data):
    pred, y = data
    ax.set_xlim(-10, 20)
    ax.set_ylim(30, 0)
    ax.set_aspect('equal')
    ax.set_xticks([0, 10], labels=['outside', 'inside'])
    ax.set_yticks([5, 25], labels=['toe', 'heel'])
    for i in range(6):
        ax.scatter(y[:, i], y[:, i + 6], s=5, label='ABCDEF'[i])
    handles = ax.get_legend_handles_labels()[0]
    for i in range(6):
        ax.scatter(pred[:, i], pred[:, i + 6], s=5, label='ABCDEF'[i], edgecolors='black', c=handles[i].get_facecolor())


def plot_force(ax, data):
    pred, y = data
    for i in range(6):
        ax.plot(pred[:, i], label='ABCDEF'[i])
    handles = ax.get_legend_handles_labels()[0]
    for i in range(6):
        ax.plot(y[:, i], label='ABCDEF'[i], c=handles[i].get_color(), linestyle='--')


def plot_cop(ax, data):
    pred, y = data
    ax.set_xlim(-10, 20)
    ax.set_ylim(30, 0)
    ax.set_aspect('equal')
    ax.set_xticks([0, 10], labels=['outside', 'inside'])
    ax.set_yticks([5, 25], labels=['toe', 'heel'])
    cop_x = (pred[:, :6] * pred[:, 6:12]).sum(axis=1)/pred[:, :6].sum(axis=1)
    cop_y = (pred[:, :6] * pred[:, 12:]).sum(axis=1)/pred[:, :6].sum(axis=1)
    cop_x = cop_x[(pred[:, :6] != 0).any(axis=1)]
    cop_y = cop_y[(pred[:, :6] != 0).any(axis=1)]
    ax.plot(cop_x, cop_y)
    cop_x = (y[:, :6] * y[:, 6:12]).sum(axis=1)/y[:, :6].sum(axis=1)
    cop_y = (y[:, :6] * y[:, 12:]).sum(axis=1)/y[:, :6].sum(axis=1)
    cop_x = cop_x[(y[:, :6] != 0).any(axis=1)]
    cop_y = cop_y[(y[:, :6] != 0).any(axis=1)]
    ax.plot(cop_x, cop_y)
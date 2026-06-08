import re
import matplotlib.pyplot as plt


def parse_log(path):
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    with open(path) as f:
        for line in f:
            m = re.match(r'step=(\d+)\s+train_loss=([\d.]+)', line)
            if m:
                train_steps.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                continue
            m = re.match(r'step=(\d+)\s+eval_loss=([\d.]+)', line)
            if m:
                eval_steps.append(int(m.group(1)))
                eval_losses.append(float(m.group(2)))
    return train_steps, train_losses, eval_steps, eval_losses


def plot_runs(runs, show_train=False, show_eval=True, title='Training & Eval Loss'):
    """Plot train and/or eval loss curves for one or more runs.

    Args:
        runs: dict mapping run name -> path to training_log.txt
        show_train: if True, plot train loss curves
        show_eval: if True, plot eval loss curves
        title: plot title
    """
    for i, (name, path) in enumerate(runs.items()):
        c = f'C{i % 10}'
        ts, tl, es, el = parse_log(path)
        if show_train:
            plt.plot(ts, tl, color=c, label=f'{name} train')
        if show_eval:
            plt.plot(es, el, color=c, linestyle='--', marker='o', label=f'{name} eval')

    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    runs = {
        'run1': '/Users/xuge/cs336/checkpoints/debug/training_log.txt',
        'run2': '/Users/xuge/cs336/checkpoints/debug/50390296-ade0-4a47-9d96-fe40ce0337af/training_log.txt',
    }
    plot_runs(runs)

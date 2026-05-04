"""Metrics and plotting helpers (minimal)."""
import matplotlib.pyplot as plt


def plot_loss_curve(losses, out_path=None):
    plt.figure()
    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if out_path:
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()

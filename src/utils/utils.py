import os
import matplotlib.pylab as plt
import metrics
import numpy as np

import threading
import queue


def get_input(message, channel):
    response = input(message)
    channel.put(response)


def input_with_timeout(message, timeout, default_answer):
    channel = queue.Queue()
    message = message + " [{} sec timeout] ".format(timeout)
    thread = threading.Thread(target=get_input, args=(message, channel))
    # by setting this as a daemon thread, python won't wait for it to complete
    thread.daemon = True
    thread.start()

    try:
        response = channel.get(True, timeout)
        return response
    except queue.Empty:
        pass
    return default_answer


def ignore_func(dir, file_list, ext=".py"):
    ignored = []
    for file_name in file_list:
        file_path = os.path.join(dir, file_name)
        if not os.path.isdir(file_path):
            if not file_path.endswith(ext):
                ignored.append(file_name)
    return ignored


def imshow_(x, **kwargs):
    if x.ndim == 2:
        plt.imshow(x, interpolation="nearest", **kwargs)
    elif x.ndim == 1:
        plt.imshow(x[:, None].T, interpolation="nearest", **kwargs)
        plt.yticks([])
    plt.axis("tight")


def visualizeHeatMapPredictions(P_test, y_test, expFolder, videoName):
    # np.random.seed(1)
    plt.rcParams["figure.figsize"] = 10, 3

    x = np.arange(0, len(P_test))
    fig, (ax, ax2, ax3) = plt.subplots(nrows=3, sharex=True)

    extent = [x[0] - (x[1] - x[0]) / 2., x[-1] + (x[1] - x[0]) / 2., 0, 1]
    im = ax.imshow(np.array(y_test)[np.newaxis, :], cmap="plasma", aspect="auto", extent=extent)
    ax.set_yticks([])
    ax.set_xlim(extent[0], extent[1])

    im = ax2.imshow(np.array(P_test)[np.newaxis, :], cmap="plasma", aspect="auto", extent=extent)
    ax2.set_yticks([])
    ax3.plot(x, P_test)
    ax3.set_yticks([])
    ax3.set_xticks([])

    fig.subplots_adjust(right=0.82)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

    cb = plt.colorbar(im, cax=cbar_ax)

    # plt.show()

    saveFolder = os.path.join(expFolder, "heatmap")
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    plt.savefig(os.path.join(saveFolder, videoName + ".pdf"))
    plt.close()


def visualize_temporal_action(P_test, y_test, save_path, videoName):
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.broken_barh([(0, len(y_test))], (3, 5), facecolors='papayawhip')
    ax.broken_barh([(0, len(P_test))], (0, 2), facecolors='papayawhip')

    ax.set_ylim(0, 5)
    ax.set_xlim(0, len(y_test))

    ax.set_xlabel('seconds')
    ax.set_yticks([1, 4])
    ax.set_yticklabels(['Pred', 'GT'])

    p_label, p_start, p_end = metrics.get_labels_start_end_time(P_test)
    y_label, y_start, y_end = metrics.get_labels_start_end_time(y_test)

    for ystart, yend in zip(y_start, y_end):
        ax.broken_barh([(ystart, yend)], (3, 5), facecolors='darkred')

    for pstart, pend in zip(p_start, p_end):
        ax.broken_barh([(pstart, pend)], (0, 2), facecolors='darkred')

    # plt.show()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    visualizeHeatMapPredictions(np.cumsum(np.random.randn(50)), np.cumsum(np.random.randn(50)), "", "")
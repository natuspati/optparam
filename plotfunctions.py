import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class PlotContainer(object):
    def __init__(self):
        self.colorlist = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
        self.fsize = (16, 10)
        self.markersize = 100

    def projections_on_img(self, imgpath, projections_dict, path=None):
        fig, ax = plt.subplots(figsize=self.fsize)
        fig.patch.set_facecolor('None')
        ax.imshow(plt.imread(imgpath))
        for index, (projection_label, projections) in enumerate(projections_dict.items()):
            projections_left = projections[0]
            projections_right = projections[1]
            xs = np.hstack((projections_left[:, 0], projections_right[:, 0]))
            ys = np.hstack((projections_left[:, 1], projections_right[:, 1]))
            ax.scatter(xs, ys, label=projection_label,  s=self.markersize, facecolors='none',
                       edgecolors=self.colorlist[index])
        ax.legend(fontsize=20)
        if path:
            fig.savefig(Path(path), format="png")

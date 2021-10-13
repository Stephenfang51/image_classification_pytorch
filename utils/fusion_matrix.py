import numpy as np
import torch
import matplotlib.pyplot as plt


def calc_cmtx(out, target, num_classes, reduce=None):
    out = out.cpu()
    target = target.cpu()
    cmtx = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    preds = torch.argmax(out, dim=1) #get index of max value which means top1 prediciton
    inds, cnts = torch.unique(torch.stack([target, preds]), dim=1, return_counts=True)
    #return the unique elements and counts for each unique element
    cmtx[inds[0], inds[1]] += cnts #zero metrix add each class unique counts
    if reduce == "mean":
        cmtx /= torch.mean(cmtx, dim=1, keepdim=True)
    return cmtx

def save_cmtx(cmtx, show_value=True, text_size=15, title="", save_to_file=""):
    fig, ax = plt.subplots()
    ax.matshow(cmtx)
    if show_value:
        for (i, j), z in np.ndenumerate(cmtx):
            ax.text(j, i, '{:.0f}'.format(z), ha='center', va='center', size=text_size)
    ax.set_title(title)

    if save_to_file:
        fig.savefig(save_to_file, dpi=256)
    # ax.show()

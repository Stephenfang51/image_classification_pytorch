import torch as t
import torch.nn as nn
import torch.nn.functional as F

class Focalloss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(Focalloss, self).__init__()
        if alpha is None:
            self.alpha = t.ones(class_num, 1)
        else:
            if alpha:
                #                 self.alpha = t.ones(class_num, 1, requires_grad=True)
                self.alpha = t.tensor(alpha, requires_grad=True)
            else:
                self.alpha = t.ones(class_num, 1 * alpha).cuda()
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)  # 經過softmax 概率

        class_mask = inputs.data.new(N, C).fill_(0)  # 做一個跟input一樣的， 並且都填入0
        class_mask = class_mask.requires_grad_()
        # ----以下不太理解------------#
        ids = targets.view(-1, 1)
        class_mask.data.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        #         alpha = self.alpha[ids.data.view(-1, 1)]
        alpha = self.alpha

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()

        batch_loss = -alpha * (t.pow((1 - probs), self.gamma)) * log_p  # 對應下面公式

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=255,
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1 - lb_one_hot)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, t.arange(label.size(1)), *b]] = 0

        if self.reduction == 'mean':
            loss = -t.sum(t.sum(logs * label, dim=1)) / n_valid
        elif self.reduction == 'none':
            loss = -t.sum(logs * label, dim=1)
        return loss
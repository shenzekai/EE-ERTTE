from utills.metrics import *
import torch.nn as nn
from models import ConSTGAT, MetaERTTE, SSML, WDR, WDR_LC, MLPTTE


class FTML(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, FLAGS):
        """

        :param FLAGS:
        """
        super(FTML, self).__init__()

        self.lr = FLAGS.lr
        if FLAGS.model == 'ConSTGAT':
            self.net = ConSTGAT(FLAGS)
        elif FLAGS.model == 'SSML':
            self.net = SSML(FLAGS)
        elif FLAGS.model == 'MetaER-TTE':
            self.net = MetaERTTE(FLAGS)
        elif FLAGS.model == 'WDR':
            self.net = WDR(FLAGS)
        elif FLAGS.model == 'WDR_LC':
            self.net = WDR_LC(FLAGS)
        elif FLAGS.model == 'MLPTTE':
            self.net = MLPTTE(FLAGS)
        else:
            raise ValueError('Model not defined')
            exit()

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def forward(self, *args):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        origin_weight = self.net.parameters()
        wide_index, wide_value, deep_category, deep_real = args[0:4]
        all_link_feature, all_real, all_flow, all_linkdistance = args[4:8]
        all_num, all_mid_num, all_re_num = args[-7:-4]
        label_spt = args[-4:-2]
        label_qry = args[-2]
        loss_func = args[-1]
        all_loss = 0
        pr_loss = 0
        er_loss = 0
        # Pre-training predict the full travel time and the traveled travel time and corresponding confidence interval
        y, y_tr = self.net(wide_index, wide_value, deep_category, deep_real, all_link_feature, all_num,
                           all_flow, all_linkdistance, all_real, all_mid_num)
        if loss_func == mape:
            pr_loss = loss_func(y_tr, label_spt[1].float())
        else:
            pr_loss = loss_func(y_tr, label_spt[1].float())
            pr_MPIW = torch.mean(y_tr[:, 2] - y_tr[:, 0])
            # loss = loss_func(y, label_spt[0].float())
        all_loss += pr_loss
        all_loss += pr_MPIW
        # Meta-training update params
        grads = torch.autograd.grad(pr_loss, self.net.parameters(), retain_graph=True, allow_unused=True)
        fast_weights = list(map(lambda p: p[1] - self.lr * p[0] if p[0] is not None else p[1],
                                zip(grads, self.net.parameters())))  # 更新网络参数
        self.update_params(self.net, fast_weights)
        # Re-training predict the remaining travel time
        re_link_feature, re_real, re_flow, re_linkdistance = args[8:12]
        y, target = self.net(wide_index, wide_value, deep_category, deep_real, re_link_feature, all_re_num, \
                             re_flow, re_linkdistance, re_real, re_target=label_qry)
        if loss_func == mape:
            er_loss = loss_func(y, target.float(), all_re_num)
        else:
            er_loss = loss_func(y, target.float())
        all_loss += er_loss

        self.update_params(self.net, origin_weight)  # 参数还原


        return all_loss, pr_loss, er_loss

    @staticmethod
    def update_params(model, fastweight):
        """ Update the learnable parameters with gradients. """
        for params, fast in zip(model.parameters(), fastweight):
            params.data = fast

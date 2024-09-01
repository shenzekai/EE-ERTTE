import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import time
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape


# 构建Dataset
class GiscupDataset(torch.utils.data.Dataset):
    def __init__(self, seq_data, FLAGS, device):
        all_num = []
        all_mid_num = []
        all_re_num = []
        all_id = []
        all_id7 = []
        all_id4 = []
        all_id1 = []
        all_time = []
        all_time7 = []
        all_time4 = []
        all_time1 = []
        all_flow = []
        all_flow7 = []
        all_flow4 = []
        all_flow1 = []
        all_linkdistance = []
        all_linkdistance7 = []
        all_linkdistance4 = []
        all_linkdistance1 = []
        all_label = []
        all_mid_label = []
        all_re_label = []
        all_cross = []
        all_oneway = []
        all_oneway7 = []
        all_oneway4 = []
        all_oneway1 = []
        all_reversed = []
        all_reversed7 = []
        all_reversed4 = []
        all_reversed1 = []
        all_highway = []
        all_highway7 = []
        all_highway4 = []
        all_highway1 = []
        all_lane = []
        all_lane7 = []
        all_lane4 = []
        all_lane1 = []
        padding_size = FLAGS.segment_num
        padding_size7 = FLAGS.Lnum7
        padding_size4 = FLAGS.Lnum4
        padding_size1 = FLAGS.Lnum1
        drivers_num = FLAGS.drivers_num
        print(seq_data.shape)

        for i in range(len(seq_data)):
            length = len(seq_data[i][4])
            #             all_num.append(length)  # list 500
            ids = seq_data[i][4] + [-1] * (padding_size - length)
            all_id.append(ids)  # link id

            id7 = seq_data[i][4][seq_data[i][22]:] + [-1] * (padding_size7 - seq_data[i][23])  # traveled link id
            id4 = seq_data[i][4][seq_data[i][24]:] + [-1] * (padding_size4 - seq_data[i][25])  # traveled link id
            id1 = seq_data[i][4][seq_data[i][26]:] + [-1] * (padding_size1 - seq_data[i][27])  # traveled link id
            all_id7.append(id7)  # list
            all_id4.append(id4)  # list
            all_id1.append(id1)  # list

            time = seq_data[i][10] + [-1] * (padding_size - length)
            time7 = seq_data[i][10][seq_data[i][22]:] + [-1] * (padding_size7 - seq_data[i][23])
            time4 = seq_data[i][10][seq_data[i][24]:] + [-1] * (padding_size4 - seq_data[i][25])
            time1 = seq_data[i][10][seq_data[i][26]:] + [-1] * (padding_size1 - seq_data[i][27])
            all_time.append(time)  # list
            all_time7.append(time7)  # list
            all_time4.append(time4)  # list
            all_time1.append(time1)  # list

            flow = seq_data[i][11] + [0] * (padding_size - length)
            flow7 = seq_data[i][11][seq_data[i][22]:] + [0] * (padding_size7 - seq_data[i][23])
            flow4 = seq_data[i][11][seq_data[i][24]:] + [0] * (padding_size4 - seq_data[i][25])
            flow1 = seq_data[i][11][seq_data[i][26]:] + [0] * (padding_size1 - seq_data[i][27])
            all_flow.append(flow)  # list
            all_flow7.append(flow7)  # list
            all_flow4.append(flow4)  # list
            all_flow1.append(flow1)  # list

            linkdistance = seq_data[i][2] + [-1] * (padding_size - length)
            linkdistance7 = seq_data[i][2][seq_data[i][22]:] + [-1] * (padding_size7 - seq_data[i][23])
            linkdistance4 = seq_data[i][2][seq_data[i][24]:] + [-1] * (padding_size4 - seq_data[i][25])
            linkdistance1 = seq_data[i][2][seq_data[i][26]:] + [-1] * (padding_size1 - seq_data[i][27])
            all_linkdistance.append(linkdistance)  # list
            all_linkdistance7.append(linkdistance7)  # list
            all_linkdistance4.append(linkdistance4)  # list
            all_linkdistance1.append(linkdistance1)  # list

            highway = seq_data[i][14] + [-1] * (padding_size - length)
            highway7 = seq_data[i][14][seq_data[i][22]:] + [-1] * (padding_size7 - seq_data[i][23])
            highway4 = seq_data[i][14][seq_data[i][24]:] + [-1] * (padding_size4 - seq_data[i][25])
            highway1 = seq_data[i][14][seq_data[i][26]:] + [-1] * (padding_size1 - seq_data[i][27])
            all_highway.append(highway)  # list
            all_highway7.append(highway7)  # list
            all_highway4.append(highway4)  # list
            all_highway1.append(highway1)  # list

            lane = seq_data[i][15] + [-1] * (padding_size - length)
            lane7 = seq_data[i][15][seq_data[i][22]:] + [-1] * (padding_size7 - seq_data[i][23])
            lane4 = seq_data[i][15][seq_data[i][24]:] + [-1] * (padding_size4 - seq_data[i][25])
            lane1 = seq_data[i][15][seq_data[i][26]:] + [-1] * (padding_size1 - seq_data[i][27])
            all_lane.append(lane)  # list
            all_lane7.append(lane7)  # list
            all_lane4.append(lane4)  # list
            all_lane1.append(lane1)  # list

            oneway = seq_data[i][12] + [-1] * (padding_size - length)
            oneway7 = seq_data[i][12][seq_data[i][22]:] + [-1] * (padding_size7 - seq_data[i][23])
            oneway4 = seq_data[i][12][seq_data[i][24]:] + [-1] * (padding_size4 - seq_data[i][25])
            oneway1 = seq_data[i][12][seq_data[i][26]:] + [-1] * (padding_size1 - seq_data[i][27])
            all_oneway.append(oneway)  # list
            all_oneway7.append(oneway7)  # list
            all_oneway4.append(oneway4)  # list
            all_oneway1.append(oneway1)  # list

            reversed = seq_data[i][13] + [-1] * (padding_size - length)
            reversed7 = seq_data[i][13][seq_data[i][22]:] + [-1] * (padding_size7 - seq_data[i][23])
            reversed4 = seq_data[i][13][seq_data[i][24]:] + [-1] * (padding_size4 - seq_data[i][25])
            reversed1 = seq_data[i][13][seq_data[i][26]:] + [-1] * (padding_size1 - seq_data[i][27])
            all_reversed.append(reversed)  # list
            all_reversed7.append(reversed7)  # list
            all_reversed4.append(reversed4)  # list
            all_reversed1.append(reversed1)  # list

            all_num.append(seq_data[i][5])  # link num porto-1记住
            all_cross.append(seq_data[i][6])  # cross num
            all_mid_num.append([seq_data[i][22], seq_data[i][24], seq_data[i][26]])  # mid link num
            all_re_num.append([seq_data[i][23], seq_data[i][25], seq_data[i][27]])  # re link num
            all_label.append(seq_data[i][7])  # label
            all_mid_label.append([seq_data[i][16], seq_data[i][18], seq_data[i][20]])  # mid_label
            all_re_label.append([seq_data[i][17], seq_data[i][19], seq_data[i][21]])  # re_label
            # all_s = [slices[i]] * padding_size
            # all_slice.append(all_s)
        self.all_num = torch.tensor(all_num, dtype=torch.int64)
        self.all_mid_num = torch.tensor(all_mid_num, dtype=torch.int64)
        self.all_re_num = torch.tensor(all_re_num, dtype=torch.int64)
        # link 平均通行时间 70% 40% 10%
        self.all_real = torch.tensor(all_time, dtype=torch.float)
        self.all_real7 = torch.tensor(all_time7, dtype=torch.float)
        self.all_real4 = torch.tensor(all_time4, dtype=torch.float)
        self.all_real1 = torch.tensor(all_time1, dtype=torch.float)
        # 流量特征 70% 40% 10%
        self.all_flow = torch.tensor(all_flow, dtype=torch.int)
        self.all_flow7 = torch.tensor(all_flow7, dtype=torch.int)
        self.all_flow4 = torch.tensor(all_flow4, dtype=torch.int)
        self.all_flow1 = torch.tensor(all_flow1, dtype=torch.int)
        # 行程中的路段ID长度 70% 40% 10%
        self.all_linkdistance = torch.tensor(all_linkdistance, dtype=torch.float)
        self.all_linkdistance7 = torch.tensor(all_linkdistance7, dtype=torch.float)
        self.all_linkdistance4 = torch.tensor(all_linkdistance4, dtype=torch.float)
        self.all_linkdistance1 = torch.tensor(all_linkdistance1, dtype=torch.float)
        # 行程中的路段ID序列 70% 40% 10%
        self.all_id = torch.tensor(all_id, dtype=torch.int).unsqueeze(2) + 1
        self.all_id7 = torch.tensor(all_id7, dtype=torch.int).unsqueeze(2) + 1
        self.all_id4 = torch.tensor(all_id4, dtype=torch.int).unsqueeze(2) + 1
        self.all_id1 = torch.tensor(all_id1, dtype=torch.int).unsqueeze(2) + 1

        self.all_highway = torch.tensor(all_highway, dtype=torch.int).unsqueeze(2) + 1
        self.all_highway7 = torch.tensor(all_highway7, dtype=torch.int).unsqueeze(2) + 1
        self.all_highway4 = torch.tensor(all_highway4, dtype=torch.int).unsqueeze(2) + 1
        self.all_highway1 = torch.tensor(all_highway1, dtype=torch.int).unsqueeze(2) + 1

        self.all_lane = torch.tensor(all_lane, dtype=torch.int).unsqueeze(2) + 1
        self.all_lane7 = torch.tensor(all_lane7, dtype=torch.int).unsqueeze(2) + 1
        self.all_lane4 = torch.tensor(all_lane4, dtype=torch.int).unsqueeze(2) + 1
        self.all_lane1 = torch.tensor(all_lane1, dtype=torch.int).unsqueeze(2) + 1

        self.all_oneway = torch.tensor(all_oneway, dtype=torch.int).unsqueeze(2) + 1
        self.all_oneway7 = torch.tensor(all_oneway7, dtype=torch.int).unsqueeze(2) + 1
        self.all_oneway4 = torch.tensor(all_oneway4, dtype=torch.int).unsqueeze(2) + 1
        self.all_oneway1 = torch.tensor(all_oneway1, dtype=torch.int).unsqueeze(2) + 1

        self.all_reversed = torch.tensor(all_reversed, dtype=torch.int).unsqueeze(2) + 1
        self.all_reversed7 = torch.tensor(all_reversed7, dtype=torch.int).unsqueeze(2) + 1
        self.all_reversed4 = torch.tensor(all_reversed4, dtype=torch.int).unsqueeze(2) + 1
        self.all_reversed1 = torch.tensor(all_reversed1, dtype=torch.int).unsqueeze(2) + 1

        self.targets = torch.tensor(all_label)
        self.mid_targets = torch.tensor(all_mid_label)
        self.re_targets = torch.tensor(all_re_label)
        wide_deep_raw = torch.tensor(
            seq_data[:, [1, 9, 3, 5, 6]].astype(float))  # driver_id slice_window distance link_num cross_num
        self.deep_category = wide_deep_raw[:, :2].long()  # driver_id slice_window
        self.deep_real = wide_deep_raw[:, 2:].float()  # distance link_num cross_num
        self.wide_index = wide_deep_raw.clone()  # [256, 5]
        self.wide_index[:, 2:] = 0
        self.wide_index += torch.tensor(
            [0, drivers_num, drivers_num + 288, drivers_num + 288 + 1, drivers_num + 288 + 1 + 1])  # WDR-LC 5个
        self.wide_index = self.wide_index.long()
        self.wide_value = wide_deep_raw.float()
        self.wide_value[:, :2] = 1.0  # 类别特征的wide value 为1，只要其embedding后的值，连续特征的wide_value为连续值

    def __getitem__(self, index):
        return self.wide_index[index], self.wide_value[index], self.deep_category[index], self.deep_real[index], \
               self.all_real[index], self.all_real7[index], self.all_real4[index], self.all_real1[index], \
               self.all_flow[index], self.all_flow7[index], self.all_flow4[index], self.all_flow1[index], \
               self.all_linkdistance[index], self.all_linkdistance7[index], self.all_linkdistance4[index], \
               self.all_linkdistance1[index], \
               self.all_id[index], self.all_id7[index], self.all_id4[index], self.all_id1[index], \
               self.all_highway[index], self.all_highway7[index], self.all_highway4[index], self.all_highway1[index], \
               self.all_lane[index], self.all_lane7[index], self.all_lane4[index], self.all_lane1[index], \
               self.all_oneway[index], self.all_oneway7[index], self.all_oneway4[index], self.all_oneway1[index], \
               self.all_reversed[index], self.all_reversed7[index], self.all_reversed4[index], self.all_reversed1[
                   index], \
               self.all_num[index], self.all_mid_num[index], self.all_re_num[index], \
               self.targets[index], self.mid_targets[index], self.re_targets[index]

    def __len__(self):
        return self.targets.shape[0]


def picp(y_true, y_pred, upper_bound, lower_bound):
    """
    y_true,y_pred : B, N, T, D
    sigma : B, N, T, D
    """
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    upper_bound = torch.tensor(upper_bound)
    lower_bound = torch.tensor(lower_bound)
    result1 = torch.where(y_true < upper_bound, torch.ones_like(y_true), torch.zeros_like(y_pred.T[0].T))
    result2 = torch.where(y_true > lower_bound, torch.ones_like(y_true), torch.zeros_like(y_pred.T[0].T))

    recalibrate_rate = torch.sum(result1 * result2) / torch.prod(torch.tensor(y_true.shape))

    return recalibrate_rate * 100


def mpiw(y_true, upper_bound, lower_bound):
    """
    y_true,y_pred : B, N, T, D
    sigma : B, N, T, D
    """
    upper_bound = torch.tensor(upper_bound)
    lower_bound = torch.tensor(lower_bound)
    MPIW = upper_bound - lower_bound
    MPIW = torch.sum(MPIW) / torch.prod(torch.tensor(y_true.shape))

    return MPIW


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mape_(y_hat, y, length):
    length = length.cpu().numpy().tolist()
    weight = [1 if i > 40 else 1.2 for i in length]
    weight = torch.tensor(weight, device=device)
    l = torch.abs(y_hat - y) / y
    l *= weight
    return l.mean()


def SR(y_true, y_pred):
    MAPE = torch.tensor(abs(y_pred - y_true) / y_true)
    # print(type(MAPE),MAPE.shape)
    sr = torch.sum(MAPE <= 0.1) / torch.prod(torch.tensor(y_true.shape))
    return sr


def mis(y_true, upper_bound, lower_bound):
    """
    y_pred: T B V F
    y_true: T B V
    """

    # pho = 0.05 # 置信度 97.5 pho/2
    pho = 0.10
    #     loss0 = torch.abs(y_pred.T[2].T - y_true) # MAE
    loss1 = np.maximum(upper_bound - lower_bound, np.array([0.]))  # u-l
    loss2 = np.maximum(lower_bound - y_true, np.array([0.])) * 2 / pho  # l-y 哪些下界值超了
    loss3 = np.maximum(y_true - upper_bound, np.array([0.])) * 2 / pho  # y-u 哪些上界值小了
    #     print(loss1,loss2,loss3)
    loss = loss1 + loss2 + loss3
    return loss.mean()


def QICE(y_true, y_upper, y_low, y_pred):
    Q1 = ((y_low - y_true) > 0).astype(int)
    q21 = ((y_true - y_low) > 0).astype(int)
    q22 = ((y_pred - y_true) > 0).astype(int)
    Q2 = q21 * q22
    q31 = ((y_true - y_pred) > 0).astype(int)
    q32 = ((y_upper - y_true) > 0).astype(int)
    Q3 = q31 * q32
    Q4 = ((y_true - y_upper) > 0).astype(int)
    PQ1 = np.absolute(np.sum(Q1) / len(y_true) - 0.05)
    PQ2 = np.absolute(np.sum(Q2) / len(y_true) - 0.45)
    PQ3 = np.absolute(np.sum(Q3) / len(y_true) - 0.45)
    PQ4 = np.absolute(np.sum(Q4) / len(y_true) - 0.05)
    # print(PQ1, PQ2, PQ3, PQ4)
    qice = PQ1 + PQ2 + PQ3 + PQ4
    return qice


def UncertaintyPercentage(sigma, real):
    UP = sigma / real
    return UP.mean()


def quantile_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    # quantiles = [0.025, 0.5, 0.975]  # 0.5 均值
    quantiles = [0.10, 0.5, 0.90]  # 0.5 均值
    losses = []
    for i, q in enumerate(quantiles):
        # print(y_true.shape,y_pred.shape)
        errors = y_true - y_pred.T[i]
        errors = errors * mask
        errors[errors != errors] = 0
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(0))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=0), dim=0))
    return loss


def maemis_loss(y_pred, y_true):
    """
    y_pred: T B V F
    y_true: T B V
    """
    # print(y_pred.T.shape)
    # print(y_true.shape)
    # exit()
    mask = (y_true != 0).float()
    mask /= mask.mean()
    # pho = 0.05 # 置信度 97.5 pho/2
    pho = 0.10
    loss0 = torch.abs(y_pred.T[2].T - y_true)
    loss1 = torch.max(y_pred.T[0].T - y_pred.T[1].T, torch.tensor([0.]).to(y_true.device))

    loss2 = torch.max(y_pred.T[1].T - y_true, torch.tensor([0.]).to(y_true.device)) * 2 / pho
    loss3 = torch.max(y_true - y_pred.T[0].T, torch.tensor([0.]).to(y_true.device)) * 2 / pho
    loss = loss0 + loss1 + loss2 + loss3
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()


def nll_loss(pi, mu, sigma, linktime, mask):
    """
    计算基于序列的混合高斯分布的负对数似然损失。
    :param pi: 混合系数（权重），形状为 (batch_size, seq_len, num_gaussians)
    :param mu: 每个高斯分布的均值，形状为 (batch_size, seq_len, num_gaussians)
    :param sigma: 每个高斯分布的标准差，形状为 (batch_size, seq_len, num_gaussians)
    :param linktime: 目标值，形状为 (batch_size, seq_len)
    :return: 负对数似然损失
    """

    # 检查 mu 和 sigma 是否包含 NaN
    if torch.isnan(mu).any() or torch.isnan(sigma).any():
        print("mu\n", mu)
        print("sigma\n", sigma)
    # 创建高斯分布
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    # print("nLL", m.shape)
    # 计算混合模型的总概率密度
    linktime = linktime.unsqueeze(-1)
    # 计算每个组件的对数似然
    log_prob = m.log_prob(linktime.float())  # (batch_size, seq_len, num_gaussians)
    # 计算混合高斯的对数似然
    weighted_log_prob = torch.logsumexp(log_prob + torch.log(pi), dim=-1)  # (batch_size, seq_len) ???
    # 计算负对数似然损失
    weighted_log_prob = weighted_log_prob * mask
    nll = -torch.mean(weighted_log_prob)
    return nll


def TransE(head, tail, gamma, negative=False):
    if negative:
        head = head.unsqueeze(1)  # b,1,f
        # tail = tail.permute(0, 2, 1)
        score = head - tail  # b,6,F
        score = gamma - torch.norm(score, p=1, dim=2)
        # score = (F.softmax(score * 1.0, dim=1).detach() * F.logsigmoid(-score)).sum(dim=1)
    else:
        score = head - tail
        score = gamma - torch.norm(score, p=1, dim=1)
        # score = F.logsigmoid(score).squeeze()
    score = F.logsigmoid(score).squeeze()  # b
    loss = - score.mean()  # 1
    return loss


def RotatE(head, relation, tail, gamma, negative=False):
    pi = 3.14159265358979323846
    if negative:
        head = head.unsqueeze(2)  # b,1,f
        tail = tail.permute(0, 2, 1)
    re_head, im_head = torch.chunk(head, 2, dim=1)  # B,F/2
    re_tail, im_tail = torch.chunk(tail, 2, dim=1)

    # Make phases of relations uniformly distributed in [-pi, pi]

    phase_relation = relation / (head.shape[-1] / pi)

    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)
    if negative:
        re_relation = re_relation.unsqueeze(2)
        im_relation = im_relation.unsqueeze(2)
    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation
    re_score = re_score - re_tail
    im_score = im_score - im_tail

    score = torch.stack([re_score, im_score], dim=0)
    score = score.norm(dim=0)

    score = gamma.item() - score.sum(dim=1)

    score = F.logsigmoid(score).squeeze()
    loss = - score.mean()
    return loss


def model_test(model, data_loader, epoch, val_eval, FLAGS, device, model_name, output_path):
    model.eval()
    predicts = []
    mid_predicts = []
    data_dict = {
        7: [[], []],
        4: [[], []],
        1: [[], []],
    }
    re_mae = []
    re_mape = []
    re_sr = []
    re_picp = []
    re_mis = []
    re_mpiw = []
    loss = FLAGS.loss
    label = []
    mid_label = []
    start_time = time.time()
    with torch.no_grad():
        # tk0 = tqdm.tqdm(val_data_loader, smoothing=0, mininterval=1.0)
        # batch_size = FLAGS.batch_size
        if FLAGS.isdropout:
            enable_dropout(model)
        for i, data in enumerate(data_loader):
            wide_index, wide_value, deep_category, deep_real, \
            all_real, all_real7, all_real4, all_real1, \
            all_flow, all_flow7, all_flow4, all_flow1, \
            all_linkdistance, all_linkdistance7, all_linkdistance4, all_linkdistance1, \
            all_id, all_id7, all_id4, all_id1, \
            all_highway, all_highway7, all_highway4, all_highway1, \
            all_lane, all_lane7, all_lane4, all_lane1, \
            all_oneway, all_oneway7, all_oneway4, all_oneway1, \
            all_reversed, all_reversed7, all_reversed4, all_reversed1, \
            all_num, all_mid_num, all_re_num, \
            targets, mid_targets, re_targets = [d.to(device) for d in data]
            all_link_feature = torch.cat([all_id, all_highway, all_lane, all_reversed, all_oneway], dim=2).to(
                device)  # [B, F, 5]
            all_link_feature7 = torch.cat([all_id7, all_highway7, all_lane7, all_reversed7, all_oneway7], dim=2).to(
                device)  # [B, F, 5]
            all_link_feature4 = torch.cat([all_id4, all_highway4, all_lane4, all_reversed4, all_oneway4], dim=2).to(
                device)  # [B, F, 5]
            all_link_feature1 = torch.cat([all_id1, all_highway1, all_lane1, all_reversed1, all_oneway1], dim=2).to(
                device)  # [B, F, 5]
            zero_target = torch.zeros(wide_index.shape[0], 1).to(device)
            y, mid_y = model.net(wide_index, wide_value, deep_category, deep_real, all_link_feature, all_num, all_flow,
                                 all_linkdistance, all_real, zero_target, all_mid_num)
            predicts += y.tolist()
            label += targets.tolist()
            mid_predicts += mid_y.tolist()
            mid_label += mid_targets.tolist()
            for k in range(1, 4):
                if k == 1:
                    re_y, re_target = model.net(wide_index, wide_value, deep_category, deep_real, all_link_feature7, \
                                                all_re_num[:, k - 1], all_flow7, all_linkdistance7, all_real7,
                                                mid_targets[:, k - 1:k],
                                                re_target=re_targets[:, k - 1:k], all_mid_num=all_re_num[:, k - 1:k])
                    data_dict[7][0] += re_target.tolist()
                    data_dict[7][1] += re_y.tolist()
                    # re_predicts7 += re_y.tolist()
                    # re_label7 += re_target.tolist()
                elif k == 2:
                    re_y, re_target = model.net(wide_index, wide_value, deep_category, deep_real, all_link_feature4, \
                                                all_re_num[:, k - 1], all_flow4, all_linkdistance4, all_real4,
                                                mid_targets[:, k - 1:k],
                                                re_target=re_targets[:, k - 1:k], all_mid_num=all_re_num[:, k - 1:k])
                    data_dict[4][0] += re_target.tolist()
                    data_dict[4][1] += re_y.tolist()
                    # re_predicts4 += re_y.tolist()
                    # re_label4 += re_target.tolist()
                else:
                    re_y, re_target = model.net(wide_index, wide_value, deep_category, deep_real, all_link_feature1, \
                                                all_re_num[:, k - 1], all_flow1, all_linkdistance1, all_real1,
                                                mid_targets[:, k - 1:k],
                                                re_target=re_targets[:, k - 1:k], all_mid_num=all_re_num[:, k - 1:k])
                    data_dict[1][0] += re_target.tolist()
                    data_dict[1][1] += re_y.tolist()
                    # re_predicts1 += re_y.tolist()
                    # re_label1 += re_target.tolist()
    end_time = time.time()
    epoch_test_time = end_time - start_time
    print('test time: ' + str(epoch_test_time))
    predicts = np.array(predicts)
    label = np.array(label)
    mid_predicts = np.array(mid_predicts)
    mid_label = np.array(mid_label)
    # 转换列表为NumPy数组
    for key in data_dict:
        data_dict[key] = [np.array(data_dict[key][0]).squeeze(), np.array(data_dict[key][1])]
    # re_predicts7 = np.array(re_predicts7)
    # re_label7 = np.array(re_label7)
    # re_predicts4 = np.array(re_predicts4)
    # re_label4 = np.array(re_label4)
    # re_predicts1 = np.array(re_predicts1)
    # re_label1 = np.array(re_label1)
    mid_mae = []
    mid_mape = []
    mid_picp = []
    mid_mis = []
    mid_mpiw = []
    # mid_mse = []
    if loss == 'maemis':
        test_mape = mape(label, predicts[:, 2])
        test_sr = SR(label, predicts[:, 2])
        test_mse = mse(label, predicts[:, 2])
        test_mae = mae(label, predicts[:, 2])
        test_picp = picp(label, predicts, predicts.T[0].T, predicts.T[1].T)
        test_mis = mis(label, upper_bound=predicts.T[0].T, lower_bound=predicts.T[1].T)
        test_mpiw = mpiw(label, predicts.T[0].T, predicts.T[1].T)
        # test_qice = QICE(label, predicts.T[0].T, predicts.T[1].T, predicts.T[2].T)
    elif loss == 'quantile':
        test_mape = mape(label, predicts[:, 1])
        test_mse = mse(label, predicts[:, 1])
        test_mae = mae(label, predicts[:, 1])
        test_sr = SR(label, predicts[:, 1])
        test_picp = picp(label, predicts, predicts[:, 2], predicts[:, 0])
        test_mis = mis(label, upper_bound=predicts[:, 2], lower_bound=predicts[:, 0])
        test_mpiw = mpiw(label, predicts[:, 2], predicts[:, 0])
        # test_qice = QICE(label, predicts[:, 2], predicts[:, 0], predicts[:, 1])
        for i in range(mid_predicts.shape[1]):
            mid_mape.append(mape(mid_label[:, i], mid_predicts[:, i][:, 1]))
            mid_mae.append(mae(mid_label[:, i], mid_predicts[:, i][:, 1]))
            mid_picp.append(
                picp(mid_label[:, i], mid_predicts[:, i], mid_predicts[:, i][:, 2], mid_predicts[:, i][:, 0]))
            mid_mis.append(
                mis(mid_label[:, i], upper_bound=mid_predicts[:, i][:, 2], lower_bound=mid_predicts[:, i][:, 0]))
            mid_mpiw.append(mpiw(mid_label[:, i], mid_predicts[:, i][:, 2], mid_predicts[:, i][:, 0]))
        # 用一个循环处理所有操作
        # 定义结果字典
        # 创建一个空字典来存储结果
        results = {}
        # 循环处理每一组数据
        for key, (targets, predicts) in data_dict.items():
            re_mae.append(mae(targets, predicts[:, 1]))
            re_mape.append(mape(targets, predicts[:, 1]))
            re_sr.append(SR(targets, predicts[:, 1]))
            re_picp.append(picp(targets, predicts, predicts[:, 2], predicts[:, 0]))
            re_mis.append(mis(targets, upper_bound=predicts[:, 2], lower_bound=predicts[:, 0]))
            re_mpiw.append(mpiw(targets, predicts[:, 2], predicts[:, 0]))
    else:
        test_mape = mape(label, predicts)
        test_sr = SR(label, predicts)
        test_mse = mse(label, predicts)
        test_mae = mae(label, predicts)
        for i in range(mid_predicts.shape[1]):
            mid_mape.append(mape(mid_label[:, i], mid_predicts[:, i]))
            mid_mae.append(mae(mid_label[:, i], mid_predicts[:, i]))
    test_rmse = np.sqrt(test_mse)
    print("Test Result:")
    if loss != "mape" and loss != "MAE":
        print('MAPE:%.3f rmse: %.3f \tMSE:%.2f\tMAE:%.2f\tPICP:%.3f\tMIS:%.3f\tMPIW：%.3f\t' % (
            test_mape * 100, test_rmse, test_mse, test_mae, test_picp, test_mis, test_mpiw))

        for i in range(len(mid_mape)):
            print('%d MAPE:%.3f\tMAE:%.2f' % (i + 1, mid_mape[i] * 100, mid_mae[i]))
            print('%d PICP:%.3f\tMIS:%.3f\tMPIW：%.3f' % (i + 1, mid_picp[i], mid_mis[i], mid_mpiw[i]))

        for i in range(len(re_mape)):
            print('%d MAPE:%.3f\tMAE:%.3f\tSR:%.3f' % (i + 1, re_mape[i] * 100, re_mae[i], re_sr[i]))
            print('%d PICP:%.3f\tMIS:%.3f\tMPIW：%.3f' % (i + 1, re_picp[i], re_mis[i], re_mpiw[i]))
    else:
        print('Full MAPE:%.3f\tSR:%.3f\t rmse: %.3f\tMSE:%.2f\tMAE:%.2f' % (
            test_mape * 100, test_sr, test_rmse, test_mse, test_mae))
        for i in range(len(mid_mape)):
            print('%d MAPE:%.3f\tMAE:%.2f' % (i + 1, mid_mape[i] * 100, mid_mae[i]))

    # if (epoch + 1) % 10 == 0 or epoch == 0:
    #     label = np.expand_dims(np.array(label, dtype=object), axis=1)
    #     if loss == 'maemis':
    #         predicts = np.expand_dims(np.array(predicts[:, 2], dtype=object), axis=1)
    #     elif loss == 'quantile':
    #         predicts = np.expand_dims(np.array(predicts[:, 1], dtype=object), axis=1)
    #     else:
    #         predicts = np.expand_dims(np.array(predicts, dtype=object), axis=1)
    # result = np.concatenate((predicts, label), axis=1)
    # np.save(output_path + f'/{model_name}_epoch{epoch}', result)
    print('predict work done')


def process_test(batch_size, test_data, model, epoch, val_eval, FLAGS, device, model_name, output_path):
    # seq_data = np.load(seq_data_name, allow_pickle=True)
    # if is_test:
    # df = df[:500]
    # seq_data = seq_data[:500]
    dataset = GiscupDataset(test_data, FLAGS, device)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    model_test(model, data_loader, epoch, val_eval, FLAGS, device, model_name, output_path)


def enable_dropout(m):
    print("MCDropout enabled")
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()


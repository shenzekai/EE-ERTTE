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
from prefetch_generator import BackgroundGenerator
from torch.cuda.amp import autocast as autocast, GradScaler

# 构建Dataset
class TTEDataset(torch.utils.data.Dataset):
    def __init__(self, seq_data, FLAGS):
        mode = FLAGS.er_mode
        all_num = []
        all_mid_num = []
        all_re_num = []
        all_id = []
        all_id_re = []
        all_time = []
        all_time_re = []
        all_flow = []
        all_flow_re = []
        all_linkdistance = []
        all_linkdistance_re = []
        all_label = []
        all_mid_label = []
        all_re_label = []
        all_cross = []
        all_oneway = []
        all_oneway_re = []
        all_reversed = []
        all_reversed_re = []
        all_highway = []
        all_highway_re = []
        all_lane = []
        all_lane_re = []
        drivers_num = FLAGS.drivers_num
        print(seq_data.shape)

        for i in range(len(seq_data)):
            ids = seq_data[i][4]
            all_id.append(ids)  # link id

            # 提取link id
            id_re = seq_data[i][4][seq_data[i][25 + mode*2]:]   # traveled link id
            all_id_re.append(id_re)  # link id

            time = seq_data[i][11]
            time_re = seq_data[i][11][seq_data[i][25 + mode*2]:]
            all_time.append(time)  # list
            all_time_re.append(time_re)  # list

            flow = seq_data[i][12]
            flow_re = seq_data[i][12][seq_data[i][25+mode*2]:]
            all_flow.append(flow)  # list
            all_flow_re.append(flow_re)  # list

            linkdistance = seq_data[i][2]
            linkdistance_re = seq_data[i][2][seq_data[i][25+mode*2]:]
            all_linkdistance.append(linkdistance)  # list
            all_linkdistance_re.append(linkdistance_re)  # list

            highway = seq_data[i][15]
            highway_re = seq_data[i][15][seq_data[i][25+mode*2]:]
            all_highway.append(highway)  # list
            all_highway_re.append(highway_re)  # list

            lane = seq_data[i][16]
            lane_re = seq_data[i][16][seq_data[i][25+mode*2]:]
            all_lane.append(lane)  # list
            all_lane_re.append(lane_re)  # list

            oneway = seq_data[i][13]
            oneway_re = seq_data[i][13][seq_data[i][25+mode*2]:]
            all_oneway.append(oneway)  # list
            all_oneway_re.append(oneway_re)  # list

            reversed = seq_data[i][14]
            reversed_re = seq_data[i][14][seq_data[i][25+mode*2]:]
            all_reversed.append(reversed)  # list
            all_reversed_re.append(reversed_re)  # list

            all_num.append(seq_data[i][5])  # link num porto-1记住
            all_cross.append(seq_data[i][6])  # cross num
            all_mid_num.append(seq_data[i][25+mode*2])  # mid link num
            all_re_num.append(seq_data[i][26+mode*2])  # re link num
            all_label.append(seq_data[i][8])  # label
            all_mid_label.append(seq_data[i][17+mode*2])  # mid_label
            all_re_label.append(seq_data[i][18+mode*2])  # re_label
        self.all_num = all_num
        self.all_mid_num = all_mid_num
        self.all_re_num = all_re_num

        # link 平均通行时间
        self.all_real = all_time
        self.all_real_re = all_time_re

        # 流量特征
        self.all_flow = all_flow
        self.all_flow_re = all_flow_re

        # 行程中的路段长度
        self.all_linkdistance = all_linkdistance
        self.all_linkdistance_re = all_linkdistance_re

        # 行程中的路段ID序列 70% 40% 10%
        self.all_id = all_id
        self.all_id_re = all_id_re

        self.all_highway = all_highway
        self.all_highway_re = all_highway_re

        self.all_lane = all_lane
        self.all_lane_re = all_lane_re

        self.all_oneway = all_oneway
        self.all_oneway_re = all_oneway_re

        self.all_reversed = all_reversed
        self.all_reversed_re = all_reversed_re

        self.targets = all_label
        self.mid_targets = all_mid_label
        self.re_targets = all_re_label

        wide_deep_raw = seq_data[:, [1, 9, 10, 3, 5]] # driver_id weekday slice_window distance link_num
        self.deep_category = wide_deep_raw[:, :3] # driver_id slice_window
        self.deep_real = wide_deep_raw[:, 3:]  # distance link_num
        self.wide_index = wide_deep_raw.copy()  # [256, 5]
        self.wide_index[:, 3:] = 0
        # self.wide_index += [0, drivers_num, drivers_num + 7, drivers_num + 7 + 288, drivers_num + 7 + 288 + 1, drivers_num + 7 + 288 + 1 + 1]  # WDR-LC 6个
        self.wide_index += [0, drivers_num, drivers_num + 7, drivers_num + 7 + 288, drivers_num + 7 + 288 + 1]  # WDR 5个
        self.wide_index = self.wide_index
        self.wide_value = wide_deep_raw
        self.wide_value[:, :3] = 1.0  # 类别特征的wide value 为1，只要其embedding后的值，连续特征的wide_value为连续值

    def __getitem__(self, index):
        attr = {}
        attr["wide_index"] = self.wide_index[index]
        attr["wide_value"] = self.wide_value[index]
        attr["deep_category"] = self.deep_category[index]
        attr["deep_real"] = self.deep_real[index]
        attr["all_real"] = self.all_real[index]
        attr["all_real_re"] = self.all_real_re[index]
        attr["all_flow"] = self.all_flow[index]
        attr["all_flow_re"] = self.all_flow_re[index]
        attr["all_linkdistance"] = self.all_linkdistance[index]
        attr["all_linkdistance_re"] = self.all_linkdistance_re[index]
        attr["all_id"] = self.all_id[index]
        attr["all_id_re"] = self.all_id_re[index]
        attr["all_highway"] = self.all_highway[index]
        attr["all_highway_re"] = self.all_highway_re[index]
        attr["all_lane"] = self.all_lane[index]
        attr["all_lane_re"] = self.all_lane_re[index]
        attr["all_oneway"] = self.all_oneway[index]
        attr["all_oneway_re"] = self.all_oneway_re[index]
        attr["all_reversed"] = self.all_reversed[index]
        attr["all_reversed_re"] = self.all_reversed_re[index]
        attr["all_num"] = self.all_num[index]
        attr["all_mid_num"] = self.all_mid_num[index]
        attr["all_re_num"] = self.all_re_num[index]
        attr["targets"] = self.targets[index]
        attr["mid_targets"] = self.mid_targets[index]
        attr["re_targets"] = self.re_targets[index]
        return attr

    def __len__(self):
        return len(self.targets)


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.wide_index = data['wide_index']
        self.wide_value = data['wide_value']
        self.deep_category = data['deep_category']
        self.deep_real = data['deep_real']
        self.all_link_feature = data['all_link_feature']
        self.all_re_num = data['all_re_num']
        self.all_flow = data['all_flow']
        self.all_linkdistance = data['all_linkdistance']
        self.all_real = data['all_real']
        self.mid_target = data['mid_target']
        self.re_target = data['re_target']

    def __getitem__(self, index):
        attr = {}
        attr["wide_index"] = self.wide_index[index]
        attr["wide_value"] = self.wide_value[index]
        attr["deep_category"] = self.deep_category[index]
        attr["deep_real"] = self.deep_real[index]
        attr["all_link_feature"] = self.all_link_feature[index]
        attr["all_re_num"] = self.all_re_num[index]
        attr["all_flow"] = self.all_flow[index]
        attr["all_linkdistance"] = self.all_linkdistance[index]
        attr["all_real"] = self.all_real[index]
        attr["mid_target"] = self.mid_target[index]
        attr["re_target"] = self.re_target[index]
        return attr

    def __len__(self):
        return len(self.re_target)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Dataset_CSI300(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S',
                 data_path='CSI300.csv', target='label', window_size=20):
        """
        root_path: 数据路径
        flag: 'train' | 'val' | 'test'
        size: [seq_len, label_len, pred_len] (这里只用 size[0] 作为窗口长度)
        data_path: 原始CSV文件
        target: 标签列名
        window_size: 滑动窗口长度 (即 seq_len)
        """

        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.root_path = root_path
        self.flag = flag
        self.data_path = data_path
        self.target = target
        self.window_size = window_size if size is None else size[0]

        self.feature_columns = [
            'prev_close','open','high','low','adjfactor','turnover','volume',
            '5_day_avg','10_day_avg','20_day_avg',
            'daily_return','weekly_return','monthly_return'
        ]
        self.label_column = [self.target]

        self.__read_data__()

    def __read_data__(self):
        # 读取原始csv
        df = pd.read_csv(f"{self.root_path}/{self.data_path}", encoding='utf-8')
        df['dt'] = pd.to_datetime(df['dt'])

        # train / test 划分
        if self.flag == 'train' or self.flag == 'val':
            df = df[df['dt'] < '2024-01-01']
        else:
            df = df[df['dt'] >= '2024-01-01']

        # 分组：按股票代码
        grouped = df.groupby('kdcode')

        # 计算最长时间序列长度
        max_time_steps = grouped.size().max()
        num_stocks = len(grouped)
        num_features = len(self.feature_columns)

        # 初始化存储 (T, S, F)
        seq_array = np.zeros((max_time_steps, num_stocks, num_features))
        label_array = np.zeros((max_time_steps, num_stocks, 1))

        # 填充
        for i, (name, group) in enumerate(grouped):
            group = group.sort_values(by='dt')
            features = group[self.feature_columns].values
            labels = group[self.label_column].values

            length = len(features)
            if length < max_time_steps:
                features = np.vstack([features, np.zeros((max_time_steps - length, num_features))])
                labels = np.vstack([labels, np.zeros((max_time_steps - length, 1))])

            seq_array[:, i, :] = features
            label_array[:, i, :] = labels

        # 滑动窗口 (T, S, F) → (N, L, S, F)
        def create_sliding_windows(array, window_size):
            shape = (array.shape[0] - window_size + 1, window_size) + array.shape[1:]
            strides = (array.strides[0],) + array.strides
            return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

        seq_array_windows = create_sliding_windows(seq_array, self.window_size)   # (N, L, S, F)
        label_array_windows = create_sliding_windows(label_array, self.window_size)  # (N, L, S, 1)

        # --- train/val 划分 ---
        if self.flag == 'train':
            split = int(len(seq_array_windows) * 0.8)
            seq_array_windows = seq_array_windows[:split]
            label_array_windows = label_array_windows[:split]
        elif self.flag == 'val':
            split = int(len(seq_array_windows) * 0.8)
            seq_array_windows = seq_array_windows[split:]
            label_array_windows = label_array_windows[split:]
        # test 不切

        # 转 torch tensor 并 reshape
        self.seqs = torch.from_numpy(seq_array_windows).float().permute(2, 0, 1, 3)   # (S, N, L, F)
        self.labels = torch.from_numpy(label_array_windows).long().permute(2, 0, 1, 3).squeeze(-1)  # (S, N, L)

        # 拉平到 (S*N, L, F)
        self.seqs = self.seqs.flatten(0, 1)   # (S*N, L, F)
        self.labels = self.labels.flatten(0, 1)   # (S*N, L)

        # 标签修正
        self.labels[self.labels < 0] = 0

    def __getitem__(self, index):
        x = self.seqs[index]           # (L, F)
        y = self.labels[index]         # (L,)
        return x, y

    def __len__(self):
        return len(self.seqs)
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN
from layers.kan_layers import TaylorKANLayer, WaveKANLayer


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.d_model = configs.d_model
        self.drop = configs.dropout
        self.revin = configs.revin
        self.affine = configs.affine
        self.e_layers = 4

        if self.revin:
            self.revin_layer = RevIN(self.channels, affine=self.affine, subtract_last=False)

        self.gate = nn.Linear(self.seq_len, self.e_layers)
        self.experts = nn.ModuleList([
            TaylorKANLayer(self.seq_len, self.pred_len, order=3, addbias=True),
            TaylorKANLayer(self.seq_len, self.pred_len, order=3, addbias=True),
            WaveKANLayer(self.seq_len, self.pred_len, wavelet_type="mexican_hat"),
            WaveKANLayer(self.seq_len, self.pred_len, wavelet_type="mexican_hat"),
        ])

        self.dropout = nn.Dropout(self.drop)
        self.classifier = nn.Linear(self.channels, 2)  # 分类头

    def forward(self, x):
        B, L, N = x.shape
        var_x = self.revin_layer(x, 'norm') if self.revin else x
        var_x = self.dropout(var_x).transpose(1, 2).reshape(B * N, L)

        score = F.softmax(self.gate(var_x), dim=-1)  # (BxN, E)
        expert_outputs = torch.stack([self.experts[i](var_x) for i in range(self.e_layers)], dim=-1)  # (BxN, Lo, E)

        prediction = torch.einsum("BLE,BE->BL", expert_outputs, score)
        prediction = prediction.reshape(B, N, -1).permute(0, 2, 1)  # (B, L, N)

        prediction = self.revin_layer(prediction, 'denorm') if self.revin else prediction
        logits = self.classifier(prediction)  # (B, L, 2)
        return logits

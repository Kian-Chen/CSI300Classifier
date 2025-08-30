import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Encoder import EncoderLayer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.num_classes = getattr(configs, "num_classes", 2)

        self.input_embedding = nn.Linear(self.enc_in, self.d_model)
        self.encoder = EncoderLayer(configs)

        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward(self, x):
        # x: [B, L, C]
        x = self.input_embedding(x)  # [B, L, d_model]
        enc_output, _ = self.encoder(x)   # [B, L, d_model]

        logits = self.classifier(enc_output)  # [B, L, num_classes]
        return logits


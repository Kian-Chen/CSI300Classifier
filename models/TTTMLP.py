import torch.nn as nn
from layers.Encoder import EncoderLayer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.num_classes = getattr(configs, "num_classes", 2)

        self.input_embedding = nn.Sequential(
            nn.LayerNorm(self.enc_in),
            nn.Linear(self.enc_in, self.d_model)
        )
        self.encoder = EncoderLayer(configs)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, int(self.d_model / 2)),
            nn.ReLU(),
            nn.Linear(int(self.d_model / 2), self.num_classes)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        # x: [B, L, C]
        x = self.input_embedding(x)          # [B, L, d_model]
        enc_output, _ = self.encoder(x)      # [B, L, d_model]
        logits = self.classifier(enc_output) # [B, L, num_classes]
        return logits

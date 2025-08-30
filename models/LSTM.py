import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.input_dim = configs.enc_in
        self.hidden_dim = configs.d_model
        self.num_layers = getattr(configs, "num_layers", 2)
        self.dropout = configs.dropout
        self.num_classes = getattr(configs, "num_classes", 2)

        self.input_embedding = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, int(self.hidden_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_dim / 2), self.num_classes)
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
        x = self.input_embedding(x)       # [B, L, hidden_dim]
        lstm_out, _ = self.lstm(x)       # [B, L, hidden_dim]
        logits = self.classifier(lstm_out)  # [B, L, num_classes]
        return logits

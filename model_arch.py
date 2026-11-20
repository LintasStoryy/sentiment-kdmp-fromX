import torch
import torch.nn as nn
from transformers import AutoModel

class AttentionLayer(nn.Module):
    """Lapisan Attention untuk memberi bobot fokus pada token paling relevan."""
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_output, mask=None):
        # lstm_output shape: (batch_size, seq_len, hidden_dim)
        scores = self.attention(lstm_output).squeeze(-1)  # (batch_size, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch_size, hidden_dim)
        return context, weights

class IndoBERTBiLSTMAttention(nn.Module):
    """Arsitektur Hybrid: IndoBERT -> BiLSTM -> Attention -> Classifier (Softmax saat inferensi)."""
    def __init__(self, pretrained_path="tokenizer", num_classes=3, lstm_hidden=128, dropout_rate=0.3, freeze_bert=False):
        super(IndoBERTBiLSTMAttention, self).__init__()
        # Memuat backbone IndoBERT
        self.bert = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        bert_hidden_size = self.bert.config.hidden_size  # 768 untuk indobert-base
        self.bilstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attention = AttentionLayer(lstm_hidden * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_out.last_hidden_state  # (batch, seq_len, 768)
        
        lstm_out, _ = self.bilstm(sequence_output)    # (batch, seq_len, lstm_hidden*2)
        context, attn_weights = self.attention(lstm_out, mask=attention_mask)
        
        out = self.dropout(context)
        logits = self.fc(out)  # Softmax diterapkan via CrossEntropyLoss saat training / torch.softmax saat prediksi
        return logits
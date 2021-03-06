import torch
import torch.nn as nn
import torch.nn.functional as F


class SASREC(nn.Module):
    def __init__(self, num_item, hidden_dim, seq_len, dropout_rate, num_blocks, num_heads, device):
        super(SASREC, self).__init__()
        self.embedding_layer = Embedding_layer(
            num_item, hidden_dim, seq_len, dropout_rate, device)
        self.attention_blocks = nn.ModuleList(
            [Self_Attention_Block(hidden_dim, num_heads, dropout_rate) for i in range(num_blocks)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.device = device

    def model(self, x):
        # 1. Embedding layer -> output : [batch_size, seq_len, emb_dimension]
        out, pad_masking = self.embedding_layer(x.to(self.device))

        # 2. self attention blocks
        seq_len = out.shape[1]
        # attention_mask : [seq_len, seq_len]
        attention_mask = ~torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device))

        for self_attention_block in self.attention_blocks:
            out = self_attention_block(
                out, attention_mask, pad_masking.to(self.device))

        # 3. layer normalization
        out = self.norm(out)

        return out

    def forward(self, x, pos_seq, neg_seq):
        # out : [batch_size, seq_len, emb_dimension]
        out = self.model(x)

        # p/n item embedding : [batch_size, seq_len, emb_dimension]
        positive_item, _ = self.embedding_layer(pos_seq.to(self.device))
        negative_item, _ = self.embedding_layer(neg_seq.to(self.device))

        # Element-wise product & Summation
        # p/n logit : [batch_size, seq_len] <- dimension reduction
        positive_logit = (out * positive_item).sum(dim=-1)
        negative_logit = (out * negative_item).sum(dim=-1)

        return positive_logit, negative_logit

    def predict(self, x, idx):

        # 예측한 값한 값

        # out : [batch_size, seq_len, emb_dimension]
        out = self.model(x)

        # out : [batch_size, emb_dimension]
        out = out[:, -1, :]

        # item_embedding : [batch_size, seq_len, emb_dimension]
        item_embed = self.embedding_layer.item_embedding(idx)

        # out : [batch_size, emb_dimension, 1]
        out = out.unsqueeze(-1)

        # logits : Matmul ([batch_size, seq_len, emb_dimension], [batch_size, emb_dimension, 1]) -> [batch_size, seq_len, 1]
        logits = item_embed.matmul(out.to(self.device))

        # logits : [batch_size, seq_len, 1] -> [batch_size, seq_len]
        logits = logits.squeeze(-1)

        pred = F.softmax(logits, dim=1)

        return logits, pred


class Embedding_layer(nn.Module):
    def __init__(self, num_item, hidden_dim, seq_len, dropout_rate, device):
        super(Embedding_layer, self).__init__()
        self.item_embedding = nn.Embedding(
            num_item, hidden_dim, padding_idx=3)
        self.position_embedding = nn.Embedding(seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.seq_len = seq_len
        self.device = device

    def forward(self, x):
        # 1. item embedding & positional embedding
        # x : [batch_size, seq_len]
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # position_masking : [batch_size, seq_len]
        position_masking = torch.tile(
            torch.arange(0, self.seq_len), (batch_size, 1))

        # input_emb : [batch_size, seq_len, emb_dimension]
        input_emb = self.item_embedding(x.to(self.device))

        # position_emb : [batch_size, seq_len, emb_dimension]
        position_emb = self.position_embedding(
            position_masking.to(self.device))

        input_emb += position_emb[:, :seq_len]
        input_emb = self.dropout(input_emb)

        # 2. timeline pad masking
        #print("x : ", x)
        pad_masking = torch.BoolTensor(x.to('cpu') == 3).to(self.device)
        #print("pad_masking : ", pad_masking)
        input_emb *= ~pad_masking.unsqueeze(-1)

        #input_emb : [batch_size, seq_len, emb_dimension]
        return input_emb, pad_masking


class Self_Attention_Block(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_rate):
        super(Self_Attention_Block, self).__init__()
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.multi_head_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout_rate, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.point_wise_feed_forward = PointWiseFeedForward(
            hidden_dim, dropout_rate)

    def forward(self, x, attention_masking, pad_masking):

        # 1. Multi-Head Attention
        # Query : [batch_size, seq_len, emb_dimsion] query dimension 확인해볼것!
        Query = self.q_norm(x)

        # out : [batch_size, seq_len, emb_dimension]
        out, _ = self.multi_head_attention(
            Query, x, x, attn_mask=attention_masking)

        # 2. Residual connection
        out = out + Query

        # 3. layer normalization
        out = self.norm(out)

        # 4. Point-Wise Feed Forward + Residual connection
        out = self.point_wise_feed_forward(out)

        # 5. timeline pad masking
        out *= ~pad_masking.unsqueeze(-1)

        return out


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):

        # (transpose) X : [batch_size, seq_len, emb_dimension] -> [batch_size, emb_dimension, seq_len]
        out = torch.transpose(x, -1, -2)
        out = self.dropout(self.conv1(out))
        out = self.relu(out)
        out = self.dropout(self.conv2(out))

        # (transpose) out : [batch_size, emb_dimension, seq_len] -> [batch_size, seq_len, emb_dimension]
        out = torch.transpose(out, -1, -2)

        # Residual connection
        out = out + x

        return out

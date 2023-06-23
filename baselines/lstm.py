# import torch
# import torch.nn as nn

# class LSTMClassifier(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, vocab_size, output_dim, n_layers, bidirectional, dropout):
#         super().__init__()

#         # 词嵌入层
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)

#         # LSTM层
#         self.lstm = nn.LSTM(embedding_dim,
#                             hidden_dim,
#                             num_layers=n_layers,
#                             bidirectional=bidirectional,
#                             dropout=dropout,
#                             batch_first=True)

#         # 全连接层
#         self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

#         # Dropout层
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, text, text_lengths):
#         # text: [batch_size, seq_length]
#         embedded = self.embedding(text)
#         # embedded: [batch_size, seq_length, emb_dim]

#         # 压缩句子中的padding
#         packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first=True)

#         packed_output, (hidden, cell) = self.lstm(packed_embedded)

#         # 解压缩句子
#         output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

#         # 使用最后时刻的隐藏状态
#         if self.lstm.bidirectional:
#             hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
#         else:
#             hidden = self.dropout(hidden[-1, :, :])
#         # hidden: [batch_size, hid_dim * num_directions]

#         # 全连接层
#         return self.fc(hidden)



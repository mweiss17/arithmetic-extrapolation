import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, vocab_size, device, bias=True, use_embedding=False,
                 set_linear_bias=False, linear_normal=False, linear_uniform=False, use_ln=False, ln_preact=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.use_embedding = use_embedding
        self.linear_normal = linear_normal
        self.linear_uniform = linear_uniform
        self.set_linear_bias = set_linear_bias
        self.device = device
        self.target_pad_idx = 3
        self.bias = bias
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.target_pad_idx)

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.linear = nn.Linear(hidden_size, self.vocab_size)
        if self.set_linear_bias:
            self.linear._parameters['bias'] = Variable(torch.tensor([.48, .48, .04, 0]), requires_grad=True)
        if self.linear_uniform:
            torch.nn.init.xavier_uniform_(self.linear.weight)
        elif self.linear_normal:
            torch.nn.init.xavier_normal_(self.linear._parameters['weight'], gain=1.0)

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=1,
            padding_idx=3
        )

    def init_hidden(self):
        h_0 = torch.zeros((1, self.batch_size, self.hidden_size))
        c_0 = torch.zeros((1, self.batch_size, self.hidden_size))
        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)
        h_0 = Variable(h_0)
        c_0 = Variable(c_0)
        return (h_0, c_0)

    def forward_save(self, x):
        self.hidden = self.init_hidden()
        batch_size, seq_len = x.size()
        x = x.unsqueeze(2).float()

        outs = []
        cs = []
        hs = []

        for cidx in range(x.size(1)):
            # [input_gate | forget_gate | b_gg | output_gate]

            input = x[:, cidx].unsqueeze(1)
            out, new_hidden = self.lstm(input, self.hidden)
            if cidx == x.size(1)-1:
                new_out, new_hidden = self.lstm(input, self.hidden)
                self.linear(new_out)

            self.hidden = new_hidden
            hs.append(self.hidden[0][0][0].detach().numpy())
            cs.append(self.hidden[1][0][0].detach().numpy())
            outs.append(out)

        x = torch.stack(outs)
        x = x.permute(1, 0, 3, 2)

        # (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)
        x = x.contiguous()
        x = x.view(-1, x.shape[2])

        # run through linear layer
        x = self.linear(x)

        # reshape to: (batch_size, seq_len, vocab_size)
        x = x.view(batch_size, seq_len, self.vocab_size)
        return x, cs, hs

    def forward(self, x, x_lens):
        self.hidden = self.init_hidden()
        batch_size, seq_len = x.size()
        if self.use_embedding:
            x = self.embedding(x)
        x = x.unsqueeze(2).float()

        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, enforce_sorted=False, batch_first=True)
        x, self.hidden = self.lstm(x_packed, self.hidden)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)
        x = x.contiguous()
        x = x.view(-1, x.shape[2])

        # run through linear layer
        x = self.linear(x)

        # reshape to: (batch_size, seq_len, vocab_size)
        x = x.view(batch_size, seq_len, self.vocab_size)
        return x

    def compute_loss(self, preds, targets):
        """before we calculate the loss, mask out activations for padded elements."""
        targets = targets.view(-1, self.vocab_size)
        preds = preds.view(-1, self.vocab_size)

        # compute cross entropy loss which ignores all <PAD> tokens
        loss = self.criterion(preds, targets.argmax(dim=1))
        return loss



class LayerNormLSTMCell(nn.LSTMCell):

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias)
        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)

    def forward(self, input, hidden=None):
        hx, cx = hidden
        gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
        i, f, o = gates[:, :(3 * self.hidden_size)].sigmoid().chunk(3, 1)
        g = gates[:, (3 * self.hidden_size):].tanh()

        cy = (f * cx) + (i * g)
        hy = o * self.ln_ho(cy).tanh()
        return hy, cy


class LayerNormLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, batch_size, vocab_size=None, num_layers=1, bias=True, use_embedding=False, linear_normal=False):
        super().__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.target_pad_idx = 3
        self.vocab_size = vocab_size
        self.linear_normal = linear_normal
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.target_pad_idx)
        self.linear = nn.Linear(hidden_size, self.vocab_size)
        self.use_embedding = use_embedding

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.input_size,
            padding_idx=3
        )


        if self.linear_normal:
            torch.nn.init.xavier_normal_(self.linear._parameters['weight'], gain=1.0)
        self.hiddencell = LayerNormLSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias)


    def init_hidden(self):
        h_0 = torch.zeros((self.batch_size, self.hidden_size))
        c_0 = torch.zeros((self.batch_size, self.hidden_size))
        h_0 = Variable(h_0)
        c_0 = Variable(c_0)
        return (h_0, c_0)

    def compute_loss(self, preds, targets, l1_lambda=0.001):
        targets = targets.view(-1, self.vocab_size)
        preds = preds.view(-1, self.vocab_size)
        loss = self.criterion(preds, targets.argmax(dim=1))
        # l1_norm = torch.sum(torch.tensor([torch.sum(torch.abs(p)) for p in self.parameters()])) / self.hidden_size
        # loss = loss + l1_lambda * l1_norm

        return loss

    def forward(self, input):
        if self.use_embedding:
            input = self.embedding(input.long())
            input = input.squeeze(2)

        h0, c0 = self.init_hidden()
        if len(input.shape) == 2:
            input = input.unsqueeze(-1)

        input = input.transpose(0, 1)
        seq_len, batch_size, hidden_size = input.size()  # supports TxNxH only

        ht = [None,] * seq_len
        ct = [None,] * seq_len

        for t, x in enumerate(input):
            if t == 0:
                ht[t], ct[t] = self.hiddencell(x, (h0, c0))
            else:
                ht[t], ct[t] = self.hiddencell(x, (ht[t-1], ct[t-1]))

        y = torch.stack(ht, dim=0)

        # run through linear layer
        y = self.linear(y.transpose(0, 1))

        # reshape to: (batch_size, seq_len, vocab_size)

        y = y.view(batch_size, seq_len, self.vocab_size)
        return y

    def forward_save(self, input):
        if self.use_embedding:
            input = self.embedding(input.long())
            input = input.squeeze(2)

        h0, c0 = self.init_hidden()
        if len(input.shape) == 2:
            input = input.unsqueeze(-1)
        input = input.transpose(0, 1)
        seq_len, batch_size, hidden_size = input.size()  # supports TxNxH only

        ht = [None,] * seq_len
        ct = [None,] * seq_len

        for t, x in enumerate(input):
            if t == 0:
                ht[t], ct[t] = self.hiddencell(x, (h0, c0))
            else:
                ht[t], ct[t] = self.hiddencell(x, (ht[t-1], ct[t-1]))
        hs = torch.stack(ht, dim=0).detach().numpy()
        cs = torch.stack(ct, dim=0).detach().numpy()
        y = torch.stack(ht, dim=0)

        # run through linear layer
        y = self.linear(y.transpose(0, 1))

        # reshape to: (batch_size, seq_len, vocab_size)

        y = y.view(batch_size, seq_len, self.vocab_size)
        return y, cs, hs


class Transformer(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_feedforward: int,
                 nlayers: int, dropout: float = 0.5, max_len: int = 5, **kwargs):
        super().__init__()
        self.model_type = 'TransformerDecoder'
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)

        transformer_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_feedforward, dropout=dropout,
                                                     **kwargs)
        transformer_norm = LayerNorm(d_model, eps=1e-5, **kwargs)
        self.transformer_block = TransformerEncoder(transformer_layers, nlayers, transformer_norm)

        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)
        self.max_len = max_len
        self.criterion = nn.CrossEntropyLoss()

        self._reset_parameters()
        self.init_weights()
        self.input_attn_mask = self.generate_square_subsequent_mask(self.max_len)

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.transformer_block.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_block(src, self.input_attn_mask)

        output = self.decoder(output)
        pred = output[-1]  # torch.softmax(output[-1], 1)
        return pred

    def loss_fn(self, pred, target):
        loss = self.criterion(pred, target)
        return loss


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        #         pe[:,0,:] = position / (max_len-1)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
#
#
# class Transformer(nn.Module):
#     def __init__(self, vocab_size=4, input_size=1, use_embedding=False, nhead=2, num_encoder_layers=2, num_decoder_layers=2, hidden_size=128, dropout=0.1, activation='relu', norm_first=True, linear_normal=True):
#         super().__init__()
#         self.target_pad_idx = 3
#         self.vocab_size = vocab_size
#         self.input_size = input_size
#         self.use_embedding = use_embedding
#         self.model = torch.nn.Transformer(batch_first=True,
#                                           nhead=nhead,
#                                           num_encoder_layers=num_encoder_layers,
#                                           num_decoder_layers=num_decoder_layers,
#                                           d_model=input_size,
#                                           dim_feedforward=input_size,
#                                           dropout=dropout,
#                                           activation=activation,
#                                           norm_first=norm_first)
#         self.linear_normal = linear_normal
#         self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.target_pad_idx)
#         self.linear = nn.Linear(hidden_size, self.vocab_size)
#
#         self.embedding = nn.Embedding(
#             num_embeddings=self.vocab_size,
#             embedding_dim=self.input_size,
#             padding_idx=self.target_pad_idx
#         )
#
#         if self.linear_normal:
#             torch.nn.init.xavier_normal_(self.linear._parameters['weight'], gain=1.0)
#
#
#     def forward(self, x, x_lens):
#         if self.use_embedding:
#             x = self.embedding(x.long())
#             x = x.squeeze(2)
#
#         # if len(input.shape) == 2:
#         #     input = input.unsqueeze(-1)
#         # input = input.transpose(0, 1)
#         breakpoint()
#
#         # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)
#         # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
#         x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, enforce_sorted=False, batch_first=True)
#         x, self.hidden = self.model(x, x)
#         x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
#
#         # (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)
#         x = x.contiguous()
#         x = x.view(-1, x.shape[2])
#
#         # run through linear layer
#         x = self.linear(x)
#
#         # reshape to: (batch_size, seq_len, vocab_size)
#         x = x.view(batch_size, seq_len, self.vocab_size)
#         return x

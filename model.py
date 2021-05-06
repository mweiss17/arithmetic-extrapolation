import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, vocab_size, device, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.device = device
        self.target_pad_idx = 0
        self.bias = bias
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.target_pad_idx)

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # [input_gate | forget_gate | b_gg | output_gate]
        # for names in self.lstm._all_weights:
        #     for name in filter(lambda n: "bias" in n, names):
        #         bias = getattr(self.lstm, name)
        #         n = bias.size(0)
        #         l = n // 4
        #         # set input gate bias
        #         bias.data[0:l].fill_(-.5)
        #         # set forget gate bias
        #         bias.data[l:l*2].fill_(1.)
        #         # set output gate bias
        #         bias.data[l*3:l*4].fill_(-1.)

        self.linear = nn.Linear(hidden_size, self.vocab_size)
        self.linear._parameters['bias'] = Variable(torch.tensor([0, .48, .48, .4]), requires_grad=True)
        torch.nn.init.xavier_normal_(self.linear._parameters['weight'], gain=1.0)

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=1,
            padding_idx=0
        )

    def init_hidden(self):
        h_0 = torch.zeros((1, self.batch_size, self.hidden_size))
        c_0 = torch.zeros((1, self.batch_size, self.hidden_size))
        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)
        h_0 = Variable(h_0)
        c_0 = Variable(c_0)
        return (h_0, c_0)

    def apply_quantize(self):
        closest = lambda x: int(ceil(log(x) / log(2)))
        torch.clip(self.linear.weight, )
        import pdb; pdb.set_trace()
        print("asdf")

    def forward_save_hiddens(self, x, x_lens):
        self.hidden = self.init_hidden()
        batch_size, seq_len = x.size()

        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)

        ### If we want embedding, then do this
        # x = self.embedding(x)
        ### If we don't want embedding, do this
        x = x.unsqueeze(2).float()

        outs = []
        hiddens = []
        cells = []
        for cidx in range(x.size(1)):
            out, self.hidden = self.lstm(x[:, cidx].unsqueeze(1), self.hidden)
            hiddens.append(self.hidden[0].detach().numpy())
            cells.append(self.hidden[1].detach().numpy())
            outs.append(out)

        hiddens = np.array(hiddens).squeeze(1).squeeze(1)
        cells = np.array(cells).squeeze(1).squeeze(1)
        x = torch.stack(outs)
        x = x.permute(1, 0, 3, 2)

        # (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)
        x = x.contiguous()
        x = x.view(-1, x.shape[2])

        # run through linear layer
        x = self.linear(x)

        # reshape to: (batch_size, seq_len, vocab_size)
        x = x.view(batch_size, seq_len, self.vocab_size)
        return x, hiddens, cells

    def forward(self, x, x_lens, record_activations=False):
        self.hidden = self.init_hidden()
        # self.apply_quantize()
        batch_size, seq_len = x.size()

        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)

        ### If we want embedding, then do this
        # x = self.embedding(x)
        ### If we don't want embedding, do this
        x = x.unsqueeze(2).float()
        if record_activations:
            outs = []
            cs = []
            hs = []
            for cidx in range(x.size(1)):
                # [input_gate | forget_gate | b_gg | output_gate]

                weight_ih_l0 = self.lstm._parameters['weight_ih_l0'] # (4 * hidden_size, input_size)
                weight_hh_l0 = self.lstm._parameters['weight_hh_l0'] # (4 * hidden_size, hidden_size)
                l = self.hidden_size
                input_gate_ih = weight_ih_l0[0:l]
                forget_gate_ih = weight_ih_l0[l:l*2]
                gg_ih = weight_ih_l0[l*2:l*3]
                output_gate_ih = weight_ih_l0[l*3:l*4]
                input_gate_hh = weight_hh_l0[0:l]
                forget_gate_hh = weight_hh_l0[l:l*2]
                gg_hh = weight_hh_l0[l*2:l*3]
                output_gate_hh = weight_hh_l0[l*3:l*4]

                input = x[:, cidx].unsqueeze(1)
                out, new_hidden = self.lstm(input, self.hidden)
                if cidx == x.size(1)-1:
                    new_out, new_hidden = self.lstm(input, self.hidden)
                    self.linear(new_out)

                self.hidden = new_hidden
                hs.append(self.hidden[0].detach().numpy())
                cs.append(self.hidden[1].detach().numpy())


                if len(cs) > 2:
                    this_one = cs[-1][0][0][0]
                    last_one = cs[-2][0][0][0]
                    print(this_one - last_one)
                    # import pdb;
                    # pdb.set_trace()
                outs.append(out)

            x = torch.stack(outs)
            x = x.permute(1, 0, 3, 2)

        else:
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

    def compute_loss(self, preds, targets, x_lens):
        """before we calculate the loss, mask out activations for padded elements."""
        targets = targets.view(-1, self.vocab_size)
        preds = preds.view(-1, self.vocab_size)

        # create a mask by filtering out all tokens that ARE NOT the padding token

        mask = (targets[:, self.target_pad_idx] == 0.)
        masked_targets = targets[mask, :].argmax(dim=1).long()
        masked_preds = preds[mask, :]

        # compute cross entropy loss which ignores all <PAD> tokens
        loss = self.criterion(masked_preds, masked_targets)
        return loss


    def pad_collate(batch):
        (xx, yy) = zip(*batch)
        x_lens = [len(x) for x in xx]
        y_lens = [len(y) for y in yy]

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
        for idx, y_len in enumerate(y_lens):
            target_pad_idx = 0
            yy_pad[idx, y_len:max(y_lens), target_pad_idx] = 1
        return xx_pad, yy_pad, x_lens, y_lens

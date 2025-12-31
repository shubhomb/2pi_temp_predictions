import torch.nn as nn
import torch
from torch.utils.data import Dataset

class RNNModel(nn.Module):
    def __init__(self, input_size: int, units: int, output_size: int):
        super(RNNModel, self).__init__()
        self.initial_state_projection = nn.Linear(output_size, units)  # activity_initial -> hidden TODO: modify
        self.rnn = nn.GRU(input_size, units, batch_first=True,
                          dropout=0.2,
                        bidirectional=False,


                          )  # returns (batch, seq, hidden)
        self.dense = nn.Linear(units, output_size)              # maps hidden -> ROI outputs

    def forward(self, inp):
        # inp is a tuple: (inputs, activity_initial)
        # inputs: Tensor shape (batch, seq_len, input_size)
        # activity_initial: Tensor shape (batch, output_size)
        inputs, activity_initial = inp

        # Ensure tensors and device alignment
        device = next(self.parameters()).device
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
        if not torch.is_tensor(activity_initial):
            activity_initial = torch.tensor(activity_initial, dtype=torch.float32, device=device)

        # project initial activity to GRU hidden state shape (1, batch, units)
        h0 = self.initial_state_projection(activity_initial)    # (batch, units)
        h0 = h0.unsqueeze(0).contiguous()                       # (1, batch, units)

        # GRU forward
        out, _ = self.rnn(inputs, h0)                           # out: (batch, seq_len, units)

        # per-timestep hidden to output ROIs
        return self.dense(out)                                  # (batch, seq_len, output_size)


# --- Dataset that returns ((inputs, activity_initial), target) ---
class SeqDataset(Dataset):
    def __init__(self, activity_init_conds, stims, activity):
        self.initial_conds = torch.tensor(activity_init_conds, dtype=torch.float32)
        self.X = torch.tensor(stims, dtype=torch.float32)
        self.Y = torch.tensor(activity, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        inp = self.X[idx]                     # (seq_len, input_size)
        target = self.Y[idx]                 # (seq_len, output_size)
        activity_initial = self.initial_conds[idx]
        return (inp, activity_initial), target


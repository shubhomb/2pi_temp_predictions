import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset

class RNNModel(nn.Module):
    def __init__(self, input_size: int, units: int, output_size: int, num_layers: int = 2):
        super(RNNModel, self).__init__()
        self.num_layers = num_layers
        self.units = units
        self.initial_state_projection = nn.Linear(output_size, units)  # activity_initial -> hidden for layer 0
        self.rnn = nn.GRU(input_size, units, 
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=0.2 if num_layers > 1 else 0.0,
                          bidirectional=False,
                          )  # returns (batch, seq, hidden)
        self.dense = nn.Sequential(
            nn.Linear(units, 2 * units),
            nn.GELU(),
            nn.Linear(2 * units, output_size)
        )  # maps hidden -> ROI outputs with nonlinearity

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

        batch_size = inputs.size(0)
        
        # Build h0 for all layers: (num_layers, batch, units)
        # Layer 0: project from initial activity
        # Layers 1+: initialize to zeros
        h0_layer0 = self.initial_state_projection(activity_initial)  # (batch, units)
        
        if self.num_layers == 1:
            h0 = h0_layer0.unsqueeze(0)  # (1, batch, units)
        else:
            # Initialize remaining layers with zeros
            h0_remaining = torch.zeros(self.num_layers - 1, batch_size, self.units, 
                                       device=device, dtype=h0_layer0.dtype)
            h0 = torch.cat([h0_layer0.unsqueeze(0), h0_remaining], dim=0)  # (num_layers, batch, units)

        # GRU forward
        out, _ = self.rnn(inputs, h0)                           # out: (batch, seq_len, units)

        # per-timestep hidden to output ROIs
        return self.dense(out)                                  # (batch, seq_len, output_size)


# --- Dataset that returns ((inputs, activity_initial), target) ---
class SeqDataset(Dataset):
    """
    Dataset for RNN training that returns ((stim_inputs, activity_initial), activity_target).
    
    Can be initialized from arrays or from a DataFrame with snippet columns.
    """
    def __init__(self, activity_init_conds=None, stims=None, activity=None, df=None):
        """
        Initialize from arrays OR from a DataFrame.
        
        Parameters
        ----------
        activity_init_conds : ndarray, optional
            Initial conditions, shape (n_samples, n_rois)
        stims : ndarray, optional
            Stimulation snippets, shape (n_samples, seq_len, n_electrodes)
        activity : ndarray, optional
            Activity snippets, shape (n_samples, seq_len, n_rois)
        df : pd.DataFrame, optional
            DataFrame with columns: initial_condition, stim_snippet, activity_snippet, valid
            If provided, arrays are extracted from the DataFrame.
        """
        if df is not None:
            # Initialize from DataFrame
            valid_df = df[df['valid']].copy()
            self.initial_conds = torch.tensor(
                np.stack(valid_df['initial_condition'].values), dtype=torch.float32
            )
            self.X = torch.tensor(
                np.stack(valid_df['stim_snippet'].values), dtype=torch.float32
            )
            self.Y = torch.tensor(
                np.stack(valid_df['activity_snippet'].values), dtype=torch.float32
            )
            # Store metadata for reference
            self.metadata = valid_df.drop(
                columns=['initial_condition', 'stim_snippet', 'activity_snippet', 'valid']
            ).reset_index(drop=True)
        else:
            # Initialize from arrays
            self.initial_conds = torch.tensor(activity_init_conds, dtype=torch.float32)
            self.X = torch.tensor(stims, dtype=torch.float32)
            self.Y = torch.tensor(activity, dtype=torch.float32)
            self.metadata = None
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        inp = self.X[idx]                     # (seq_len, input_size)
        target = self.Y[idx]                 # (seq_len, output_size)
        activity_initial = self.initial_conds[idx]
        return (inp, activity_initial), target
    
    def get_metadata(self, idx):
        """Get metadata for a specific sample (session, trial, config, etc.)"""
        if self.metadata is not None:
            return self.metadata.iloc[idx].to_dict()
        return None


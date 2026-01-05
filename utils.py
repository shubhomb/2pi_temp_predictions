import os
import sys
import datetime
import logging


import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm, colors
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
import scipy
import scipy.io





def plot_stim_ticks(stim_times, session_ids=None, tick_width=3,
                    cmap_name='Reds', figsize=(15, 5),
                    out_path=None, invert_y=True):
    """
    stim_times: ndarray, shape (n_sessions, n_times, n_electrodes)
    session_ids: list of session indices to plot (default: all)
    """

    n_sessions, n_times, n_electrodes = stim_times.shape

    if session_ids is None:
        session_ids = list(range(n_sessions))

    cmap = cm.get_cmap(cmap_name)

    # global amplitude range (ignore zeros)
    global_max = np.max(stim_times)
    print(f'Global max amplitude across sessions: {global_max}')
    print("Unique current values:",
          np.unique(stim_times[stim_times > 0]))

    norm = colors.Normalize(
        vmin=0.5,
        vmax=max(1.0, float(global_max))
    )

    f, axes = plt.subplots(
        1, len(session_ids),
        figsize=figsize,
        squeeze=False
    )

    for ax, sid in zip(axes[0], session_ids):
        data = stim_times[sid]  # (n_times, n_electrodes)

        for elec in range(n_electrodes):
            stim_idx = np.nonzero(data[:, elec])[0]
            if stim_idx.size == 0:
                continue

            amps = data[stim_idx, elec]
            for t, a in zip(stim_idx, amps):
                ax.vlines(
                    x=t,
                    ymin=elec - 0.5,
                    ymax=elec + 0.5,
                    color=cmap(norm(a)),
                    linewidth=1,
                    alpha=0.9
                )

        ax.set_xlim(-0.5, n_times - 0.5)
        ax.set_ylim(-0.5, n_electrodes - 0.5)
        if invert_y:
            ax.invert_yaxis()

        ax.set_yticks(np.arange(n_electrodes))
        ax.set_yticklabels(np.arange(1, n_electrodes + 1))
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Electrode')

        n_stims = int((data > 0).sum())
        ax.set_title(f'Session {sid + 1}, n_stims={n_stims}')

    # shared colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    f.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        label='Current Amplitude',
        fraction=0.02,
        pad=0.02
    )

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight')

    plt.show()

def plot_activity_per_session(activity, session_ids=None,
                              figsize=(15, 5), out_path=None):
    """
    activity: ndarray, shape (n_sessions, n_times, n_rois)
    """

    n_sessions, n_times, n_rois = activity.shape

    if session_ids is None:
        session_ids = list(range(n_sessions))

    f, axes = plt.subplots(
        1, len(session_ids),
        figsize=figsize,
        squeeze=False
    )

    for ax, sid in zip(axes[0], session_ids):
        data = activity[sid]  # (n_times, n_rois)

        im = ax.imshow(
            data.T,
            aspect='auto',
            cmap='Blues',
            origin='lower'
        )

        ax.set_title(f'Session {sid + 1}')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('ROI')

        ax.set_yticks(
            np.arange(0, n_rois, 10),
            labels=np.arange(1, n_rois + 1, 10)
        )

        ax.grid(False)
        f.colorbar(im, ax=ax, label='Activity (dF/F)')

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight')

    plt.show()

def make_snippets(activity, stim_times_sess, length, overlap=True, aligned=False):
    '''
    Create snippets of activity and stimulation data for RNN training.
    Each snippet is of size (length, n_rois) for activity and
    (length, n_electrodes) for stimulation.
    :param activity:
    :param stim_times_sess:
    :param length:
    :param overlap:
    :param aligned:
    :return:
    '''
    initial_conditions = []
    activity_snippets = []
    stim_snippets = []
    if aligned:
        raise NotImplementedError("Not yet implemented!")
    for session_id in range(3):
        activity_from_session = activity[session_id, ...]
        stimulation_from_session = stim_times_sess[session_id, ...]
        # overlapping snippets of size snippet_length
        if overlap:
            num_snippets = activity_from_session.shape[0] - length + 1
            for i in range(num_snippets):
                if i == 0: # TODO: How to deal with initial condition at start of series?
                    initial_conditions.append(activity_from_session[0])
                else:
                    initial_conditions.append(activity_from_session[i-1])
                activity_snippets.append(activity_from_session[i:i + length])
                stim_snippets.append(stimulation_from_session[i:i + length])
        else:
            for i in range(0, activity_from_session.shape[0] - length, length + 1):
                if i == 0:
                    initial_conditions.append(activity_from_session[0])
                else:
                    initial_conditions.append(activity_from_session[i-1])
                activity_snippets.append(activity_from_session[i:i + length])
                stim_snippets.append(stimulation_from_session[i:i + length])

    initial_conditions = np.array(initial_conditions)
    activity_snippets = np.array(activity_snippets)
    stim_snippets = np.array(stim_snippets)
    return initial_conditions, activity_snippets, stim_snippets


def make_snippets_df(trials_df, activity, stim_times_sess, length, overlap=True, stride=1):
    """
    Create snippets of activity and stimulation data with full metadata.
    
    Two modes:
    1. overlap=True (default): Creates overlapping snippets from the entire timeseries.
       Labels each snippet based on the FIRST stimulation within the snippet window.
       Also tracks ALL stimulations in the snippet for holdout filtering.
    2. overlap=False: Only creates snippets aligned to trial stim times (one per trial).
    
    Parameters
    ----------
    trials_df : pd.DataFrame
        DataFrame with columns: session, trial, config, electrode, current, stim_time, is_stim
    activity : ndarray, shape (n_sessions, n_times, n_rois)
        Activity data (dfof) for each session
    stim_times_sess : ndarray, shape (n_sessions, n_times, n_electrodes)
        Stimulation time series for each session
    length : int
        Snippet length in frames
    overlap : bool
        If True, create overlapping snippets with given stride. If False, only stim-aligned.
    stride : int
        Step size between snippet starts when overlap=True (default=1 for full overlap)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - session, snippet_start: location info
        - first_config, first_electrode, first_current, first_trial: metadata for FIRST stim
        - all_configs, all_electrodes, all_currents, all_trials: lists of ALL stims in snippet
        - has_stim: whether snippet contains any stimulation
        - initial_condition, activity_snippet, stim_snippet: data arrays
        - valid: bool
    """
    import pandas as pd
    
    # Build a lookup from (session, stim_time) -> trial metadata
    stim_lookup = {}
    for _, row in trials_df.iterrows():
        key = (row['session'], row['stim_time'])
        stim_lookup[key] = {
            'config': row['config'],
            'electrode': row['electrode'],
            'current': row['current'],
            'trial': row['trial'],
            'is_stim': row['is_stim']
        }
    
    results = []
    
    if overlap:
        # Create overlapping snippets from entire timeseries
        for session in range(activity.shape[0]):
            n_times = activity[session].shape[0]
            
            for i in range(0, n_times - length + 1, stride):
                # Get initial condition
                if i == 0:
                    initial_cond = activity[session][0, :]
                else:
                    initial_cond = activity[session][i - 1, :]
                
                # Get snippets
                activity_snippet = activity[session][i:i + length, :]
                stim_snippet = stim_times_sess[session][i:i + length, :]
                
                # Find ALL stimulations within this snippet
                all_configs = []
                all_electrodes = []
                all_currents = []
                all_trials = []
                all_stim_times_in_snippet = []
                
                for t_rel in range(length):
                    t_abs = i + t_rel
                    # Check if there's a stim at this timepoint
                    stim_at_t = stim_snippet[t_rel, :]
                    electrodes_active = np.where(stim_at_t > 0)[0]
                    
                    for e in electrodes_active:
                        curr = int(stim_at_t[e])
                        cfg = e * 3 + (curr - 3) + 1
                        
                        # Try to get trial info from lookup
                        key = (session, t_abs)
                        if key in stim_lookup:
                            trial = stim_lookup[key]['trial']
                        else:
                            trial = -1
                        
                        all_configs.append(cfg)
                        all_electrodes.append(int(e))
                        all_currents.append(curr)
                        all_trials.append(trial)
                        all_stim_times_in_snippet.append(t_rel)
                
                # Determine first stim metadata (if any)
                has_stim = len(all_configs) > 0
                if has_stim:
                    # Sort by time to get first stim
                    first_idx = np.argmin(all_stim_times_in_snippet)
                    first_config = all_configs[first_idx]
                    first_electrode = all_electrodes[first_idx]
                    first_current = all_currents[first_idx]
                    first_trial = all_trials[first_idx]
                    first_stim_time = all_stim_times_in_snippet[first_idx]
                else:
                    first_config = 0
                    first_electrode = -1
                    first_current = 0
                    first_trial = -1
                    first_stim_time = -1
                
                results.append({
                    'session': session,
                    'snippet_start': i,
                    # First stim metadata (for primary labeling)
                    'first_config': first_config,
                    'first_electrode': first_electrode,
                    'first_current': first_current,
                    'first_trial': first_trial,
                    'first_stim_time': first_stim_time,  # relative to snippet start
                    # All stims in snippet (for holdout filtering)
                    'all_configs': all_configs if all_configs else [],
                    'all_electrodes': all_electrodes if all_electrodes else [],
                    'all_currents': all_currents if all_currents else [],
                    'all_trials': all_trials if all_trials else [],
                    # Flags
                    'has_stim': has_stim,
                    'stim_at_t0': first_stim_time == 0 if has_stim else False,
                    'num_stims': len(all_configs),
                    # Data
                    'initial_condition': initial_cond,
                    'activity_snippet': activity_snippet,
                    'stim_snippet': stim_snippet,
                    'valid': True
                })
    else:
        # Only create snippets aligned to trial stim times
        for idx, row in trials_df.iterrows():
            session = row['session']
            stim_time = row['stim_time']
            
            n_times = activity[session].shape[0]
            
            # Check if snippet fits within session
            if stim_time + length > n_times:
                results.append({
                    'session': session,
                    'snippet_start': stim_time,
                    'first_config': row['config'],
                    'first_electrode': row['electrode'],
                    'first_current': row['current'],
                    'first_trial': row['trial'],
                    'first_stim_time': 0,
                    'all_configs': [row['config']],
                    'all_electrodes': [row['electrode']],
                    'all_currents': [row['current']],
                    'all_trials': [row['trial']],
                    'has_stim': row['is_stim'],
                    'stim_at_t0': True,
                    'num_stims': 1,
                    'initial_condition': None,
                    'activity_snippet': None,
                    'stim_snippet': None,
                    'valid': False
                })
                continue
            
            # Get initial condition
            if stim_time == 0:
                initial_cond = activity[session][0, :]
            else:
                initial_cond = activity[session][stim_time - 1, :]
            
            # Get snippets
            activity_snippet = activity[session][stim_time:stim_time + length, :]
            stim_snippet = stim_times_sess[session][stim_time:stim_time + length, :]
            
            # For non-overlap mode, find all stims in snippet (similar to overlap mode)
            all_configs = [row['config']]
            all_electrodes = [row['electrode']]
            all_currents = [row['current']]
            all_trials = [row['trial']]
            
            # Check for additional stims after t=0
            for t_rel in range(1, length):
                t_abs = stim_time + t_rel
                stim_at_t = stim_snippet[t_rel, :]
                electrodes_active = np.where(stim_at_t > 0)[0]
                for e in electrodes_active:
                    curr = int(stim_at_t[e])
                    cfg = e * 3 + (curr - 3) + 1
                    key = (session, t_abs)
                    if key in stim_lookup:
                        trial = stim_lookup[key]['trial']
                    else:
                        trial = -1
                    all_configs.append(cfg)
                    all_electrodes.append(int(e))
                    all_currents.append(curr)
                    all_trials.append(trial)
            
            results.append({
                'session': session,
                'snippet_start': stim_time,
                'first_config': row['config'],
                'first_electrode': row['electrode'],
                'first_current': row['current'],
                'first_trial': row['trial'],
                'first_stim_time': 0,
                'all_configs': all_configs,
                'all_electrodes': all_electrodes,
                'all_currents': all_currents,
                'all_trials': all_trials,
                'has_stim': row['is_stim'],
                'stim_at_t0': True,
                'num_stims': len(all_configs),
                'initial_condition': initial_cond,
                'activity_snippet': activity_snippet,
                'stim_snippet': stim_snippet,
                'valid': True
            })
    
    return pd.DataFrame(results)


def snippets_df_to_arrays(snippets_df):
    """
    Convert a snippets DataFrame to arrays for model training.
    
    Parameters
    ----------
    snippets_df : pd.DataFrame
        DataFrame from make_snippets_df with valid snippets
        
    Returns
    -------
    initial_conditions : ndarray, shape (n_samples, n_rois)
    activity_snippets : ndarray, shape (n_samples, length, n_rois)
    stim_snippets : ndarray, shape (n_samples, length, n_electrodes)
    """
    valid_df = snippets_df[snippets_df['valid']].copy()
    
    initial_conditions = np.stack(valid_df['initial_condition'].values)
    activity_snippets = np.stack(valid_df['activity_snippet'].values)
    stim_snippets = np.stack(valid_df['stim_snippet'].values)
    
    return initial_conditions, activity_snippets, stim_snippets


def get_next_versioned_directory(base_dir_name='rnn_training'):
    """
    Creates a time-stamped directory, checking for existing versions.
    If 'rnn_training/20251218_211900' exists, it creates 'rnn_training/20251218_211900_2'.
    """
    # 1. Get the base timestamp string
    DATETIME_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. Start with the base directory path
    base_path = os.path.join(base_dir_name, DATETIME_STAMP)

    # 3. Initialize path and version counter
    current_path = base_path
    # 4. Create the new, unique directory
    os.makedirs(current_path)

    # 6. Return the path of the created directory
    return current_path

# --- Usage Example ---
new_dir_path = get_next_versioned_directory('rnn_training')
print(f"Created new directory: {new_dir_path}")


def partition_trials(
    initial_conds,
    stim_snippets,
    activity_snippets,
    SNIPPET_LENGTH,
    val_size=0.15,
    test_size=0.15,
    method="timeorder",
):
    """
    Partition trials into train / val / test with no temporal overlap.

    Ensures that snippets in different sets cannot overlap in time by
    inserting SNIPPET_LENGTH gaps at boundaries.

    Returns:
        (init_train, stim_train, act_train,
         init_val,   stim_val,   act_val,
         init_test,  stim_test,  act_test)
    """

    # -------------------- Sanity checks --------------------
    assert initial_conds.shape[0] == stim_snippets.shape[0] == activity_snippets.shape[0]
    assert 0 < val_size < 1
    assert 0 < test_size < 1
    assert val_size + test_size < 1
    assert SNIPPET_LENGTH >= 1

    n_trials = activity_snippets.shape[0]

    if method != "timeorder":
        raise ValueError("Only method='timeorder' is supported.")

    # -------------------- Compute split indices --------------------
    n_test = int(round(n_trials * test_size))
    n_val  = int(round(n_trials * val_size))

    test_start = n_trials - n_test
    val_start  = test_start - n_val

    # -------------------- Enforce non-overlap --------------------
    # Remove SNIPPET_LENGTH samples around boundaries
    train_end = val_start - SNIPPET_LENGTH
    val_end   = test_start - SNIPPET_LENGTH

    if train_end <= 0 or val_end <= val_start:
        raise ValueError(
            "Not enough data to enforce non-overlapping splits. "
            "Reduce SNIPPET_LENGTH or val/test sizes."
        )

    # -------------------- Slice data --------------------
    init_train = initial_conds[:train_end]
    stim_train = stim_snippets[:train_end]
    act_train  = activity_snippets[:train_end]

    init_val = initial_conds[val_start:val_end]
    stim_val = stim_snippets[val_start:val_end]
    act_val  = activity_snippets[val_start:val_end]

    init_test = initial_conds[test_start:]
    stim_test = stim_snippets[test_start:]
    act_test  = activity_snippets[test_start:]

    return (
        init_train, stim_train, act_train,
        init_val,   stim_val,   act_val,
        init_test,  stim_test,  act_test,
    )



def collate_to_device(batch, device):
    # First unzip samples
    inputs_and_init, targets = zip(*batch)

    # Now unzip the nested tuple
    inputs, activity_initial = zip(*inputs_and_init)

    inputs = torch.stack(inputs).to(device, non_blocking=True)
    activity_initial = torch.stack(activity_initial).to(device, non_blocking=True)
    targets = torch.stack(targets).to(device, non_blocking=True)

    return (inputs, activity_initial), targets

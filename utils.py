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

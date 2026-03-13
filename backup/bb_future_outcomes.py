import itertools
import numpy as np
import gymnasium as gym
import uppaal_gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from collections.abc import Iterable

from sklearn.calibration import calibration_curve
from torch.nn import functional as F
from torch.utils.data import random_split, TensorDataset, DataLoader

from models import FFNet, Predictor, EventGRU, EventGRUBitLevel
from stable_baselines3 import PPO
from trees.models import QTree, DecisionTree
from utils import load_or_train_model, is_in, normalize


class EventTracker:
    def __init__(self, n=8):
        self.n = n
        self.dtype = np.int8

    def get_events(self, s0, a0, s1):
        p0, v0 = s0
        p1, v1 = s1

        out = np.array([
            a0 == 1,                                    # did we swing?
            v0 < 0,                                     # were the ball falling?
            v1 < 0,                                     # is the ball falling?
            v1 < -4,                                    # velocity to low to hit
            p1 < 4,                                     # position to low to hit
            v0 > 0 and v1 < 0 and a0 == 0,              # did the ball start to drop?
            v0 < 0 and v1 > 0,                          # did we bounce?
            p0 < 4 and p1 < 4 and v0 > 0 and v1 < 0,    # are we going to die?
        ], dtype=self.dtype)

        assert len(out) == self.n
        return out

    def state_to_string(self, state):
        out = np.array([
            'swing',
            'were falling',
            'is falling',
            'is falling fast',
            'too low',
            'start drop',
            'bounce',
            'dying'
        ])

        msg = [
            'swing' if state[0] else 'no swing',
            'were falling' if state[1] else 'were rising',
            'is falling' if state[2] else 'is rising',
            'is falling fast' if state[3] else '',
            'too low' if state[4] else '',
            'start drop' if state[5] else '',
            'bounce' if state[6] else '',
            'dying' if state[7] else ''
        ]
        return ' - '.join(msg)

        # return ', '.join(list(out[np.array(state, dtype=np.bool)]))

    def state_to_index(self, state):
        """Convert a state (list of events) to a unique index"""
        return int("".join(map(str, state)), 2)

    def index_to_state(self, index):
        """Convert a state index back into a list of events"""
        return [int(b) for b in format(index, f'0{self.n}b')]

    def tensor_to_state(self, tensor):
        arr = tensor.detach().numpy()
        arr[arr >= 0.5] = 1
        arr[arr <  0.5] = 0
        return arr


def generate_agent_data(model, env, n_runs=500, eps=0.0):
    state_data = []
    event_data = []
    tracker = EventTracker()

    for _ in range(n_runs):
        episode_states = []
        episode_events = []

        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        done = False
        while not done:
            nobs, reward, done, _, _ = env.step(action)

            episode_states.append((*obs, action.item()))
            episode_events.append(tracker.get_events(obs, action.item(), nobs))

            # choose next action
            if np.random.random() < eps:
                action = model.action_space.sample()
            else:
                # if is_in(nobs, [5,6], 0.5):
                #     action = np.array(0)
                # else:
                action, _ = model.predict(nobs, deterministic=True)
            obs = nobs

        state_data.append(episode_states)
        event_data.append(episode_events)

    state_data = np.array(state_data)
    event_data = np.array(event_data)

    return state_data, event_data


def prepare_data_for_pytorch(state_data, event_data, horizon=10, mean=None, std=None):

    assert state_data.shape[:2] == event_data.shape[:2]
    episodes, timesteps = state_data.shape[:2]


    X, y = [], []
    for ep in range(episodes):
        states, events = state_data[ep], event_data[ep]

        for i in range(timesteps-horizon):
            X.append(states[i])
            y.append(events[i:i+horizon])

    X = np.array(X)
    y = np.array(y)

    # normalize states
    # mean = X[:,:2].mean(axis=0)
    # std = X[:,:2].std(axis=0) + 1e-8
    # X[:,:2] = (X[:,:2] - mean) / std
    if mean is not None and std is not None:
        X[:,:2] = normalize(X[:,:2], mean, std)

    # prepare data for PyTorch
    X_train = torch.tensor(np.array(X), dtype=torch.float32)
    Y_train = torch.tensor(np.array(y), dtype=torch.float32)

    dataset = TensorDataset(X_train, Y_train)

    batch_size = 256
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(
        train_loader, val_loader, in_dim, horizon, n_events,
        n_epochs=200, lr=1e-3, device='cpu'):
    model = Predictor(in_dim, horizon, n_events).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    best_val = float('inf')

    # training
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)

            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)

        avg_train = total_loss / len(train_loader.dataset)

        # validation
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                vloss += criterion(logits, yb).item() * xb.size(0)

            avg_val = vloss / len(val_loader.dataset)

        print(f'Epoch {epoch} train_loss {avg_train:.4f} val_loss {avg_val:.4f}')
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), 'models/best_pred.pt')

    model.load_state_dict(torch.load('models/best_pred.pt', weights_only=True))
    return model


def train_gru(
        train_loader, val_loader, horizon, event_dim, in_dim,
        lr=1e-3, n_epochs=50, device='cpu'
    ):
    model = EventGRU(in_dim, horizon, event_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val = float('inf')
    for epoch in range(1, n_epochs+1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb, targets=yb, teacher_forcing=True)
            loss = criterion(logits, yb.float())

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * xb.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb, targets=yb, teacher_forcing=True)
                loss = criterion(logits, yb.float())
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch:02d}: train {train_loss:.4f}, val {val_loss:.4f}')
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), './models/best_gru.pt')

    model.load_state_dict(torch.load('./models/best_gru.pt', weights_only=True))
    return model


def predict_sequence(model, x, horizon=10, device='cpu'):
    """
    Autoregressively predict event probabilities for horizon steps.

    Args:
        model: trained EventGRU
        x: (B, in_dim) tensor [normalized state + action scalar]
        horizon: number of future steps (default=10)

    Returns:
        probs: (B, horizon, event_dim) tensor of probabilities in [0,1]
    """
    model.eval()
    B = x.size(0)
    # device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        # encode initial hidden state
        h = model.encoder(x).unsqueeze(0)   # (1, B, hidden)
        prev_event = torch.zeros(B, 1, model.event_dim, device=device)

        outputs = []
        for t in range(horizon):
            out, h = model.gru(prev_event, h)       # (B,1,H)
            logits = model.decoder(out.squeeze(1))  # (B, event_dim)
            probs = torch.sigmoid(logits)           # convert logits to probs
            outputs.append(probs)

            # feed prediction back for next step
            prev_event = probs.unsqueeze(1)

        outputs = torch.stack(outputs, dim=1)
    return outputs


def predict_sequence_bitlevel(model, x, horizon=10):
    model.eval()
    with torch.no_grad():
        logits = model(x, teacher_forcing=False)
        probs = torch.sigmoid(logits)
    return probs


def brier_score(probs, targets):
    """
    Calculate Brier score (a proper scoring rule for probabilities). Lower means
    better calibrated and more accurate.

    Brier = (1/N) \Sum_{i=1} (p_i - y_i)^2

    probs: (N, horizon, event_dim) predicted probabilities
    targets: (N, horizon, event_dim) binary ground truth
    """
    return torch.mean((probs - targets.float())**2).item()


def reliability_diagram(probs, targets, n_bins=10, event_idx=0, horizon_idx=0):
    """
    Plot reliability curve for a single event at a specific horizon. Answering
    the question: if the model outputs ~0.8 for many events, do 80% of them
    really happen?
    """
    # flatten across samples
    y_true = targets[:, horizon_idx, event_idx].cpu().numpy().ravel()
    y_prob = probs[:, horizon_idx, event_idx].detach().cpu().numpy().ravel()

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    plt.plot(mean_pred, frac_pos, marker='o', label='Model')
    plt.plot([0,1],[0,1],'--', color='gray', label='Perfect')
    plt.xlabel('Predicted probability')
    plt.ylabel('Empirical frequency')
    plt.title(f'Reliability: event {event_idx}, horizon {horizon_idx}')
    plt.legend()
    plt.show()
    plt.close()


def get_args_in_range(data, low1, high1, low2, high2, action=None):
    return np.argwhere(
        np.logical_and(
            data[:,:,0] > low1,
            np.logical_and(
                data[:,:,0] < high1,
                np.logical_and(
                    data[:,:,1] > low2,
                    np.logical_and(
                        data[:,:,1] < high2,
                        True if action is None else data[:,:,2] == action
                    )
                )
            )
        )
    )


def joint_from_independent(probs, normalize=False, eps=1e-12):
    """
    Compute joint distribution over 2^D configurations assuming independent bits.

    Args:
        probs: (B,H,D) tensor of marginal probabilities from sigmoids
        normalize: bool, whether to renormalize the joint distribution
                   (useful after sharpening, pruning, or masking)
        eps: small constant for numerical stability

    Returns:
        joint: (B,H,2^D) tensor of probabilities
    """
    B,H,D = probs.shape
    device = probs.device

    # Precompute all bit configs in lexicographic order
    configs = torch.tensor(
        [list(map(int, f"{i:0{D}b}")) for i in range(2**D)],
        device=device, dtype=probs.dtype
    )  # (2^D,D)

    # Expand for broadcasting
    p = probs.unsqueeze(2)             # (B,H,1,D)
    configs = configs.view(1,1,2**D,D) # (1,1,2^D,D)

    # Compute log-prob for each configuration
    logp = configs * torch.log(p + eps) + (1-configs) * torch.log(1-p + eps)
    logp = logp.sum(-1)  # (B,H,2^D)

    joint = torch.exp(logp)

    if normalize:
        joint = joint / (joint.sum(-1, keepdim=True) + eps)

    return joint


def plot_transition_sankey(probs, timestep=0, min_flow=1e-6):
    # probs: n x m array where probs[t, s]
    n, m = probs.shape
    if timestep < 0 or timestep >= n-1:
        raise ValueError("timestep must be in range [0, n-2]")

    src_states = [f"t{timestep}:{s}" for s in range(m)]
    dst_states = [f"t{timestep+1}:{s}" for s in range(m)]

    # Node list (keep unique labels)
    nodes = src_states + dst_states
    node_colors = ["rgba(31,119,180,0.9)"]*m + ["rgba(255,127,14,0.9)"]*m

    # Build links from each source state to each destination state weighted by count/probability:
    # We treat flows as number of items = probs[timestep, s_src] * probs[timestep+1, s_dst]
    flows = []
    sources = []
    targets = []
    for s in range(m):
        for d in range(m):
            value = probs[timestep, s] * probs[timestep+1, d]
            if value >= min_flow:
                sources.append(s)  # index in nodes list (source block)
                targets.append(m + d)  # destination block index
                flows.append(value)

    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, color=node_colors),
        link=dict(source=sources, target=targets, value=flows,
                  hovertemplate='Source: %{source.label}<br>Target: %{target.label}<br>Value: %{value}<extra></extra>')
    ))
    fig.update_layout(title=f"Transition Sankey t={timestep} → t={timestep+1}", font_size=10)
    fig.show()


def plot_event_probabilities(*joints):
    tracker = EventTracker()
    fig, axs = plt.subplots(1, len(joints))

    if not isinstance(axs, Iterable):
        axs = [axs]

    for ax, probs in zip(axs, joints):
        n, m = probs.shape

        # Identify outcomes with nonzero probability
        nonzero_cols = np.any(probs > 0, axis=0)

        active_outcomes = np.where(nonzero_cols)[0]
        m_active = len(active_outcomes)

        # Assign colors only to active outcomes
        cmap = plt.get_cmap("tab20", m_active)
        outcome_to_color = {outcome: cmap(i) for i, outcome in enumerate(active_outcomes)}

        # Plot stacked bars
        bottom = np.zeros(n)
        bars = []

        for outcome in active_outcomes:
            j = outcome
            color = outcome_to_color[j]
            b = ax.bar(range(n), probs[:, j], bottom=bottom, color=color)

            for i, rect in enumerate(b):
                rect._bar_index = i
                rect._outcome_index = j
                bars.append(rect)
            bottom += probs[:, j]

        # Legend only for active outcomes
        handles = [plt.Rectangle((0,0),1,1, color=outcome_to_color[j]) for j in active_outcomes]
        labels = [f"{tracker.state_to_string(tracker.index_to_state(j))}" for j in active_outcomes]
        ax.legend(handles, labels, title="Outcomes")

        ax.set_xticks(range(n))
        ax.set_xticklabels([f"Timestep {i+1}" for i in range(n)])
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)

    plt.show()


def sharpen_probs(probs, temp=0.5, eps=1e-8):
    logit = torch.log(probs + eps) - torch.log(1 - probs + eps)
    sharp = torch.sigmoid(logit / temp)
    return sharp


def prune_and_renorm_joint(joint, thresh=1e-4):
    # joint: (B,H,256)
    j = joint.clone()
    j[j < thresh] = 0.0
    s = j.sum(dim=-1, keepdim=True)
    zero_mask = (s.squeeze(-1) == 0)

    s = s + (s==0).float() # avoid divide by zero
    j = j / s
    if zero_mask.any():
        # fallback to original normalized joint when pruning zeroed everything
        orig = joint / (joint.sum(-1, keepdim=True) + 1e-12)
        j[zero_mask] = orig[zero_mask]
    return j


STEP_TS = 0.3
N_STEPS = 400
ENV_ID = 'BouncingBall-v0'

STATE_OF_INTEREST = ((5,6), 1)

if __name__ == '__main__':
    env_kwargs = {
        'ts_size': STEP_TS,
        'max_n_steps': N_STEPS,
    }
    env = gym.make(ENV_ID, **env_kwargs)

    model = load_or_train_model(env, './models/bb_ppo_with_replay_250.zip')

    state_data, event_data = generate_agent_data(model, env, n_runs=100, eps=0.0)

    # get mean and std for normalization
    mean = state_data.reshape((-1,3))[:,:2].mean(axis=0)
    std = state_data.reshape((-1,3))[:,:2].std(axis=0)

    train_loader, val_loader = prepare_data_for_pytorch(state_data, event_data, mean=mean, std=std)

    # net = train_model(train_loader, val_loader, 3, 10, 8)
    # try:
    #     net = EventGRU(3, 10, 8)
    #     net.load_state_dict(torch.load('./models/best_gru.pt', weights_only=True))
    #     print('state dict loaded')
    # except:
    #     net = train_gru(train_loader, val_loader, 10, 8, 3, n_epochs=100)



    # import ipdb; ipdb.set_trace()

    # select single instance and see result
    # state_idx = np.argwhere(state_data[:,:,-1] == 1)[0]
    # state = state_data[tuple(state_idx)]
    # state_norm = normalize(state[:2], mean, std)
    # state[:2] = state_norm
    # x_tensor = torch.tensor(state[None,:], dtype=torch.float32)
    # probs = predict_sequence_bitlevel(net, x_tensor)
    # print('Event probabilities for 10 steps:')
    # print(probs.squeeze(0).cpu().numpy())

    # # test with Brier score and reliability diagram
    # test_states, test_events = generate_agent_data(model, env, n_runs=2, eps=0.0)
    # test_loader, _ = prepare_data_for_pytorch(test_states, test_events, mean=mean, std=std)
    # X_test_tensor, targets = test_loader.dataset.dataset.tensors
    # probs = predict_sequence_bitlevel(net, X_test_tensor)

    # print('Brier score', brier_score(probs, targets))
    # reliability_diagram(probs, targets, event_idx=6, horizon_idx=5)

    # here, we count the event states visited after being in some region
    tracker = EventTracker()
    count_states = np.zeros((10,256))
    state = [5,6]
    action = 1

    eps = 0.5
    p, v = state
    args = get_args_in_range(
        state_data[:,:-10,:], p-eps, p+eps, v-eps, v+eps, action=action
    )
    print(args.shape)

    for ep, idx in args:
        for i in range(10):
            event_idx = tracker.state_to_index(event_data[ep,idx+i])
            count_states[i,event_idx] += 1

    # normalize to probabilities
    joint1 = count_states / count_states.sum(axis=1, keepdims=True)
    plot_event_probabilities(joint1)
    # plot_transition_sankey(joint1)
    import ipdb; ipdb.set_trace()

    # x_norm = normalize(state, mean, std)
    # x_tensor = torch.tensor(np.concat([x_norm, [action]])[None,:], dtype=torch.float32)
    # probs = predict_sequence(net, x_tensor)
    # probs_sharp = sharpen_probs(probs, temp=0.5)
    # joint2 = joint_from_independent(probs_sharp)
    # joint_pruned = prune_and_renorm_joint(joint2, thresh=1e-4)
    # joint2 = joint2.squeeze(0).cpu().detach().numpy()
    # joint_pruned = joint_pruned.squeeze(0).cpu().detach().numpy()

    # plot_event_probabilities(joint_pruned)


    # import ipdb; ipdb.set_trace()
    # import sys; sys.exit(0)

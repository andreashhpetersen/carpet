import sys
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical
from torch.optim import Adam
from sklearn.neural_network import MLPRegressor


class BouncingBallActionInfluenceModel:
    def __init__(self, action, pa_pos, pa_vel, p_id=0, v_id=1):

        self.p_id = p_id
        self.v_id = v_id

        self.action = action
        self.pa_pos = pa_pos
        self.pa_vel = pa_vel

        self.clf_p = MLPRegressor(hidden_layer_sizes=(16, 32, 16), max_iter=500, alpha=1e-5)
        self.clf_v = MLPRegressor(hidden_layer_sizes=(16, 32, 16), max_iter=500, alpha=1e-5)

    def fit(self, X, y):
        print(f'Fitting AIM for {"`HIT`" if self.action == 1 else "`NO HIT`"}')
        Xp = X[:,self.pa_pos]
        yp = y[:,self.p_id]

        Xv = X[:,self.pa_vel]
        yv = y[:,self.v_id]

        self.clf_p.fit(Xp, yp)
        self.clf_v.fit(Xv, yv)

    def predict(self, inp, target):
        if target == 0:
            return self.clf_p.predict(inp)
        else:
            return self.clf_v.predict(inp)

    def evaluate(self, X, y, target):
        y_hat = self.predict(X, target)
        return np.square(y[:,target] - y_hat).mean()


class VPG(nn.Module):
    def __init__(
            self, in_dims, num_actions, train_v_iters=80,
            hidden_size=32, pi_lr=3e-4, vf_lr=1e-3, gamma=0.99, lamb=0.97
    ):
        super(VPG, self).__init__()

        self.hidden_size = hidden_size
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.gamma = gamma
        self.lamb = lamb
        self.train_v_iters = train_v_iters

        self.logits_net = nn.Sequential(
            nn.Linear(in_dims, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions),
            nn.Identity()
        )

        self.value_net = nn.Sequential(
            nn.Linear(in_dims, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Identity()
        )

        self.pi_optimizer = Adam(self.logits_net.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.value_net.parameters(), lr=vf_lr)

    def get_policy(self, obs):
        """get the action distribution from `obs`"""
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        return Categorical(logits=self.logits_net(obs))

    def get_action(self, obs):
        """sample a random action from `obs`"""
        return self.get_policy(obs).sample().item()

    def predict(self, obs, *args, **kwargs):
        """predict action from `obs`, in accordance with sb3 and gym API"""
        if obs.ndim == 1:
            obs = np.array([obs])

        act = torch.argmax(self.get_policy(obs).logits)
        out = act.reshape(obs.shape[0], 1).detach().numpy(), None
        return out

    def discounted_cumsum(self, xs, discount):
        """calculate the discounted cumulative sums of xs"""
        n = len(xs)
        rtgs = np.zeros_like(xs, dtype=np.float32)
        for i in reversed(range(n)):
            rtgs[i] = xs[i] + discount * (rtgs[i+1] if i+1 < n else 0)
        return rtgs

    def compute_loss_pi(self, obs, act, weights):
        """loss function for policy network"""
        logp = self.get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    def compute_loss_v(self, obs, rets):
        """loss function for value network"""
        loss_func = nn.MSELoss()
        return loss_func(self.value_net(obs).squeeze(), rets)

    def compute_gae(self, rews, vals):
        """compute the generalized advantage estimations"""
        deltas = np.zeros_like(rews)
        n = len(rews)
        for i in reversed(range(n)):
            deltas[i] = rews[i] - vals[i] + self.gamma * (vals[i+1] if i+1 < n else 0)
        return self.discounted_cumsum(deltas, self.gamma * self.lamb)

    def learn(self, env_name, epochs=50, batch_size=5000):
        env = gym.make(env_name)

        for e in range(epochs):
            batch_obs = []
            batch_acts = []
            batch_weights = []
            batch_rets = []
            batch_rtgs = []
            batch_lens = []
            batch_vals = []

            obs, _ = env.reset()
            done = False
            ep_rews = []
            ep_obs = []
            ep_vals = []

            while True:
                ep_obs.append(obs.copy())

                act = self.get_action(torch.as_tensor(obs, dtype=torch.float32))
                obs, rew, done, _, _ = env.step(act)

                batch_acts.append(act)
                ep_rews.append(rew)

                if done:
                    ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                    batch_rets.append(ep_ret)
                    batch_lens.append(ep_len)

                    # discounted rewards to go
                    ep_rtgs = self.discounted_cumsum(ep_rews, self.gamma)

                    # estimated values
                    ep_vals = self.value_net(torch.as_tensor(np.array(ep_obs)))

                    batch_obs += list(ep_obs)
                    batch_rtgs += list(ep_rtgs)
                    batch_vals += list(ep_vals)
                    # batch_weights += list(ep_rtgs - ep_vals.squeeze().detach().numpy())

                    batch_weights += list(self.compute_gae(ep_rews, ep_vals))

                    (obs, _), done = env.reset(), False
                    ep_rews, ep_obs, ep_vals = [], [], []

                    if len(batch_obs) > batch_size:
                        break

            batch_obs = np.array(batch_obs)
            batch_rtgs = np.array(batch_rtgs)

            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(
                obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                act=torch.as_tensor(batch_acts, dtype=torch.float32),
                weights=torch.as_tensor(batch_weights, dtype=torch.float32)
            )
            loss_pi.backward()
            self.pi_optimizer.step()

            v_losses = []
            for i in range(self.train_v_iters):
                self.vf_optimizer.zero_grad()
                loss_v = self.compute_loss_v(
                    torch.as_tensor(batch_obs, dtype=torch.float32),
                    torch.as_tensor(batch_rtgs, dtype=torch.float32)
                )
                v_losses.append(loss_v.item())
                loss_v.backward()
                self.vf_optimizer.step()

            print(
                'epoch: %3d \t loss_pi: %.3f \t loss_vf: %.3f \t return: %.3f \t ep_len: %.3f' %
                (e, loss_pi, np.mean(v_losses), np.mean(batch_rets), np.mean(batch_lens))
            )


class A2C(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, gamma=0.99, lr=3e-4):
        super(A2C, self).__init__()

        self.gamma = gamma

        self.num_actions = num_actions
        self.critic_lin1 = nn.Linear(num_inputs, hidden_size)
        self.critic_lin2 = nn.Linear(hidden_size, 1)

        self.actor_lin1 = nn.Linear(num_inputs, hidden_size)
        self.actor_lin2 = nn.Linear(hidden_size, num_actions)

        self.optimizer = Adam(self.parameters(), lr=lr)

    def forward(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        dim = state.ndim - 1

        value = F.relu(self.critic_lin1(state))
        value = self.critic_lin2(value)

        action_probs = F.relu(self.actor_lin1(state))
        action_probs = F.softmax(self.actor_lin2(action_probs), dim=dim)

        return value, action_probs

    def predict(self, obs, *args, **kwargs):
        _, probs = self.forward(obs)
        act = torch.argmax(probs).detach().numpy().reshape(obs.shape[0], 1)
        return act, None

    def learn(self, env, epochs=50):

        all_lengths, avg_lengths, all_rewards = [], [], []
        entropy_term = 0

        for e in range(epochs):
            ep_rewards, ep_values, ep_log_probs = [], [], []
            obs, _ = env.reset()

            while True:
                value, action_probs = self.forward(obs)
                value = value.detach().numpy()[0,0]
                dist = action_probs.detach().numpy()

                action = np.random.choice(self.num_actions, p=np.squeeze(dist))
                log_prob = torch.log(action_probs.squeeze(0)[action])
                entropy = -np.sum(np.mean(dist) * np.log(dist))
                new_obs, reward, done, _, _ = env.step(action)

                ep_rewards.append(reward)
                ep_values.append(value)
                ep_log_probs.append(log_prob)
                entropy_term += entropy
                obs = new_obs

                if done:
                    q_val, _ = self.forward(new_obs)
                    q_val = q_val.detach().numpy()[0,0]
                    all_rewards.append(np.sum(ep_rewards))
                    all_lengths.append(len(ep_rewards))
                    avg_lengths.append(np.mean(all_lengths[-10:]))
                    if e % 10 == 0:
                        sys.stdout.write(
                            "episode: {}, reward: {}, total_length: {}, " \
                            "average_length: {} \n".format(
                                e, np.sum(ep_rewards), len(ep_rewards), avg_lengths[-1]
                            )
                        )
                    break

            q_vals = np.zeros_like(ep_values)
            for t in reversed(range(len(ep_rewards))):
                q_val = ep_rewards[t] + self.gamma * q_val
                q_vals[t] = q_val

            values = torch.as_tensor(ep_values, dtype=torch.float32)
            q_vals = torch.as_tensor(q_vals, dtype=torch.float32)
            log_probs = torch.stack(ep_log_probs)

            advantage = q_vals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

            self.optimizer.zero_grad()
            ac_loss.backward()
            self.optimizer.step()


class FFNet(nn.Module):
    def __init__(self, out_dim):
        super(FFNet, self).__init__()

        self.layer1 = nn.Linear(4, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, out_dim)

    def forward(self, inp):
        y1 = F.relu(self.layer1(inp))
        y2 = F.relu(self.layer2(y1))
        return self.layer3(y2)


class Predictor(nn.Module):
    def __init__(self, in_dim, horizon, event_dim, hidden=256, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.event_dim = event_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.horizon*self.event_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        logits = logits.view(-1, self.horizon, self.event_dim)
        return logits


class EventGRU(nn.Module):
    def __init__(self, in_dim, horizon, event_dim, hidden=128):
        super().__init__()
        self.horizon = horizon
        self.event_dim = event_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        self.gru = nn.GRU(input_size=event_dim, hidden_size=hidden, batch_first=True)

        self.decoder = nn.Linear(hidden, event_dim)

    def forward(self, x, targets=None, teacher_forcing=True):
        """
        x: (B, in_dim)  [state+action]
        targets: (B, horizon, event_dim) or None
        teacher_forcing: use ground truth events as inputs during training
        """
        B = x.size(0)
        device = x.device

        # encode input to initial hidden state
        h0 = self.encoder(x).unsqueeze(0)   # (1, B, hidden)

        # start with zeros as the 'previous event'
        prev_event = torch.zeros(B, 1, self.event_dim, device=device)
        outputs = []
        h = h0

        for t in range(self.horizon):
            out, h = self.gru(prev_event, h)        # out: (B,1,H)
            logits = self.decoder(out.squeeze(1))   # (B, event_dim)
            outputs.append(logits)

            if teacher_forcing and targets is not None:
                prev_event = targets[:, t].unsqueeze(1).float()     # ground truth
            else:
                prev_event = torch.sigmoid(logits).unsqueeze(1)     # model prediction

        outputs = torch.stack(outputs, dim=1)   # (B, horizon, event_dim)
        return outputs


class EventGRUBitLevel(nn.Module):
    def __init__(self, in_dim, horizon, event_dim, hidden=128):
        super().__init__()
        self.horizon = horizon
        self.event_dim = event_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        self.gru = nn.GRU(input_size=event_dim, hidden_size=hidden, batch_first=True)

        self.bit_decoder = nn.Linear(hidden + event_dim, 1)

    def forward(self, x, targets=None, teacher_forcing=True):
        """
        x: (B, in_dim) state+action
        targets: (B, horizon, event_dim) binary ground truth data
        """
        B = x.size(0)
        device = x.device

        # initial hidden state from encoder
        h = self.encoder(x).unsqueeze(0)    # (1,B,H)
        prev_event = torch.zeros(B, 1, self.event_dim, device=device)

        all_logits = []

        for t in range(self.horizon):
            # temporal update
            out, h = self.gru(prev_event, h)    # out: (B,1,H)
            ht = out.squeeze(1)                 # (B,H)

            step_logits = []
            prev_bits = torch.zeros(B, 0, device=device)

            for j in range(self.event_dim):
                # conditioning input = hidden state + previous bits (zero-padded to length j)
                if j > 0:
                    bit_input = torch.cat([prev_bits, torch.zeros(B, self.event_dim-j, device=device)], dim=1)
                else:
                    bit_input = torch.zeros(B, self.event_dim, device=device)

                decoder_in = torch.cat([ht, bit_input], dim=1)  # (B, H+event_dim)
                logit = self.bit_decoder(decoder_in)            # (B, 1)
                step_logits.append(logit)

                # teacher forcing or sampling
                if teacher_forcing and targets is not None:
                    bit = targets[:,t,j].unsqueeze(1).float()
                else:
                    bit = torch.sigmoid(logit) > 0.5
                    bit = bit.float()
                prev_bits = torch.cat([prev_bits, bit], dim=1)

            step_logits = torch.cat(step_logits, dim=1)     # (B, event_dim)
            all_logits.append(step_logits)
            prev_event = step_logits.unsqueeze(1)           # feed into GRU

        return torch.stack(all_logits, dim=1)   # (B, horizon, event_dim)

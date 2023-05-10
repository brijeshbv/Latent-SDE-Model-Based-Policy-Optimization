import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Normal
import torchsde
import itertools
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class StandardScaler(object):
    def __init__(self):
        pass

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out


class LatentSDE(nn.Module):
    sde_type = "stratonovich"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, reward_size, context_size, hidden_size, action_dim, t0=0,
                 skip_every=1,
                 t1=0.1, dt=0.05):
        super(LatentSDE, self).__init__()
        # hyper-parameters
        kl_anneal_iters = 700
        lr_init = 1e-3
        lr_gamma = 0.9997

        # Encoder.
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.skip_every = skip_every
        self.reward_size = reward_size
        self.latent_size = latent_size
        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, latent_size),
        )

        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1),
                )
                for _ in range(latent_size)
            ]
        )

        self.projector = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, reward_size + data_size),
        )
        latent_and_action_size = latent_size + action_dim + data_size
        self.action_encode_net = nn.Sequential(
            nn.Linear(latent_and_action_size, latent_size))
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None
        self.optimizer = optim.Adam(params=self.parameters(), lr=lr_init)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=lr_gamma)
        self.kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def contextualize_time(self, i):
        self.ti = i

    def f(self, t, y):
        ctx, ts = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t, y):
        out = self.h_net(y)
        return out

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs, actions, rewards, adjoint=True, method="reversible_heun"):
        assert xs.shape[0] != 0, f'xs does not contain data, {xs.shape}'
        ts = torch.linspace(self.t0, self.t1, steps=xs.shape[0], device=device)
        ts = torch.permute(ts.repeat(xs.shape[1], 1).to(device), (1, 0))
        noise_std = 0.01
        # Contextualization is only needed for posterior inference.
        ctx = self.encoder(torch.flip(xs, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        ts_horizon = ts.permute((1, 0))
        self.contextualize((ctx, ts_horizon[0]))
        sampled_t = list(t for t in range(ts.shape[0] - 1) if t % self.skip_every == 0)
        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
        zs = torch.reshape(z0, (1, z0.shape[0], z0.shape[1]))

        xs_mean = self.projector(zs[-1, :, :])

        predicted_xs = xs_mean.reshape(1, xs_mean.shape[0], xs_mean.shape[1])
        for i in sampled_t:
            self.contextualize_time(i)
            if i == 0:
                latent_and_data = torch.cat((zs[-1, :, :], actions[i, :, :], xs[0, :, :]), dim=1)
            elif i < ts.shape[0] - 1:
                latent_and_data = torch.cat((zs[-1, :, :], actions[i - 1, :, :], xs[i - 1, :, :]), dim=1)
            z_encoded = self.action_encode_net(latent_and_data)
            if self.skip_every == 1:
                t_horizon = ts_horizon[0][i: i + self.skip_every + 1]
            else:
                t_horizon = ts_horizon[0][i: i + self.skip_every]
            if adjoint:
                # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
                adjoint_params = (
                        (ctx,) +
                        tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(
                    self.h_net.parameters()))
                z_pred, log_ratio = torchsde.sdeint_adjoint(
                    self, z_encoded, t_horizon, adjoint_params=adjoint_params, dt=self.dt, logqp=True, method=method,
                    adjoint_method='adjoint_reversible_heun')
            else:
                z_pred, log_ratio = torchsde.sdeint(self, z_encoded, t_horizon, dt=self.dt, logqp=True, method=method)
            xs_mean = self.projector(z_pred)
            if i == 0:
                predicted_xs = xs_mean
                zs = z_pred
            else:
                # xs_ = xs_.reshape(1, xs_.shape[0], xs_.shape[1])
                predicted_xs = torch.cat((predicted_xs, xs_mean[-1].reshape(1, xs_mean.shape[1], xs_mean.shape[2])),
                                         dim=0)
                zs = torch.cat((zs, z_pred[-1].reshape(1, z_pred.shape[1], z_pred.shape[2])), dim=0)
            if i == 0:
                cum_log_ratio = log_ratio
            else:
                cum_log_ratio = torch.cat((cum_log_ratio, log_ratio), dim=0)
        xs_dist = Normal(loc=predicted_xs, scale=noise_std)
        xs_target = torch.concatenate((xs, rewards), dim=2)
        log_pxs = xs_dist.log_prob(xs_target).sum(dim=(0, 2)).mean(dim=0)

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = cum_log_ratio.sum(dim=0).mean(dim=0)
        return log_pxs, logqp0 + logqp_path

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=10, norm_type=2.0)
        self.optimizer.step()
        self.scheduler.step()
        self.kl_scheduler.step()

    def loss(self, log_pxs, log_ratio):
        loss = -log_pxs + log_ratio * self.kl_scheduler.val
        return loss

    @torch.no_grad()
    def sample_fromx0(self, xs, actions=None, z0=None, batch_size=32, steps=2):
        bm = torchsde.BrownianInterval(t0=self.t0, t1=self.t1, size=(batch_size, self.latent_size,), device=device,
                                       levy_area_approximation="space-time")
        t_horizon = torch.linspace(self.t0, self.t1, steps=steps, device=device)
        print(f'predicting samples, input_size: {xs.shape}')
        assert not torch.isnan(z0).any(), f'z0 latent vector was nan, {z0}'
        for i in range(xs.shape[0]):
            latent_and_data = torch.cat((z0[-1, :, :], actions[i, :, :], xs[i, :, :]), dim=1)
            z_encoded = self.action_encode_net(latent_and_data)
            assert not torch.isnan(z_encoded).any(), f'input latent vector was nan, {z_encoded}'
            z_pred = torchsde.sdeint(self, z_encoded, t_horizon, dt=self.dt, names={'drift': 'h'}, bm=bm,
                                     method="reversible_heun")
            assert not torch.isnan(z_pred).any(), f'some latent vector was nan, {z_pred.shape} , t_h {t_horizon}'
            xs_hat = self.projector(z_pred)
            assert not torch.isnan(xs_hat).any(), f'some pred state vector was nan, check projector'
            z0 = z_pred[-1].reshape((1, z_pred.shape[1], z_pred.shape[2]))
            if i == 0:
                predicted_xs = xs_hat[-1]
            else:
                predicted_xs = torch.cat((predicted_xs, xs_hat[-1]), dim=0)
        return predicted_xs


class LatentSDEModel:
    def __init__(self, network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=200,
                 context_size=64):
        self._snapshots = None
        self._state = None
        self._max_epochs_since_update = None
        self._epochs_since_update = None
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.network_size = network_size
        self.elite_model_idxes = []
        self.ensemble_model = LatentSDE(state_size, state_size + reward_size, 1, context_size, hidden_size, action_size)
        self.scaler = StandardScaler()

    def train(self, inputs, labels, actions, rewards, batch_size=256, holdout_ratio=0., max_epochs_since_update=5):
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}
        steps_factor = 5
        num_holdout = int(inputs.shape[0] * holdout_ratio)
        # no permuting required for SDE models
        # permutation = np.random.permutation(inputs.shape[0])
        # inputs, labels = inputs[permutation], labels[permutation]
        inputs, labels, rewards = inputs[: ((inputs.shape[0] // steps_factor) * steps_factor)], labels[: ((labels.shape[0] // steps_factor) * steps_factor)], rewards[: ((rewards.shape[0] // steps_factor) * steps_factor)]
        train_inputs, train_labels, train_rewards = inputs[num_holdout:], labels[num_holdout:], rewards[num_holdout:]
        train_actions_inputs = actions[num_holdout:]
        holdout_inputs, holdout_labels, holdout_rewards = inputs[:num_holdout], labels[:num_holdout], rewards[
                                                                                                      :num_holdout]
        holdout_actions_inputs = actions[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        self.scaler.fit(train_actions_inputs)
        train_actions_inputs = self.scaler.transform(train_actions_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_rewards = torch.from_numpy(holdout_rewards).float().to(device)
        holdout_actions_inputs = torch.from_numpy(holdout_actions_inputs).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])
        holdout_rewards = holdout_rewards[None, :].repeat([self.network_size, 1])
        holdout_actions_inputs = holdout_actions_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_steps = holdout_inputs.shape[1] // steps_factor
        holdout_inputs = self.chunkify_into_steps(holdout_inputs, holdout_steps)
        holdout_actions_inputs = self.chunkify_into_steps(holdout_actions_inputs, holdout_steps)
        holdout_rewards = self.chunkify_into_steps(holdout_rewards, holdout_steps)
        batch_size = 250
        print(f'training model, train_size : {train_inputs.shape}')
        #todo itertools.count()
        for epoch in itertools.count():
            train_idx = np.vstack([range(train_inputs.shape[0]) for _ in range(self.network_size)])
            # train_idx = np.vstack([np.arange(train_inputs.shape[0])] for _ in range(self.network_size))
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos: start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(device)
                train_reward = torch.from_numpy(train_rewards[idx]).float().to(device)
                train_action_input = torch.from_numpy(train_actions_inputs[idx]).float().to(device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)
                losses = []
                # batch the data in steps of {steps} variable size
                train_steps = train_input.shape[1] // steps_factor

                train_input = self.chunkify_into_steps(train_input, train_steps)
                train_action_input = self.chunkify_into_steps(train_action_input, train_steps)
                train_reward = self.chunkify_into_steps(train_reward, train_steps)
                for i in range(train_input.shape[0]):
                    assert train_input[i].shape[1] > 1, f'steps for prediction is not > 1, {train_input[i].shape[1]}'
                    log_pxs, logqp_path = self.ensemble_model(train_input[i], train_action_input[i], train_reward[i])
                    loss = self.ensemble_model.loss(log_pxs, logqp_path)
                    self.ensemble_model.train(loss)
                    losses.append(loss)

            with torch.no_grad():
                # batch the data in steps of {steps} variable size

                holdout_mse_losses = np.asarray([])
                for i in range(train_input.shape[0]):
                    ho_log_pxs, ho_logqp_path = self.ensemble_model(holdout_inputs[i], holdout_actions_inputs[i],
                                                                    holdout_rewards[i])
                    holdout_mse_loss = self.ensemble_model.loss(ho_log_pxs, ho_logqp_path)
                    holdout_mse_loss = holdout_mse_loss.detach().cpu().numpy()
                    holdout_mse_losses = np.append(holdout_mse_losses, holdout_mse_loss)
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    print(f'training model ended...')
                    break

    def batchify(self, data, batch_size):
        batches, dim = data.shape
        big_chunk = np.array([], dtype=np.float32)
        for j in range(batches):
            if j % batch_size == 0:
                chunk = data[j:j + batch_size]
                if j == 0:
                    big_chunk = np.array(chunk).reshape((1, chunk.shape[0], chunk.shape[1]))
                else:
                    big_chunk = np.append(big_chunk,
                                          np.array(chunk).reshape((1, chunk.shape[0], chunk.shape[1])), axis=0)
        return torch.asarray(big_chunk, dtype=torch.float32).to(device)

    def chunkify_into_steps(self, data, steps):
        op = np.array([], dtype=np.float32)
        if len(data.shape) == 3:
            no_ensemble, no_states, state_dim = data.shape
            for i in range(no_ensemble):
                ensemble = data[i]
                big_chunk = np.array([], dtype=np.float32)
                for j in range(ensemble.shape[0]):
                    if j % steps == 0:
                        chunk = ensemble[j:j + steps]
                        if j == 0:
                            big_chunk = torch.asarray(chunk).reshape((1, chunk.shape[0], chunk.shape[1]))
                        else:
                            assert big_chunk.shape[1] == chunk.shape[0], f'shapes do not match for concatenation, {data.shape}, {steps}'
                            big_chunk = torch.concatenate((big_chunk,
                                                           torch.asarray(chunk).reshape(
                                                               (1, chunk.shape[0], chunk.shape[1]))), dim=0)
                if i == 0:
                    op = big_chunk.reshape((1, big_chunk.shape[0], big_chunk.shape[1], big_chunk.shape[2]))
                else:
                    op = torch.concatenate((op,
                                            big_chunk.reshape(
                                                (1, big_chunk.shape[0], big_chunk.shape[1], big_chunk.shape[2]))),
                                           dim=0)
            op = torch.as_tensor(op)
            op = torch.permute(op, (0, 2, 1, 3))
        elif len(data.shape) == 2:
            data = torch.reshape(data, (data.shape[0], data.shape[1], 1))
            no_ensemble, no_states, state_dim = data.shape
            for i in range(no_ensemble):
                ensemble = data[i]
                big_chunk = np.array([], dtype=np.float32)
                for j in range(ensemble.shape[0]):
                    if j % steps == 0:
                        chunk = ensemble[j:j + steps]
                        if j == 0:
                            big_chunk = torch.asarray(chunk).reshape((1, chunk.shape[0], chunk.shape[1]))
                        else:
                            big_chunk = torch.concatenate((big_chunk,
                                                           torch.asarray(chunk).reshape(
                                                               (1, chunk.shape[0], chunk.shape[1]))), dim=0)
                if i == 0:
                    op = big_chunk.reshape((1, big_chunk.shape[0], big_chunk.shape[1], big_chunk.shape[2]))
                else:
                    op = np.append(op,
                                   big_chunk.reshape((1, big_chunk.shape[0], big_chunk.shape[1], big_chunk.shape[2])),
                                   axis=0)
            op = torch.as_tensor(op)
            op = torch.permute(op, (0, 2, 1, 3))
        else:
            big_chunk = np.array([], dtype=np.float32)
            for j in range(data.shape[0]):
                if j % steps == 0:
                    chunk = data[j:j + steps]
                    if j == 0:
                        big_chunk = torch.asarray(chunk).reshape((1, chunk.shape[0], chunk.shape[1]))
                    else:
                        big_chunk = torch.concatenate((big_chunk,
                                                       torch.asarray(chunk).reshape(
                                                           (1, chunk.shape[0], chunk.shape[1]))), dim=0)
            op = torch.as_tensor(big_chunk)
            op = torch.permute(op, (1, 0, 2))

            # ensemble, steps, batch, dim
        return op

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(self, inputs, actions, batch_size=128, factored=True):
        inputs = self.batchify(inputs, batch_size)
        actions = self.batchify(actions, batch_size)
        z0 = self.ensemble_model.pz0_mean + self.ensemble_model.pz0_logstd.exp() * torch.randn_like(
            self.ensemble_model.pz0_mean)
        z0 = torch.reshape(z0, (z0.shape[0], 1, z0.shape[1])).repeat(1, batch_size, 1).to(device)
        model_op = self.ensemble_model.sample_fromx0(inputs, actions, z0, batch_size).repeat([self.network_size, 1, 1])
        assert not torch.isnan(model_op).any(), f'some predicted state vector was nan, halting progress'
        return model_op

import fire
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import init
from torch.distributions import Normal
import torchsde
import itertools
import tqdm
import wandb
import math

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
    def __init__(self, input_size, hidden_size, output_size, ensemble_size):
        super(Encoder, self).__init__()
        self.lin = nn.Sequential(
            EnsembleFC(input_size, hidden_size, ensemble_size),
            nn.Sigmoid(),
            EnsembleFC(hidden_size, output_size, ensemble_size),
        )

    def forward(self, inp):
        out = self.lin(inp)
        assert not torch.isnan(out).any(), f'encode vector was nan'

        return out


class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0.000075,
                 bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features)).to(device)
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features)).to(device)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #     init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class LatentSDE(nn.Module):
    sde_type = "stratonovich"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, context_size, hidden_size, action_dim, network_size, t0=0,
                 skip_every=1,
                 t1=1, dt=0.5):
        super(LatentSDE, self).__init__()
        # hyper-parameters
        kl_anneal_iters = 700
        lr_init = 1e-3
        lr_gamma = 0.9992

        # Encoder.
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size,
                               ensemble_size=network_size)
        self.qz0_net = EnsembleFC(context_size, latent_size + latent_size, network_size)
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.skip_every = skip_every
        self.latent_size = latent_size
        self.use_decay = True
        self.noise_std = 0.7  # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, latent_size),
        )

        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Sigmoid(),
                    nn.Linear(hidden_size, 1),
                )
                for _ in range(latent_size)
            ]
        )

        self.projector = nn.Sequential(
            EnsembleFC(latent_size, hidden_size, network_size),
            nn.Sigmoid(),
            EnsembleFC(hidden_size, hidden_size, network_size),
            nn.Sigmoid(),
            EnsembleFC(hidden_size, data_size, network_size)
        )
        latent_and_action_size = latent_size + action_dim + data_size
        self.action_encode_net = nn.Sequential(
            EnsembleFC(latent_and_action_size, latent_size, network_size))
        self.pz0_mean = nn.Parameter(torch.zeros(network_size, 1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(network_size, 1, latent_size))

        self._ctx = None
        self.optimizer = optim.Adam(params=self.parameters(), lr=lr_init)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=lr_gamma)
        self.kl_scheduler = LinearScheduler(iters=kl_anneal_iters)
        self.apply(init_weights)

    def f(self, t, y):
        return self.f_net(y)

    def h(self, t, y):
        out = self.h_net(y)
        return out

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
                # print(m.weight.shape)
                # print(m, decay_loss, m.weight_decay)
        return decay_loss

    def forward(self, xs, actions, method="reversible_heun"):
        no_networks, no_states, dim = xs.shape
        no_networks, no_actions, action_dim = actions.shape
        no_batches = 50
        xs, actions = xs.reshape((no_networks, no_states // no_batches, no_batches, dim)), actions.reshape(
            (no_networks, no_actions // no_batches, no_batches, action_dim))
        ts = torch.linspace(self.t0, self.t1, steps=xs.shape[1] + 1, device=device)
        ts = torch.permute(ts.repeat(xs.shape[1], 1).to(device), (1, 0))
        sampled_t = list(t for t in range(ts.shape[0] - 1) if t % self.skip_every == 0)
        for i in sampled_t:
            ctx = self.encoder(xs[:, i, :, :])
            assert not torch.isnan(ctx).any(), f'ctx vector was nan, {i}'
            qz0_mean, qz0_logstd = self.qz0_net(ctx).chunk(chunks=2, dim=2)
            assert not torch.isnan(qz0_mean).any(), f'qz0_mean vector was nan, {i}'

            z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
            if i == 0:
                latent_and_data = torch.cat((z0[:, :, :], actions[:, 0, :, :], xs[:, 0, :, :]), dim=2)
            elif i < ts.shape[0] - 1:
                latent_and_data = torch.cat((zs[:, -1, :, :], actions[:, i, :, :], xs[:, i, :, :]), dim=2)
            z_encoded = self.action_encode_net(latent_and_data)

            z_encoded = z_encoded.reshape((no_networks * no_batches, -1))
            t_horizon = torch.asarray([0, 1])
            adjoint_params = (
                    (ctx,) +
                    tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(
                self.h_net.parameters()))
            z_pred, log_ratio = torchsde.sdeint_adjoint(
                self, z_encoded, t_horizon, adjoint_params=adjoint_params, dt=self.dt, logqp=True, method=method,
                adjoint_method='adjoint_reversible_heun')
            if i == 0:
                zs = z_pred[-1:].reshape((no_networks, 1, no_batches, -1))
            else:
                zs = torch.cat((zs, z_pred[-1:].reshape((no_networks, 1, no_batches, -1))), dim=1)
            xs_mean = self.projector(zs[:, -1, :, :])
            if i == 0:
                predicted_xs = xs_mean
            else:
                predicted_xs = torch.cat((predicted_xs, xs_mean),
                                         dim=1)

            qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
            pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
            logqp0 = torch.distributions.kl_divergence(qz0, pz0)
            logqp0 = logqp0.sum(dim=2).mean(dim=1)

            if i == 0:
                cum_log_ratio = log_ratio.reshape((no_networks, -1, no_batches))
                logqp0_cum = logqp0.reshape(-1, 1)
            else:
                cum_log_ratio = torch.cat((cum_log_ratio, log_ratio.reshape((no_networks, -1, no_batches))), dim=1)
                logqp0_cum = torch.cat((logqp0_cum, logqp0.reshape(-1, 1)), dim=1)

        logqp_path = cum_log_ratio.mean(dim=2).sum(dim=1) + logqp0_cum.sum(dim=1)

        return logqp_path, torch.FloatTensor(predicted_xs)

    def opt_loss(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=10, norm_type=2.0)
        self.optimizer.step()
        self.scheduler.step()
        self.kl_scheduler.step()

    def loss(self, logqp_path, predicted_xs, xs_target):
        xs_dist = Normal(loc=predicted_xs, scale=self.noise_std)
        log_pxs = xs_dist.log_prob(xs_target).mean(dim=(2)).mean(dim=1)
        # * self.kl_scheduler.val
        loss_ensemble = -log_pxs + logqp_path
        loss = loss_ensemble.mean(dim=0)
        if self.use_decay:
            loss += self.get_decay_loss()
        return loss, loss_ensemble


class LatentSDEModel:
    def __init__(self, network_size, elite_size, state_size, action_size, agent, hidden_size=32,
                 context_size=32):
        self.agent = agent
        self._snapshots = None
        self._state = None
        self._max_epochs_since_update = None
        self._epochs_since_update = None
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.network_size = network_size
        self.elite_model_idxes = []
        self.ensemble_model = LatentSDE(state_size, state_size // 2 , context_size, hidden_size, action_size,
                                        self.network_size)
        self.state_scaler = StandardScaler()
        self.action_scaler = StandardScaler()

    @torch.no_grad()
    def ensemble_mse_loss(self, xs_pred, xs):
        assert len(xs_pred.shape) == len(xs.shape) == 3
        mse_loss = torch.mean(torch.pow(xs_pred - xs, 2), dim=(1, 2))
        return mse_loss

    def plot_gym_results(self, X, Xrec, idx=0, show=False, fname='reconstructions.png'):
        tt = 50
        D = np.ceil(X.shape[1]).astype(int)
        nrows = np.ceil(D).astype(int)
        lag = 0
        plt.figure(2, figsize=(40, 40))
        for i in range(D):
            plt.subplot(nrows, 3, i + 1)
            plt.plot(range(0, tt), X[-tt:, i].detach().cpu().numpy(), 'r.-')
            plt.plot(range(lag, tt), Xrec[-tt:, i].detach().cpu().numpy(), 'b.-')
        plt.savefig(f'{fname}-{idx}')
        if show is False:
            plt.close()

    def train(self, args, inputs, labels, actions, total_step, batch_size=256, holdout_ratio=0.,
              max_epochs_since_update=10):
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}
        steps_factor = 5
        num_holdout = int(inputs.shape[0] * holdout_ratio)
        # no permuting required for SDE models
        # permutation = np.random.permutation(inputs.shape[0])
        # inputs, labels = inputs[permutation], labels[permutation]
        inputs, labels = inputs[: ((inputs.shape[0] // steps_factor) * steps_factor)], labels[: (
                (labels.shape[0] // steps_factor) * steps_factor)]
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        train_actions_inputs = actions[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]
        holdout_actions_inputs = actions[:num_holdout]

        self.state_scaler.fit(train_inputs)
        train_inputs = self.state_scaler.transform(train_inputs)
        holdout_inputs = self.state_scaler.transform(holdout_inputs)

        self.action_scaler.fit(train_actions_inputs)
        train_actions_inputs = self.action_scaler.transform(train_actions_inputs)
        holdout_actions_inputs = self.action_scaler.transform(holdout_actions_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)

        holdout_actions_inputs = torch.from_numpy(holdout_actions_inputs).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])

        holdout_actions_inputs = holdout_actions_inputs[None, :, :].repeat([self.network_size, 1, 1])
        batch_size = train_inputs.shape[0]
        print(f'training lsde model, train_size : {train_inputs.shape}')
        max_epoch = 30
        for epoch in itertools.count():
            train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)])
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[start_pos: start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)
                train_action_input = torch.from_numpy(train_actions_inputs[idx]).float().to(device)
                losses = []
                logqp_path, predicted_xs = self.ensemble_model(train_input, train_action_input)
                loss, _ = self.ensemble_model.loss(logqp_path, predicted_xs, train_label)
                self.ensemble_model.opt_loss(loss)
                losses.append(loss)

            with torch.no_grad():

                ho_logqp_path, xs_pred = self.ensemble_model(holdout_inputs, holdout_actions_inputs)
                holdout_mse_loss = self.ensemble_mse_loss(xs_pred, holdout_labels)
                holdout_mse_loss = holdout_mse_loss.detach().cpu().numpy()

                sorted_loss_idx = np.argsort(holdout_mse_loss)
                self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_loss)
                if break_train and total_step > 1000:
                    if total_step % 250 == 0:
                        self.plot_gym_results(holdout_labels[0], xs_pred[0],
                                              fname=f'results/{args.resdir}/train_plt_{total_step}')
                    print(f'training ended epoch no, {epoch}, {holdout_mse_loss}')
                    break
                elif total_step <= 1250 and epoch > 100:
                    if total_step % 250 == 0:
                        self.plot_gym_results(holdout_labels[0], xs_pred[0],
                                              fname=f'results/{args.resdir}/train_plt_{total_step}')
                    print(f'early data training ended epoch no, {epoch}, {holdout_mse_loss}')
                    break

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

    @torch.no_grad()
    def predict(self, args, inputs, actions, steps_to_predict, total_step, normalizer):
        assert len(inputs) > 0, f'predict input is empty'
        inputs = torch.asarray(inputs, dtype=torch.float32).repeat(
            [self.network_size, 1, 1]).to(device)
        actions = torch.asarray(actions, dtype=torch.float32).repeat(
            [self.network_size, 1, 1]).to(device)
        inputs_norm = torch.asarray(self.state_scaler.transform(inputs), dtype=torch.float32)
        actions = torch.asarray(self.action_scaler.transform(actions), dtype=torch.float32)
        num_nets, og_batches, og_dim = inputs.shape
        print(f'predicting {inputs.shape} x {steps_to_predict} samples')
        model_op = torch.empty((0, inputs.shape[0], inputs.shape[1], inputs.shape[2]), dtype=torch.float32)
        for i in range(steps_to_predict):
            _, step_op = self.ensemble_model(inputs_norm, actions)
            step_op = normalizer.inverse_transform(step_op.detach().cpu().numpy())
            step_op = np.add(step_op, inputs)
            inputs = step_op
            inputs = torch.asarray(inputs.reshape((num_nets * og_batches, -1)), dtype=torch.float32).to(device)
            actions = torch.asarray(self.action_scaler.transform(self.agent.select_action(inputs)))
            inputs = inputs.reshape((num_nets, og_batches, -1))
            inputs_norm = torch.asarray(self.state_scaler.transform(inputs),
                                        dtype=torch.float32)
            actions = actions.reshape((num_nets, og_batches, -1))
            model_op = np.concatenate((model_op, step_op.reshape(1, step_op.shape[0], step_op.shape[1], step_op.shape[2])),
                                 axis=0)
        num_steps, ensemb, batch, dim = model_op.shape
        model_op = model_op.reshape((ensemb, num_steps * batch, dim))
        return model_op

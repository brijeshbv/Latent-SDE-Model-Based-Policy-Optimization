import fire
import matplotlib.pyplot as plt
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
            nn.ReLU(),
            EnsembleFC(hidden_size, hidden_size, ensemble_size),
            nn.ReLU(),
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

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float =0.000075, bias: bool = True) -> None:
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
        #self.apply(self._init_weights)
        self.reset_parameters()

    # def _init_weights(self, module):
    #     self.weight.data.normal_(mean=0.0, std=1.0)
    #     if self.bias is not None:
    #         self.bias.data.zero_()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class LatentSDE(nn.Module):
    sde_type = "stratonovich"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, reward_size, context_size, hidden_size, action_dim, network_size ,t0=0,
                 skip_every=1,
                 t1=10, dt=0.1):
        super(LatentSDE, self).__init__()
        # hyper-parameters
        kl_anneal_iters = 700
        lr_init = 0.5e-3
        lr_gamma = 0.9997

        # Encoder.
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size, ensemble_size=network_size)
        self.qz0_net = EnsembleFC(context_size, latent_size + latent_size, network_size)
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.skip_every = skip_every
        self.reward_size = reward_size
        self.latent_size = latent_size
        self.use_decay = True
        self.noise_std = 0.01        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
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
            EnsembleFC(latent_size, hidden_size, network_size),
            nn.Tanh(),
            EnsembleFC(hidden_size, hidden_size, network_size),
            nn.Tanh(),
            EnsembleFC(hidden_size, data_size, network_size),
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
        self.mse_loss = torch.nn.MSELoss()

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
        xs, actions = xs.reshape((no_networks, no_states // no_batches, no_batches, dim)), actions.reshape((no_networks, no_actions // no_batches, no_batches, action_dim))
        ts = torch.linspace(self.t0, self.t1, steps=xs.shape[1]+1, device=device)
        ts = torch.permute(ts.repeat(xs.shape[1], 1).to(device), (1, 0))

        ts_horizon = ts.permute((1, 0))
        sampled_t = list(t for t in range(ts.shape[0] - 1) if t % self.skip_every == 0)
        for i in sampled_t:
            ctx = self.encoder(xs[:,i,:,:])
            assert not torch.isnan(ctx).any(), f'ctx vector was nan, {i}'
            qz0_mean, qz0_logstd = self.qz0_net(ctx).chunk(chunks=2, dim=2)
            assert not torch.isnan(qz0_mean).any(), f'qz0_mean vector was nan, {i}'

            z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
            if i == 0:
                latent_and_data = torch.cat((z0[:, :, :], actions[:, 0, :, :], xs[:, 0, :, :]), dim=2)
            elif i < ts.shape[0] - 1:
                latent_and_data = torch.cat((zs[:, -1, :, :], actions[:, i, :, :], xs[:, i, :, :]), dim=2)
            z_encoded = self.action_encode_net(latent_and_data)

            z_encoded = z_encoded.reshape((no_networks * no_batches,-1))
            if self.skip_every == 1:
                t_horizon = ts_horizon[0][i: i + self.skip_every + 1]
            else:
                t_horizon = ts_horizon[0][i: i + self.skip_every]

            adjoint_params = (
                    (ctx,) +
                    tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(
                self.h_net.parameters()))
            z_pred, log_ratio = torchsde.sdeint_adjoint(
                self, z_encoded, t_horizon, adjoint_params=adjoint_params, dt=self.dt, logqp=True, method=method,
                adjoint_method='adjoint_reversible_heun')
            if i == 0:
                zs = z_pred[-1:].reshape((no_networks,1, no_batches,-1))
            else:
                zs = torch.cat((zs, z_pred[-1:].reshape((no_networks, 1, no_batches,-1))), dim=1)

            xs_mean = self.projector(zs[:, -1, :, :])

            if i == 0:
                predicted_xs = xs_mean
            else:
                predicted_xs = torch.cat((predicted_xs, xs_mean),
                                         dim=1)

            qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
            pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
            logqp0 = torch.distributions.kl_divergence(qz0, pz0)
            logqp0 = logqp0.sum(dim=(2)).mean(dim=1)

            if i == 0:
                cum_log_ratio = log_ratio.reshape((no_networks, -1, no_batches))
                logqp0_cum = logqp0.reshape(-1,1)
            else:
                cum_log_ratio = torch.cat((cum_log_ratio, log_ratio.reshape((no_networks, -1, no_batches))), dim=1)
                logqp0_cum = torch.cat((logqp0_cum, logqp0.reshape(-1,1)), dim=1)

        logqp_path = cum_log_ratio.mean(dim=2).sum(dim=1) + logqp0_cum.sum(dim=1)

        return  logqp_path, predicted_xs

    def opt_loss(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=10, norm_type=2.0)
        self.optimizer.step()
        self.scheduler.step()
        self.kl_scheduler.step()

    def loss(self,logqp_path,  predicted_xs, xs_target):
        xs_dist = Normal(loc=predicted_xs, scale=self.noise_std)
        log_pxs = xs_dist.log_prob(xs_target).sum(dim=(2)).mean(dim=1)

        loss_ensemble = -log_pxs + logqp_path * self.kl_scheduler.val
        loss = loss_ensemble.mean(dim=0)
        if self.use_decay:
            loss += self.get_decay_loss()
        return loss, loss_ensemble

    @torch.no_grad()
    def sample_fromx0(self, xs, actions=None, batch_size=32, steps=2):
        bm = torchsde.BrownianInterval(t0=self.t0, t1=self.t1, size=(xs.shape[0] * xs.shape[1], self.latent_size,), device=device,
                                       levy_area_approximation="space-time")
        t_horizon = torch.linspace(self.t0, self.t1, steps=steps, device=device)
        print(f'predicting samples, input_size: {xs.shape}')

        z0_mean, z0_sigma = self.qz0_net(self.encoder(xs)).chunk(chunks=2, dim=2)
        z0 = z0_mean + z0_sigma.exp() * torch.randn_like(z0_mean)

        assert not torch.isnan(z0).any(), f'z0 latent vector was nan, {z0}'
        latent_and_data = torch.cat((z0[ :, :, :], actions[:, :,  :], xs[:, :, :]), dim=2)
        z_encoded = self.action_encode_net(latent_and_data)
        # merge ensemble
        z_encoded = z_encoded.reshape((z_encoded.shape[0]*z_encoded.shape[1], z_encoded.shape[2]))
        assert not torch.isnan(z_encoded).any(), f'input latent vector was nan, {z_encoded}'
        z_pred = torchsde.sdeint(self, z_encoded, t_horizon, dt=self.dt, bm=bm,
                                 method="reversible_heun")
        z_pred = z_pred.reshape((xs.shape[0], 2,xs.shape[1], -1))
        assert not torch.isnan(
            z_pred).any(), f'some latent vector was nan, {z_pred.shape}, {z_encoded.shape} , {torch.gather(z_encoded, 0, torch.argwhere(torch.isnan(z_pred[-1])))}'
        xs_hat = self.projector(z_pred[:, -1, :, :])

        return xs_hat


class LatentSDEModel:
    def __init__(self, network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=64,
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
        self.ensemble_model = LatentSDE(state_size, state_size, 1, context_size, hidden_size, action_size, self.network_size)
        self.scaler = StandardScaler()

    def plot_gym_results(self, X, Xrec, idx=0, show=False, fname='reconstructions.png'):
        tt = 50
        D = np.ceil(X.shape[1]).astype(int)
        nrows = np.ceil(D).astype(int)
        lag = 0
        plt.figure(2, figsize=(40, 40))
        for i in range(D):
            plt.subplot(nrows, 3, i + 1)
            plt.plot(range(0, tt), X[:tt, i].detach().cpu().numpy(), 'r.-')
            plt.plot(range(lag, tt), Xrec[:tt, i].detach().cpu().numpy(), 'b.-')
        plt.savefig(f'{fname}-{idx}')
        if show is False:
            plt.close()

    def train(self, args, inputs, labels, actions, total_step, batch_size=256, holdout_ratio=0.,
              max_epochs_since_update=5):
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

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        self.scaler.fit(holdout_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        self.scaler.fit(train_actions_inputs)
        train_actions_inputs = self.scaler.transform(train_actions_inputs)
        self.scaler.fit(holdout_actions_inputs)
        holdout_actions_inputs = self.scaler.transform(holdout_actions_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)

        holdout_actions_inputs = torch.from_numpy(holdout_actions_inputs).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])

        holdout_actions_inputs = holdout_actions_inputs[None, :, :].repeat([self.network_size, 1, 1])
        batch_size = train_inputs.shape[0]
        print(f'training model, train_size : {train_inputs.shape}')
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
                holdout_mse_loss, holdout_ensemble_loss = self.ensemble_model.loss(ho_logqp_path, xs_pred, holdout_labels)
                holdout_ensemble_loss = holdout_ensemble_loss.detach().cpu().numpy()

                sorted_loss_idx = np.argsort(holdout_ensemble_loss)
                self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_ensemble_loss)
                if break_train:
                    if total_step % 500 == 0:
                        self.plot_gym_results(holdout_labels[0], xs_pred[0],
                                          fname=f'results/{args.resdir}/train_plt_{total_step}')
                    print(f'training ended epoch no, {epoch}')
                    break


    def batchify(self, data, batch_size):
        no_batches, dim = data.shape
        big_chunk = np.array([], dtype=np.float32)
        flow_over = np.array([], dtype=np.float32)
        for j in range(no_batches):
            if j % batch_size == 0 and (j + batch_size) <= no_batches:
                chunk = data[j:j + batch_size]
                if j == 0:
                    big_chunk = np.array(chunk).reshape((1, chunk.shape[0], chunk.shape[1]))
                else:
                    big_chunk = np.append(big_chunk,
                                          np.array(chunk).reshape((1, chunk.shape[0], chunk.shape[1])), axis=0)
            elif j % batch_size == 0 and (j + batch_size) > no_batches:
                chunk = data[j:j + batch_size]
                flow_over = np.array(chunk).reshape((1, chunk.shape[0], chunk.shape[1]))

        return torch.asarray(big_chunk, dtype=torch.float32).to(device), torch.asarray(flow_over,
                                                                                       dtype=torch.float32).to(device)


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
    def predict(self,args, inputs, actions, batch_size=128, total_step =0):
        assert len(inputs) > 0, f'predict input is empty'
        self.scaler.fit(inputs)
        inputs = torch.asarray(self.scaler.transform(inputs), dtype=torch.float32).repeat([self.network_size, 1, 1]).to(device)
        self.scaler.fit(actions)
        actions = torch.asarray(self.scaler.transform(actions), dtype=torch.float32).repeat([self.network_size, 1, 1]).to(device)
        num_nets, og_batches, og_dim = inputs.shape
        model_op = self.ensemble_model.sample_fromx0(inputs, actions, batch_size)
        assert not torch.isnan(model_op).any(), f'some predicted state vector was nan, halting progress'
        assert model_op.shape[1] == og_batches, f'some predictions were lost, {model_op.shape[1]}, {og_batches}'
        return model_op


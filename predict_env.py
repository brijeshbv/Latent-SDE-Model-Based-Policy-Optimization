import numpy as np

import matplotlib.pyplot as plt

class PredictEnv:
    def __init__(self, model, env_name, model_type):
        self.model = model
        self.env_name = env_name
        self.model_type = model_type

    def _termination_fn(self, env_name, obs, act, next_obs):
        # TODO
        if  env_name == "Hopper-v4":
            # healthy_state_range = (-100.0, 100.0),
            # healthy_z_range = (0.7, float("inf")),
            # healthy_angle_range = (-0.2, 0.2),
            # assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
            #
            # height = next_obs[:, 1]
            # angle = next_obs[:, 2]
            # not_done = np.isfinite(next_obs).all(axis=-1) \
            #            * np.abs(next_obs[:, 2:] < 100).all(axis=-1) \
            #            * (height > .7) \
            #            * (np.abs(angle) < .2)
            #
            # done = ~not_done
            # done = done[:, None]

            z, angle = obs[:, 1], obs[:, 2]
            state = obs[:, 2:]

            min_state, max_state = -100.0, 100.0
            min_z, max_z = 0.7, float("inf")
            min_angle, max_angle = -0.2, 0.2

            healthy_state = np.logical_and((min_state < state).all(axis=1), (state < max_state).all(axis=1))
            healthy_z = np.logical_and((min_z < z), (z < max_z))
            healthy_angle =np.logical_and( (min_angle < angle), (angle < max_angle) )

            is_healthy = np.logical_and(healthy_state, healthy_z, healthy_angle)

            return np.invert(is_healthy).reshape(-1, 1)
        elif env_name == "Walker2d-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 1]
            angle = next_obs[:, 2]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        elif 'walker_' in env_name:
            torso_height =  next_obs[:, -2]
            torso_ang = next_obs[:, -1]
            if 'walker_7' in env_name or 'walker_5' in env_name:
                offset = 0.
            else:
                offset = 0.26
            not_done = (torso_height > 0.8 - offset) \
                       * (torso_height < 2.0 - offset) \
                       * (torso_ang > -1.0) \
                       * (torso_ang < 1.0)
            done = ~not_done
            done = done[:, None]
        elif 'InvertedPendulum-v4' in env_name:
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            notdone = np.isfinite(next_obs).all(axis=-1) \
                      * (np.abs(next_obs[:, 1]) <= .2)
            done = ~notdone

            done = done[:, None]

        return done

    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means, 0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, total_step, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        if self.model_type == 'pytorch':
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        else:
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means[:, :, 1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
        if total_step % 500 == 0:
            self.plt_predictions(ensemble_samples[:,:, 1:], fname=f'results/plt_o/prediction_{total_step}')

        num_models, batch_size, _ = ensemble_model_means.shape
        if self.model_type == 'pytorch':
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        else:
            model_idxes = self.model.random_inds(batch_size)
        batch_idxes = np.arange(0, batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        model_means = ensemble_model_means[model_idxes, batch_idxes]
        model_stds = ensemble_model_stds[model_idxes, batch_idxes]

        #log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:, :1], samples[:, 1:]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:, :1], terminals, model_means[:, 1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:, :1], np.zeros((batch_size, 1)), model_stds[:, 1:]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]
        #'std': return_stds, 'log_prob': log_prob, 'dev': dev
        info = {'mean': return_means }
        return next_obs, rewards, terminals, info

    def plt_predictions(self, X, fname='reconstructions.png'):
        tt = 50
        D = np.ceil(X.shape[2]).astype(int)
        nrows = np.ceil(D).astype(int)
        plt.figure(2, figsize=(40, 40))
        for i in range(D):
            plt.subplot(nrows, 3, i + 1)
            plt.plot(range(0, tt), X[0, :tt, i], 'g.-')
        plt.savefig(f'{fname}')
        plt.close()
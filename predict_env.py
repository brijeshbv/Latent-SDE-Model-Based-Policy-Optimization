import numpy
import numpy as np
import matplotlib
import torch

matplotlib.use('Agg')
from matplotlib import pyplot as plt


class PredictEnv:
    def __init__(self, model, env_name, model_type):
        self.model = model
        self.env_name = env_name
        self.model_type = model_type

    def _termination_fn(self, env_name, obs, act, next_obs):
        # TODO
        if env_name == "Hopper-v4":
            z, angle = obs[:, 1], obs[:, 2]
            state = obs[:, 2:]

            min_state, max_state = -100.0, 100.0
            min_z, max_z = 0.7, float("inf")
            min_angle, max_angle = -0.2, 0.2

            healthy_state = np.logical_and((min_state < state).all(axis=1), (state < max_state).all(axis=1))
            healthy_z = np.logical_and((min_z < z), (z < max_z))
            healthy_angle = np.logical_and((min_angle < angle), (angle < max_angle))

            is_healthy = np.logical_and(healthy_state, healthy_z, healthy_angle)

            return np.invert(is_healthy).reshape(-1, 1)
        elif env_name == "Walker2d-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        elif 'walker_' in env_name:
            torso_height = next_obs[:, -2]
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
            return done
        elif 'InvertedPendulum-v4' in env_name:
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            notdone = np.isfinite(next_obs).all(axis=-1) \
                      * (np.abs(next_obs[:, 1]) <= .2)
            done = ~notdone

            done = done[:, None]
            return done
        elif 'Swimmer-v4' in env_name:
            done = np.zeros_like(obs[:, 0], dtype=bool)
            return done[:, None]

    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (
                k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means, 0).mean(-1)

        return log_prob, stds

    def get_reward(self, env, action, curr_pos, next_pos):
        reward = numpy.zeros_like(curr_pos[:, 0])
        if env == "Hopper-v4":
            reward = (next_pos[:, 0] - curr_pos[:, 0]) / 0.008
            reward += numpy.ones_like(curr_pos[:, 0])
            cost = 1e-3 * np.square(action).sum(axis=1)
            reward = reward - cost
        elif env == "Swimmer-v4":
            ctrl_cost_weight = 0.001
            xy_position_before = curr_pos[:, 0:2]

            xy_position_after = next_pos[:, 0:2]
            xy_velocity = (xy_position_after - xy_position_before) / 0.04
            x_velocity, y_velocity = xy_velocity[:, 0], xy_velocity[:, 1]
            forward_reward = 1 * x_velocity
            control_cost = ctrl_cost_weight * np.sum(np.square(action), axis=1)
            reward = forward_reward - control_cost
        elif env == "InvertedPendulum-v4":
            return numpy.ones_like(curr_pos[:, 0])
        elif env == "LunarLander-v2":
            reward = numpy.zeros_like(curr_pos[:, 0])
            prev_shaping = (
                    -100 * np.sqrt(curr_pos[:, 0] * curr_pos[:, 0] + curr_pos[:, 1] * curr_pos[:, 1])
                    - 100 * np.sqrt(curr_pos[:, 2] * curr_pos[:, 2] + curr_pos[:, 3] * curr_pos[:, 3])
                    - 100 * abs(curr_pos[:, 4])
                    + 10 * curr_pos[:, 6]
                    + 10 * curr_pos[:, 7]
            )
            shaping = (
                    -100 * np.sqrt(next_pos[:, 0] * next_pos[:, 0] + next_pos[:, 1] * next_pos[:, 1])
                    - 100 * np.sqrt(next_pos[:, 2] * next_pos[:, 2] + next_pos[:, 3] * next_pos[:, 3])
                    - 100 * abs(next_pos[:, 4])
                    + 10 * next_pos[:, 6]
                    + 10 * next_pos[:, 7]
            )

            if prev_shaping is not None:
                reward = shaping - prev_shaping
            m_power = (np.clip(action[:, 0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
            assert 0.5 <= m_power <= 1.0
            s_power = np.clip(np.abs(action[:, 1]), 0.5, 1.0)
            assert s_power >= 0.5 and s_power <= 1.0
            reward -= (
                    m_power * 0.30
            )  # less fuel spent is better, about -30 for heuristic landing
            reward -= s_power * 0.03

            if self.game_over or abs(next_pos[0]) >= 1.0:
                reward = -100
            if not self.lander.awake:
                reward = +100
        return reward

    def step(self, args, obs, act, total_step, normalizer):
        global ensemble_model_op, model_idxes, batch_idxes
        if obs.shape[0] > args.rollout_batch_size:
            obs = obs[:args.rollout_batch_size]
            act = act[:args.rollout_batch_size]
            print(f'more obs than batch size, {obs.shape[0]}, batch_size : {args.rollout_batch_size}')
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        if args.model_type == 'bnn':
            inputs = np.concatenate((obs, act), axis=-1)
            ensemble_model_means_bnn, ensemble_model_vars_bnn = self.model.predict(inputs)
            ensemble_model_stds = np.sqrt(ensemble_model_vars_bnn)
            ensemble_model_op = ensemble_model_means_bnn + np.random.normal(
                size=ensemble_model_means_bnn.shape) * ensemble_model_stds
            ensemble_model_op = normalizer.inverse_transform(ensemble_model_op)
            ensemble_model_op[:, :, 1:] += obs
            num_models, batch_size, _ = ensemble_model_op.shape
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        elif args.model_type == 'torchsde':
            ensemble_model_op, ensemble_model_input, ensemble_model_actions, first_pred = self.model.predict(args, obs,
                                                                                                             act,
                                                                                                             args.steps_to_predict,
                                                                                                             total_step,
                                                                                                             normalizer)
            num_models, batch_size, _ = ensemble_model_op.shape
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
            batch_idxes = np.arange(0, batch_size)
            obs = ensemble_model_input[model_idxes, batch_idxes]
            act = ensemble_model_actions[model_idxes, batch_idxes]

        batch_idxes = np.arange(0, batch_size)
        samples = ensemble_model_op[model_idxes, batch_idxes]

        next_obs = samples[:, 1:]
        rewards = samples[:, 0].reshape((samples.shape[0], -1))
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)
        return_means = np.concatenate((samples[:, :], terminals, samples[:, :]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            rewards = rewards[0]
            terminals = terminals[0]
        info = {'mean': return_means, }
        print(f'{args.model_type} is being used for prediction')
        return obs, next_obs, rewards, terminals, act, info

    def plt_predictions(self, X, X_bnn, fname='reconstructions.png'):
        tt = 50
        D = np.ceil(X.shape[2]).astype(int)
        nrows = np.ceil(D).astype(int)
        plt.figure(2, figsize=(40, 40))
        for i in range(D):
            plt.subplot(nrows, 3, i + 1)
            plt.plot(range(0, tt), X[0, -tt:, i], 'g.-')
            plt.plot(range(0, tt), X_bnn[0, -tt:, i], 'r.-')
        plt.savefig(f'{fname}')
        plt.close()

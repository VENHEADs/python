import numpy as np
import torch
from reward_calc import calc_reward
from config import _get_default_config
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = _get_default_config()
N_ACTIONS = config.N_ACTIONS
MAX_CAPACITY = config.MAX_CAPACITY
DAYS_OF_MAX_CAPACITY = config.DAYS_OF_MAX_CAPACITY
ADDITIONAL_REWARD = config.ADDITIONAL_REWARD
REWARD_SCALE = config.REWARD_SCALE
GAMMA = config.gamma


def multiply(value):
    number, power = value
    return number * GAMMA ** (power - 1)


def summa_(value):
    number, power = value
    return number


def give_reward(state, dqn, population_dict, df, df_standard, final_reward, episodes=100):
    pop_dict_local = population_dict.copy()
    state_local = torch.empty_like(state).copy_(state)
    indexes_zero = np.where(state_local.numpy() == 0)[1]
    indexes_zero = indexes_zero[indexes_zero < 5000]
    episode_counter = 0
    number_episodes = min(episodes, len(indexes_zero))
    episodes_indexes = list(np.random.choice(indexes_zero, size=number_episodes, replace=False))
    seq_rewards = []  # to store Gt and index t
    reward_list = [[final_reward, 1]]  # to store reward Rt and index t
    for position in episodes_indexes:  # not allocated families
        mask_local = torch.zeros((1, N_ACTIONS))
        current_row = df[position]  # take not allocated family
        current_row = np.array(current_row[1:N_ACTIONS + 1].tolist() + [current_row[-1]])
        days = current_row[:-1]
        n_members = current_row[-1]
        for n_pos, day_var in enumerate(days[:-1]):
            if pop_dict_local[day_var] + n_members > MAX_CAPACITY:
                mask_local[0, n_pos] = -1 * np.inf
        # blocked = (mask_local == -np.inf).sum().numpy()
        data = torch.Tensor(df_standard[position][1:]).unsqueeze(0)
        nn_state = torch.cat((data, state_local), dim=1)
        # if blocked != N_ACTIONS:
        action, model_output = dqn.select_max_action(nn_state, mask_local)
        array_actions = action.numpy()
        action = action.numpy()[0]

        if action != N_ACTIONS - 1:
            day = current_row[:-1][action]
        else:
            valid_days = np.array(list(map(int, pop_dict_local.values()))) + n_members <= MAX_CAPACITY
            valid_days = np.array(list(range(1, 101)))[valid_days]
            day = np.random.choice(valid_days)
            array_actions = [999]

        pop_dict_local[day] += n_members
        state_local[0, position + day - 1] += n_members / MAX_CAPACITY  # update state
        state_local[0, position] = (day - 50.5) / 29.8

        g_t = np.sum(list(map(multiply, reward_list))) + GAMMA ** (episode_counter + 1) * model_output
        seq_rewards.append([g_t, episode_counter + 1])
        reward, penalty = calc_reward(array_actions, n_members)
        reward /= REWARD_SCALE
        episode_counter += 1
        reward_list.append([reward, 1 + episode_counter])

    return (1 - GAMMA) * np.sum(list(map(multiply, seq_rewards)))

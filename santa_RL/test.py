from data_loader_monte import train_loader
from neural_network_monte_santa import Dqn
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os
from reward_calc import calc_reward
from give_reward_monte import give_reward
from config import _get_default_config

config = _get_default_config()
device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
N_ACTIONS = config.N_ACTIONS
MAX_CAPACITY = config.MAX_CAPACITY
DAYS_OF_MAX_CAPACITY = config.DAYS_OF_MAX_CAPACITY
ADDITIONAL_REWARD = config.ADDITIONAL_REWARD
REWARD_SCALE = config.REWARD_SCALE

try:
    os.remove('rewards.txt')
except:
    pass


def write_to_txt(value: object, name: object) -> object:
    with open(f"{name}.txt", "a") as myfile:
        myfile.write(f"{value}" + '\n')


path_data = ''
dqn = Dqn(config.n_neurons, N_ACTIONS, 0.9)
# dqn.load()
df_standard = np.array(pd.read_csv(path_data + 'family_data_standard_scaled.csv'))
df = np.array(pd.read_csv(path_data + 'family_data.csv'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gamma = 0.9
sub = pd.read_csv('sample_submission.csv')
results = {day: [] for day in range(100)}
min_penalty = 999999
for epoch in tqdm(range(1000)):
    empty_days = []
    c = 0
    total_reward = 0
    total_penalty = 0
    reward = 0
    population_dict = {day: 0 for day in range(1, 101)}
    family_states = torch.zeros((1, 5100))
    for id_n, (data, n) in enumerate(train_loader):
        mask = torch.zeros((1, N_ACTIONS))  # mask if day is full - you can't choose it . zero in the begining
        current_row = df[int(n.numpy())]
        current_row = np.array(current_row[1:N_ACTIONS + 1].tolist() + [current_row[-1]])
        n_members = current_row[-1]
        days = current_row[:-1]
        for n_pos, day_var in enumerate(days[:-1]):   # fill mask with  -inf if you can't choose it
            if population_dict[day_var] + n_members > MAX_CAPACITY:
                mask[0, n_pos] = -1 * np.inf
        # blocked = (mask == -np.inf).sum().numpy()
        data_state = torch.cat((data, family_states), dim=1)
        # if blocked != N_ACTIONS:
        action = dqn.update(data_state, reward, mask, 'train').detach().cpu()
        array_actions = action.numpy()
        selected_action = array_actions[0]
        if selected_action != N_ACTIONS - 1:  # if it is not last pick - take a day
            day = current_row[:-1][selected_action]
        else:  # if it is last pick - it means we choose last random free day
            valid_days = np.array(list(map(int, population_dict.values()))) + n_members <= MAX_CAPACITY
            valid_days = np.array(list(range(1, 101)))[valid_days]
            day = np.random.choice(valid_days)
            array_actions = [999]

        population_dict[day] += n_members  # fill the dict with people who chosen specific day
        reward, penalty = calc_reward(array_actions, n_members)

        family_states[0, int(n[0])] = (day - 50.5) / 29.8
        family_states[0, int(n[0]) + day - 1] += n_members/MAX_CAPACITY
        reward = give_reward(family_states, dqn, population_dict, df, df_standard, reward,
                             config.episdodes_monte)
        total_reward += reward
        total_penalty += penalty

        sub.at[int(n[0]), 'assigned_day'] = day
    print(total_reward, 'reward', total_penalty, 'penalty')
    write_to_txt(str(total_reward) + '   ' + str(epoch) + '   ' + str(total_penalty),  'rewards')

    if abs(total_penalty) < min_penalty:
        dqn.save()
        min_penalty = abs(total_penalty)
        sub.to_csv('test.csv', index=None)

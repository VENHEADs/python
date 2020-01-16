import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from numba import njit
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
root_path = Path(r'')
best_submit_path = Path(r'')
MAX_CHOICE = 5


fpath = root_path / 'family_data.csv'
data = pd.read_csv(fpath, index_col='family_id')
data_choices = data.values

fpath = root_path / 'sample_submission.csv'
submission = pd.read_csv(fpath, index_col='family_id')

dummies = []
for i in range(MAX_CHOICE):
    tmp = pd.get_dummies(data[f'choice_{i}']).values * (data['n_people'].values.reshape(-1, 1))
    dummies.append((
                       np.concatenate([tmp, tmp[:, -1].reshape(-1, 1)], axis=1)
                   ).reshape(5000, 101, 1))
dummies = np.concatenate(dummies, axis=2)
dummies = np.swapaxes(dummies, 1, 2)

penalties = {n: [0, 50, 50 + 9 * n, 100 + 9 * n, 200 + 9 * n, 200 + 18 * n, 300 + 18 * n, 300 + 36 * n, 400 + 36 * n,
                 500 + 36 * n + 199 * n] for n in np.unique(data['n_people'])}

mat = []
for i in range(5000):
    n = data.iloc[i]['n_people']
    mat.append(penalties[n][:MAX_CHOICE])
mat = np.array(mat)


def create_init(initial_sub):
    fam_choices = data
    a = pd.merge(initial_sub, fam_choices, on='family_id')

    initial_choices = []
    for i in range(MAX_CHOICE):
        initial_choices.append(((a[f'choice_{i}'] == a['assigned_day'])).values.reshape(-1, 1))
    initial_choices = np.concatenate(initial_choices, axis=1)
    initial_choices = torch.tensor(
        initial_choices * 10
        , dtype=torch.float32).cuda()
    return initial_choices


initial_sub = pd.read_csv('best_submission.csv')
initial_choices = create_init(initial_sub)

family_sizes = data.n_people.values.astype(np.int8)
cost_dict = {0: [0, 0],
             1: [50, 0],
             2: [50, 9],
             3: [100, 9],
             4: [200, 9],
             5: [200, 18],
             6: [300, 18],
             7: [300, 36],
             8: [400, 36],
             9: [500, 36 + 199],
             10: [500, 36 + 398],
             }


def cost(choice, members, cost_dict):
    x = cost_dict[choice]
    return x[0] + members * x[1]


all_costs = {k: pd.Series([cost(k, x, cost_dict) for x in range(2, 9)], index=range(2, 9)) for k in cost_dict.keys()}
df_all_costs = pd.DataFrame(all_costs)

family_cost_matrix = np.zeros((100, len(family_sizes)))  # Cost for each family for each day.

for i, el in enumerate(family_sizes):
    family_cost_matrix[:, i] += all_costs[10][el]  # populate each day with the max cost
    for j, choice in enumerate(data.drop("n_people", axis=1).values[i, :]):
        family_cost_matrix[choice - 1, i] = all_costs[j][el]


def accounting(today, previous):
    return ((today - 125) / 400) * today ** (.5 + (abs(today - previous) / 50))


acc_costs = np.zeros([176, 176])

for i, x in enumerate(range(125, 300 + 1)):
    for j, y in enumerate(range(125, 300 + 1)):
        acc_costs[i, j] = accounting(x, y)


@njit(fastmath=True)
def cost_function(prediction, family_size, family_cost_matrix, accounting_cost_matrix):
    N_DAYS = 100
    MAX_OCCUPANCY = 300
    MIN_OCCUPANCY = 125
    penalty = 0
    accounting_cost = 0
    max_occ = False

    daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int16)
    for i, (pred, n) in enumerate(zip(prediction, family_size)):
        daily_occupancy[pred - 1] += n
        penalty += family_cost_matrix[pred - 1, i]

    daily_occupancy[-1] = daily_occupancy[-2]
    for day in range(N_DAYS):
        n_next = daily_occupancy[day + 1]
        n = daily_occupancy[day]
        max_occ += MIN_OCCUPANCY > n
        max_occ += MAX_OCCUPANCY < n
        accounting_cost += accounting_cost_matrix[n - MIN_OCCUPANCY, n_next - MIN_OCCUPANCY]
    if max_occ:
        return 1e11
    return penalty + accounting_cost

cost_function(initial_sub['assigned_day'].values, family_sizes, family_cost_matrix, acc_costs)


class Model(nn.Module):
    def __init__(self, mat, dummies):
        super().__init__()
        self.mat = torch.from_numpy(mat).type(torch.int16).cuda()
        self.dummies = torch.from_numpy(dummies).type(torch.float32).cuda()
        self.weight = torch.nn.Parameter(data=torch.Tensor(5000, MAX_CHOICE).type(torch.float32).cuda()
                                         , requires_grad=True)
        self.fc1 = nn.Linear(5000, 2500)
        self.fc2 = nn.Linear(2500, MAX_CHOICE)
        self.weight.data.copy_(initial_choices)

    def forward(self):
        prob = F.softmax(self.weight, dim=1)

        x = (prob * self.mat).sum()

        daily_occupancy = torch.zeros(101, dtype=torch.float32).cuda()
        for i in range(MAX_CHOICE):
            daily_occupancy += (prob[:, i] @ self.dummies[:, i, :])

        diff = torch.abs(daily_occupancy[:-1] - daily_occupancy[1:])
        daily_occupancy = daily_occupancy[:-1]
        y = (
                torch.relu(daily_occupancy - 125.0) / 400.0 * daily_occupancy ** (0.5 + diff / 50.0)
        ).sum()

        v = ((torch.relu(125 - daily_occupancy)) ** 2 + (torch.relu(daily_occupancy - 300)) ** 2).sum()

        entropy_loss = -1.0 * (prob * F.log_softmax(self.weight, dim=1)).sum()
        return x, y, v * 10000, entropy_loss


model = Model(mat, dummies)
best_score = 10e10
best_pos = None
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

for epoch in tqdm.tqdm_notebook(range(1_001)):
    optimizer.zero_grad()
    x, y, v, ent = model()
    loss = x + y + v + 0 * ent
    loss.backward()
    optimizer.step()

    pos = model.weight.argmax(1).cpu().numpy()
    pred = []
    for i in range(5000):
        pred.append(data_choices[i, pos[i]])
    pred = np.array(pred)
    score = cost_function(pred, family_sizes, family_cost_matrix, acc_costs)
    if (score < best_score):
        best_score = score
        best_pos = pred
        print(best_score)
        submission['assigned_day'] = best_pos
        submission.to_csv(f'submission.csv')
    if epoch % 1000 == 0:
        x = np.round(x.item(), 1)
        y = np.round(y.item(), 1)
        print(f'{epoch}\t{x}\t{y}    \t{np.round(score, 2)}')

prev_best_score = best_score
coef = 1
count_failures = 0
for _ in range(10_000):

    initial_sub = pd.read_csv('submission.csv')
    initial_choices = create_init(initial_sub)

    model = Model(mat, dummies)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in tqdm.tqdm_notebook(range(1_001)):
        optimizer.zero_grad()
        x, y, v, ent = model()
        loss = x + coef * y + v + 0 * ent
        loss.backward()
        optimizer.step()

        pos = model.weight.argmax(1).cpu().numpy()
        pred = []
        for i in range(5000):
            pred.append(data_choices[i, pos[i]])
        pred = np.array(pred)
        score = cost_function(pred, family_sizes, family_cost_matrix, acc_costs)
        if (score < best_score):
            best_score = score
            best_pos = pred
            print(best_score)
            submission['assigned_day'] = best_pos
            submission.to_csv(f'submission.csv')
        if (epoch % 1000 == 0) and epoch != 0:
            x = np.round(x.item(), 1)
            y = np.round(y.item(), 1)
            print(f'{epoch}\t{x}\t{y}    \t{np.round(score, 2)}')
    if best_score == prev_best_score:
        count_failures += 1
        if count_failures > 10:
            break
        coef = coef * 1.05
    #         break
    else:
        prev_best_score = best_score
        count_failures = 0
        coef = 1

prev_best_score = best_score
coef = 1
count_failures = 0
for _ in range(10_000):

    initial_sub = pd.read_csv('submission.csv')
    initial_choices = create_init(initial_sub)

    model = Model(mat, dummies)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in tqdm.tqdm_notebook(range(1_001)):
        optimizer.zero_grad()
        x, y, v, ent = model()
        loss = coef * x + y + v + 10 * ent
        loss.backward()
        optimizer.step()

        pos = model.weight.argmax(1).cpu().numpy()
        pred = []
        for i in range(5000):
            pred.append(data_choices[i, pos[i]])
        pred = np.array(pred)
        score = cost_function(pred, family_sizes, family_cost_matrix, acc_costs)
        if (score < best_score):
            best_score = score
            best_pos = pred
            print(best_score)
            submission['assigned_day'] = best_pos
            submission.to_csv(f'submission.csv')
        if (epoch % 1000 == 0) and epoch != 0:
            x = np.round(x.item(), 1)
            y = np.round(y.item(), 1)
            print(f'{epoch}\t{x}\t{y}    \t{np.round(score, 2)}')
    if best_score == prev_best_score:
        count_failures += 1
        if count_failures > 10:
            break
        coef = coef * 1.01
    #         break
    else:
        prev_best_score = best_score
        count_failures = 0
        coef = 1

submission['assigned_day'] = best_pos
submission.to_csv(f'submission.csv')
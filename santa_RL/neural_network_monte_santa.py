import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import _get_default_config

config = _get_default_config()
INPUT_NEURONS = 5111
BATCH_SIZE = config.batch_size


class Network(nn.Module):

    def __init__(self, n_neurons, nb_action, n_filters=64):
        super(Network, self).__init__()
        self.linear_units = n_neurons
        self.nb_action = nb_action

        self.conv1 = nn.Conv1d(1, n_filters, kernel_size=3)
        self.conv2 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=3)
        self.conv3 = nn.Conv1d(n_filters * 2, n_filters * 3, kernel_size=3)
        self.conv_to_fc_1 = nn.Linear(n_filters * 3 * 5, 128)
        self.conv_to_fc_2 = nn.Linear(128, nb_action)

        self.fc1 = nn.Linear(INPUT_NEURONS, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons//2)
        self.fc3 = nn.Linear(n_neurons//2, n_neurons//4)
        self.fc4 = nn.Linear(n_neurons//4, n_neurons//8)
        self.fc5 = nn.Linear(n_neurons // 8, nb_action)

        self.bn1 = nn.BatchNorm1d(n_neurons)
        self.bn2 = nn.BatchNorm1d(n_neurons//2)
        self.bn3 = nn.BatchNorm1d(n_neurons//4)

    def forward(self, state, mode='predict'):
        # x = F.elu(self.conv1(state))
        # x = F.elu(self.conv2(x))
        # x = F.elu(self.conv3(x))
        # x = F.elu(self.conv_to_fc_1(torch.flatten(x)))
        # q_values = self.conv_to_fc_2(x)

        # if mode == 'train':
        #     self.train()
        #     x = F.elu(self.fc1(state.cuda()))
        #     x = F.elu(self.fc2(x))
        #     x = F.elu(self.fc3(x))
        #     q_values = self.fc4(x)
        #     return q_values
        #
        # self.eval()
        x = F.elu(self.fc1(state.cuda()))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        q_values = self.fc5(x)
        return q_values

        # x = torch.relu(self.fc1(state.cuda()))
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        # q_values = self.fc4(x)
        # return q_values


# Implementing Experience Replay

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: torch.cat(x, 0), samples)


# Implementing Deep Q-Learning

class Dqn(object):

    def __init__(self, n_neurons, nb_action, gamma):
        self.gamma = gamma
        self.model = Network(n_neurons, nb_action).cuda()
        self.memory = ReplayMemory(capacity=100000)
        self.optimizer = optim.Adam(params=self.model.parameters())
        self.last_state = torch.Tensor(INPUT_NEURONS).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        self.rewards = []

    def clear_reward_list(self):
        self.rewards = []

    def select_max_action(self, state, mask):
        output = (self.model(state) + mask.cuda())
        return torch.max(output, dim=1)[1].detach().cpu(), output.detach().cpu().max(1)[0].numpy()[0]

    def select_action(self, state, mask):
        probs = F.softmax(self.model(state) + mask.cuda(), dim=1)
        action = probs.multinomial(len(probs))
        return action[0]  # .item()

    def learn(self, batch_states, batch_actions, batch_rewards, batch_next_states, mode):
        batch_outputs = self.model(batch_states.cuda(), mode).gather(1, batch_actions.cuda().unsqueeze(1)).squeeze(1)
        #         batch_next_outputs = self.model(batch_next_states).detach().max(1)[0]
        batch_targets = batch_rewards  # + self.gamma * batch_next_outputs
        td_loss = F.smooth_l1_loss(batch_outputs.cuda(), batch_targets.cuda())
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

    def update(self, new_state, new_reward, mask, mode='train'):
        new_state = torch.Tensor(new_state).float()
        self.memory.push(
            (self.last_state, torch.LongTensor([int(self.last_action)]),
             torch.Tensor([self.last_reward]), new_state))

        new_action = self.select_action(new_state.cuda(), mask.cuda())
        if len(self.memory.memory) > BATCH_SIZE:
            batch_states, batch_actions, batch_rewards, batch_next_states = self.memory.sample(BATCH_SIZE)
            self.learn(batch_states, batch_actions, batch_rewards, batch_next_states, mode)
        self.last_state = new_state
        self.last_action = new_action
        self.last_reward = new_reward
        return new_action

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")

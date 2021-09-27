import torch
import torch.nn as nn
import torch.nn.functional as F


# with preprocessed meaningful features
class WorldPredictor(nn.Module):
    def __init__(self, input, batch_size): 
        super(WorldPredictor, self).__init__()
        
        # memory of last input states
        self.last_state = torch.zeros(batch_size, input+1)
        # transition layers
        # Take in previous state s_{t-1} and action a_{t-1} and predicts the next state s_t , represents p(s_t|s_{t-1}, a_{t-1})
        self.transition = nn.Sequential(
            nn.Linear((input + 1)*2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, input),
        )

        # reward prediction layer
        # Take in state s_{t} and predicts reward r_{t}
        self.reward = nn.Sequential(
            nn.Linear(input, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    # gets last state together concat with action
    # predicts next state 
    # then predicts its reward
    def forward(self, last_state, action):
        lsa = torch.cat((last_state, action), dim=1)
        # predict next state with given state and prior state from memory
        state = self.transition(torch.cat((lsa, self.last_state), dim=1))   
        self.last_state = lsa     
        # predict reward with given state
        reward = self.reward(state)
        return state, reward


# policy model
class Policy(nn.Module):
    def __init__(self, input, hidden, actions): 
        super(Policy, self).__init__()
        print("Policy net has", input, "input nodes,", hidden, "hidden nodes and", actions, "output nodes")
        self.h = nn.Linear(input, hidden)
        self.out = nn.Linear(hidden, actions)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.h(x))
        return F.softmax(self.out(x), dim=1)
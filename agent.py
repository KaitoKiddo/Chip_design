import os
import numpy as np
import torch 
import torch.optim as optim
from torch.autograd import Variable
from model import Actor,Critic, BasicBlock
from memory import PPOMemory
import random
import datetime
class PPO:
    def __init__(self, action_dim, cfg):
        self.env = cfg.env
        self.gamma = cfg.gamma
        self.policy_clip = cfg.policy_clip
        self.n_epochs = cfg.n_epochs
        self.gae_lambda = cfg.gae_lambda
        self.device = cfg.device
        self.actor = Actor(action_dim, BasicBlock).to(self.device)
        self.critic = Critic(BasicBlock).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PPOMemory(cfg.batch_size)
        self.loss = 0

    def choose_action(self, state, mask):
        # st = datetime.datetime.now()
        state = torch.tensor(state, dtype=torch.float)
        state = Variable(torch.unsqueeze(state, dim=0).float(), requires_grad=False).to(self.device)
        # et = datetime.datetime.now()
        dist = self.actor(state)
        # print(dist.probs)
        value = self.critic(state)
        
        # print(et-st)
        while True:
            rand = random.random()
            if rand < 0.05:
                action = random.randint(0, 624)
                action = torch.tensor(action).to(self.device)
            else:
                action = dist.sample()
            # print(action)
            probs = torch.squeeze(dist.log_prob(action)).item()
            # print(probs)
            action = torch.squeeze(action).item()
            
            if mask[action] != 1:
                break
        value = torch.squeeze(value).item()

        # print(action)
        mask[action] = 1
        return action, probs, value

    def update(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.sample()
            # print(reward_arr)
            values = vals_arr
            ### compute advantage ###
            advantage = np.zeros(len(reward_arr), dtype=np.float)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)
            ### SGD ###
            values = torch.tensor(values).to(self.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5*critic_loss
                self.loss = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()  
    # def save(self,path):
    #     actor_checkpoint = os.path.join(path, str(self.env)+'_actor.pt')
    #     critic_checkpoint= os.path.join(path, str(self.env)+'_critic.pt')
    #     torch.save(self.actor.state_dict(), actor_checkpoint)
    #     torch.save(self.critic.state_dict(), critic_checkpoint)
    # def load(self,path):
    #     actor_checkpoint = os.path.join(path, str(self.env)+'_actor.pt')
    #     critic_checkpoint= os.path.join(path, str(self.env)+'_critic.pt')
    #     self.actor.load_state_dict(torch.load(actor_checkpoint))
    #     self.critic.load_state_dict(torch.load(critic_checkpoint))



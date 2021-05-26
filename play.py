import sys,os
curr_path = os.path.dirname(__file__)
parent_path=os.path.dirname(curr_path) 
sys.path.append(parent_path) # add current terminal path to sys.path
import env
import numpy as np
import torch
import datetime
from agent import PPO


SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间
SAVED_MODEL_PATH = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"+SEQUENCE+'/' # 生成保存的模型路径
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"): # 检测是否存在文件夹
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/")
if not os.path.exists(SAVED_MODEL_PATH): # 检测是否存在文件夹
    os.mkdir(SAVED_MODEL_PATH)
RESULT_PATH = os.path.split(os.path.abspath(__file__))[0]+"/results/"+SEQUENCE+'/' # 存储reward的路径
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/results/"): # 检测是否存在文件夹
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/results/")
if not os.path.exists(RESULT_PATH): # 检测是否存在文件夹
    os.mkdir(RESULT_PATH)

class PPOConfig:
    def __init__(self) -> None:
        self.env = env.Env()
        self.batch_size = 64
        self.gamma = 0.99
        self.n_epochs = 4
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.gae_lambda = 0.95
        self.policy_clip = 0.2
        self.hidden_dim = 1024
        self.update_fre = 1162  # frequency of agent update
        self.train_eps = 2000  # max training episodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # check gpu
        
def play(cfg, env, agent):
    
    rewards= []
    
    running_steps = 0
    
    for i_episode in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        best_reward = 0
        reward_index = 0
        ma_rewards = []

        while not done:
            observation = state
            action, prob, val = agent.choose_action(observation)
            state_, reward, done = env.step(action)
            # print(state_, reward, action)
            running_steps += 1
            ep_reward = reward
            agent.memory.push(observation, action, prob, val, 0, done)
            if running_steps % cfg.update_fre == 0:
                agent.update()
            state = state_
            reward_index = (running_steps-1) % cfg.update_fre
        
        # 更新前面的reward，往前取581个
        update_reward(agent.memory, reward_index, ep_reward)
        
        rewards.append(ep_reward)

        # 统计时使用的reward数据
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)

        avg_reward = np.mean(rewards[-100:])

        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(path=SAVED_MODEL_PATH)
        print('Episode:{}/{}, Reward:{:.1f}, avg reward:{:.1f}, Done:{}'.format(i_episode+1,cfg.train_eps,ep_reward,avg_reward,done))

        action_space = env.get_action_space()
    return rewards, ma_rewards, action_space

def update_reward(memory, reward_index, reward):
    if reward_index == 580:
        for i in range(reward_index+1):
            memory.rewards[i] = reward
    else:
        for i in range(581, reward_index+1):
            memory.rewards[i] = reward

if __name__ == '__main__':

    cfg = PPOConfig()
    env = cfg.env
    
    # state预处理得到observation的dim传到PPO中
    state = env.reset()
    state_dim = len(state)
    print(state_dim)

    action_dim = len(env.get_action_space())
    print(action_dim)

    agent = PPO(state_dim, action_dim, cfg)
    rewards, ma_rewards, action_space = play(cfg, env, agent)
    print(rewards)
import numpy as np
import pandas as pd
import math
class Port_set:
    def __init__(self,mother_port,mother_die_set, daughter_port,daughter_die_set):
        self.mother_port = mother_port
        self.mother_die_set = mother_die_set
        self.daughter_port = daughter_port
        self.daughter_die_set = daughter_die_set
class Env():
    
    def __init__(self):
        self.total_reward_list=[]
        self.lowest_reward = 999999
        self.count_581 = 0
        self.HB_centre_points = np.zeros([625,2])
        self.HB_upper_left_points = np.zeros([625,2])
        self.port_set_list=[]
        self.reward = 0
        self.action_space = np.zeros(625)
        self.count = 0      # 计数，每调用一次setp（） +1
        self.total_reward = 0       # 累计reward
        top_connect = pd.read_csv('top_conn' ,header=None, delimiter=r"\s+")
        mother_die = pd.read_csv('mother_die.port_conn.xy' ,header=None, delimiter=r"\s+")
        daughter_die = pd.read_csv('daughter_die.port_conn.xy' ,header=None, delimiter=r"\s+")
        mother_die = mother_die.values
        top_connect = top_connect.values
        daughter_die = daughter_die.values
        for i in range(0,top_connect.shape[0]):
            top_connect[i,0] = top_connect[i,0][13:]
            top_connect[i,1] = top_connect[i,1][11:]
        for i in range(0,mother_die.shape[0]):
            mother_die[i,2] = mother_die[i,2][1:]
            mother_die[i,3] = mother_die[i,3][:-1]
        for i in range(0,daughter_die.shape[0]):
            daughter_die[i,2] = daughter_die[i,2][1:]
            daughter_die[i,3] = daughter_die[i,3][:-1]   
        for i in range(0,top_connect[:,1].shape[0]):
            mother_die_set = np.array([['#','#','#']])
            daughter_die_set = np.array([['#','#','#']])
            for j in range(0,mother_die[:,1].shape[0]):
                if mother_die[j,0] == top_connect[i,1]:
                    mother_die_set = np.r_[mother_die_set,[[mother_die[j,1],mother_die[j,2],mother_die[j,3]]]]
            for k in range(0,daughter_die[:,1].shape[0]):
                if daughter_die[k,0] == top_connect[i,0]:
                    daughter_die_set = np.r_[daughter_die_set,[[daughter_die[k,1],daughter_die[k,2],daughter_die[k,3]]]]
            if mother_die_set.shape[0] != 1:
                mother_die_set =np.delete(mother_die_set, 0, 0) 
            if daughter_die_set.shape[0] != 1: 
                daughter_die_set =np.delete(daughter_die_set, 0, 0) 
            port_set1 = Port_set(top_connect[i,0],mother_die_set,top_connect[i,1],daughter_die_set)
            self.port_set_list.append(port_set1)
        self.state =[]
        
        for i in range(0,10):
            action_position = np.zeros([25,25])
            self.state.append(action_position)
        for m in range(0,625):      # 计算HB坐标
            self.HB_centre_points[m,0] = 4*(2*(m%25)+1)
            self.HB_centre_points[m,1] = 4*(2*math.floor(m/25)+1)
            self.HB_upper_left_points[m,0] = self.HB_centre_points[m,0]-1
            self.HB_upper_left_points[m,1] = self.HB_centre_points[m,1]+1
        
            
    def step(self,action):
        done = False
        if self.count < len(self.port_set_list):     # 判断action次数
            self.action_space[action] = self.count + 1
            self.total_reward,reward_= self.calculate_reward(action,self.count)
            self.total_reward = -self.total_reward
            self.reward = 0
            x=math.floor(action/25)
            y=action%25
            if self.count == 0:
                action_position = np.zeros([25,25])
                action_position[x,y] = 1
                self.state[self.count] = action_position
            elif self.count < 10:
                action_position = np.zeros([25,25])
                for i in range(0,25):
                    for j in range(0,25):
                        action_position[i,j] = self.state[self.count-1][i,j]
                action_position[x,y] = 1 
                self.state[self.count] = action_position
            else:
                action_position = np.zeros([25,25])
                for k in range(0,9):
                    for i in range(0,25):
                        for j in range(0,25):                    
                            self.state[k][i,j] = self.state[k+1][i,j]
                for i in range(0,25):
                    for j in range(0,25):
                        action_position[i,j] = self.state[8][i,j]
                action_position[x,y] = 1
                self.state[9] = action_position
            self.count = self.count + 1
            self.state_ = self.state

            if self.count == len(self.port_set_list):
                if -self.total_reward < self.lowest_reward:
                    self.lowest_reward = -self.total_reward
                self.total_reward_list.append(self.total_reward)
                done = True
                if self.count_581 == 0:
                    self.reward= 1
                else:
                    if -self.total_reward<self.lowest_reward:
                        self.reward = 1
                    else:
                        self.reward = 0

                self.count_581 = self.count_581+1
        return self.state_, self.reward,done

    def reset(self):
        self.count = 0
        self.total_reward = 0
        self.reward = 0
        self.action_space = np.zeros(625)
        
        for i in range(0,10):
            action_position = np.zeros([25,25])
            self.state[i]= action_position
        return self.state
        
    def get_total_reward(self):
        return self.total_reward
    
    def calculate_reward(self,action,count):      # 计算HB和对应端口集的reward
        reward = -self.total_reward
        reward1 = 0
        if self.port_set_list[count].daughter_die_set[0,0] != '#':      # 如果daughter找不到top对应端口，视为无连接reward为0
            daughter_die_set_x = self.port_set_list[count].daughter_die_set[:,1].astype('float64')
            daughter_die_set_y =200.07-self.port_set_list[count].daughter_die_set[:,2].astype('float64')     # 翻转，daughter本地左边转化为全局坐标
            daughter_die_points=np.zeros([len(self.port_set_list[count].daughter_die_set[:,1]),2])
            daughter_die_points[:,0] = daughter_die_set_x
            daughter_die_points[:,1] = daughter_die_set_y
            for i in range(0,daughter_die_points.shape[0]):
                 reward = reward + math.sqrt(math.pow((self.HB_centre_points[action,0]-daughter_die_points[i,0]),2)+math.pow((self.HB_centre_points[action,1]-daughter_die_points[i,1]),2))
                 reward1 = reward1 + math.sqrt(math.pow((self.HB_centre_points[action,0]-daughter_die_points[i,0]),2)+math.pow((self.HB_centre_points[action,1]-daughter_die_points[i,1]),2))  
        if self.port_set_list[count].mother_die_set[0,0] != '#':
            mother_die_set_x = self.port_set_list[count].mother_die_set[:,1].astype('float64')
            mother_die_set_y = self.port_set_list[count].mother_die_set[:,2].astype('float64')
            mother_die_points= np.zeros([len(self.port_set_list[count].mother_die_set[:,1]),2])
            mother_die_points[:,0] = mother_die_set_x
            mother_die_points[:,1] = mother_die_set_y
            for j in range(0,mother_die_points.shape[0]):
                 reward = reward + math.sqrt(math.pow((self.HB_centre_points[action,0]-mother_die_points[j,0]),2)+math.pow((self.HB_centre_points[action,1]-mother_die_points[j,1]),2))
                 reward1 = reward1 + math.sqrt(math.pow((self.HB_centre_points[action,0]-mother_die_points[j,0]),2)+math.pow((self.HB_centre_points[action,1]-mother_die_points[j,1]),2))
        return reward,reward1
    

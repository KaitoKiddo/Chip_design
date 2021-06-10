import numpy as np
import pandas as pd
import math
import copy
class Data:
    def __init__(self, data=None):
        self.data = data
        self.next = None


class LinkedListTail:
    def __init__(self):
        self.head = None
        self.tail = None

    def add_head(self, data):
        new_node = Data(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
            return 'New value added'
        new_node.next = self.head
        self.head = new_node
        return 'New value added'

    def add_tail(self, data):
        new_node = Data(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
            return 'New value added'
        self.tail.next = new_node
        self.tail = new_node
        return 'New value added'

    def remove_head(self):
        if self.head is None:
            return 'No value to remove'
        gone = self.head
        self.head = self.head.next
        return gone.data, 'removed'

    def remove_tail(self):
        if self.head is None:
            return 'No value to remove'
        gone = self.head
        previous = self.head
        while gone.next is not None:
            previous = gone
            gone = gone.next
        previous.next = None
        self.tail = previous
        return gone.data, 'removed'

    def remove(self, remove):
        if self.head is None:
            return 'No value to remove'
        if self.head.data == remove:
            self.head = self.head.next
            return remove, 'removed'
        gone = self.head
        previous = self.head
        while gone.next is not None or gone.data == remove:
            if gone.data == remove and gone.next is None:
                self.tail = previous
                previous.next = None
                return gone.data, 'removed'
            if gone.data == remove:
                previous.next = previous.next.next
                return remove, 'removed'
            previous = gone
            gone = gone.next
        return 'No value to remove'

    def search(self, seek):
        if self.head.data == seek:
            return True, f'{seek} is included'
        current = self.head
        while current.next is not None:
            if current.data == seek:
                return True, f'{seek} is included'
            current = current.next
        if current.data == seek:
            return True, f'{seek} is included'
        return False, f'{seek} not included'
    def get_i(self,i):
        current = self.head
        for k in range(0,i-1):
             current = current.next
        return current.data
    def seek(self, search):
        elements = self.display()
        for element in elements:
            if search == element:
                return True, f'{search} is included'
        return False, f'{search} not included'
        
    def display(self):
        elements = []
        current = self.head
        while current is not None:
            elements.append(current.data)
            current = current.next
        return elements

    def show_ht(self):
        return f'The beginning of the List is {self.head.data} and the end is {self.tail.data}'

    def clear_all(self):
        self.head = None
        self.tail = None
        return 'All values removed'
    def insert(self,data,i):
        current = self.head
        for k in range(0,i-1):
            current = current.next
        new_node = Data(data)

        new_node.next = current.next
        current.next = new_node
        self.remove_tail()
        if i == 9:
            self.tail = new_node

        return 'New value added'
class Queue:
    def __init__(self):
        self.queue = LinkedListTail()

    def push(self, data):
        return self.queue.add_tail(data)

    def pop(self):
        return self.queue.remove_head()

    def peak_head(self):
        return self.queue.head.data
    def peak_tail(self):
        return self.queue.tail.data

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

            
        for m in range(0,625):      # 计算HB坐标
            self.HB_centre_points[m,0] = 4*(2*(m%25)+1)
            self.HB_centre_points[m,1] = 4*(2*math.floor(m/25)+1)
            self.HB_upper_left_points[m,0] = self.HB_centre_points[m,0]-1
            self.HB_upper_left_points[m,1] = self.HB_centre_points[m,1]+1
        self.queue = Queue()
        for i in range(0,10):
            action_position = np.zeros([25,25])
            self.queue.push(action_position)
        self.state =  self.queue.queue.display()
            
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
                self.queue.queue.add_head(action_position)
                self.queue.queue.remove_tail()
                self.state = self.queue.queue.display()
            elif self.count < 10:
                action_position = copy.deepcopy(self.queue.queue.get_i(self.count))
                action_position[x,y] = 1 
                self.queue.queue.insert(action_position,self.count)
                self.state = self.queue.queue.display()
            else:
                action_position = copy.deepcopy(self.queue.peak_tail())
                self.queue.pop()
                action_position[x,y] = 1
                self.queue.push(action_position)
                self.state = self.queue.queue.display()
            self.count = self.count + 1
            self.state_ = self.state

            if self.count == len(self.port_set_list):

                self.total_reward_list.append(self.total_reward)
                done = True
                if -self.total_reward<self.lowest_reward:
                    self.reward = 1
                    self.lowest_reward = -self.total_reward
                else:
                    self.reward = 0

                    
                self.count_581 = self.count_581+1
        return self.state_, self.reward,done

    def reset(self):
        self.count = 0
        self.total_reward = 0
        self.reward = 0
        self.queue.queue.clear_all()
        for i in range(0,10):
            action_position = np.zeros([25,25])
            self.queue.push(action_position)
        self.state =  self.queue.queue.display()
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
    

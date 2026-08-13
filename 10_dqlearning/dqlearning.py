"""
deep q learning
cartpole
"""
import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
import matplotlib
from collections import namedtuple,deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



env = gym.make("CartPole-v1",render_mode="human")
device = "cpu"

Transition = namedtuple("Transition",
                        ("State","action","next_state","reward"))


# %% replay memory oluşturma
class ReplayMemory(object):
    def __init__(self,capacity):
        self.memory = deque([],maxlen=capacity)#kapasiteye göre memoryi düzenler
    def push(self,*args): #save a transition
        self.memory.append(Transition(*args))
    def sample(self,batch_size):#rastgele sample seç
        return random.sample(self.memory,batch_size)
    def __len__(self):#memory uzunluğu
        return len(self.memory)

# %% dql modeli oluşturma
class DQN(nn.Module):
    def __init__(self,n_observations,n_actions):
        super(DQN,self).__init__()
        self.layer1 = nn.Linear(n_observations,128)
        self.layer2 = nn.Linear(128,128)
        self.layer3 = nn.Linear(128,n_actions)
    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# %% hipermatre ve yardımcı fonksiyonlar

batch_size = 128
gamma = 0.99 #discount factor
eps_start = 0.9
eps_end = 0.05
eps_decay = 1000
tau = 0.005 #update rate of targets network
lr = 1e-4

n_actions = env.action_space.n #type: ignore # action

state, info = env.reset() #initial state
n_observations = len(state)

policy_net = DQN(n_observations,n_actions).to(device)
target_net = DQN(n_observations,n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(),lr=lr)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end)*math.exp(-1*steps_done/eps_decay)
    steps_done += 1

    #eğer sample eps_threshold'dan bütükse ajan neural network ile action seçer , değilse rastgele action seçilir.
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1,-1)
    else:
        return torch.tensor([[env.action_space.sample()]],device=device,dtype=torch.long)


episode_durations = []
def plot_duration(show_result = False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations,dtype = torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Traning..")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())

    if len(durations_t) > 100:
        means = durations_t.unfold(0,100,1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99),means))
        plt.plot(means.numpy())

    plt.pause(0.001)


def optimize_model():
    #hafızada yeterli sayıda deneyim varmı yok mu kontrol et ,yoksa fonksiyondan çık
    if len(memory) < batch_size:
        return
    #hafızadan rastgele bir grup deneyim
    transitions = memory.sample(batch_size)

    #ayırma işlemi
    batch = Transition(*zip(*transitions))

    #sonraki durumları none olmayan bir boolean maskesi oluşturur.
    non_final_mask = torch.tensor(tuple(map(lambda s : s is not None,batch.next_state)),device=device,dtype=torch.bool)

    #terminal olmayan tüm durumları tek tensor olarak birleştirme
    non_final_next_state = torch.cat([s for s in batch.next_state if s is not None])

    #grup içindeki state, action ve reward birleştirilir
    state_batch = torch.cat(batch.State)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1,action_batch)

    next_state_values = torch.zeros(batch_size,device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_state).max(1).values

    expected_state_actions_values = (next_state_values*gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,expected_state_actions_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(),100)
    optimizer.step()



#%% model eğitimi ve sonuçların değerlendirilmesi

num_episode = 300

for i_episode in range(num_episode):
    #reset env
    state,info = env.reset()
    state = torch.tensor(state, dtype = torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state) # action seç

        observation, reward, terminated,truncated, _ =  env.step(action.item())
        reward = torch.tensor([reward],device=device)
        done = terminated or truncated #kaybetme durumu
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation,dtype=torch.float32,device=device).unsqueeze(0)

        #transitionsları memory'da depolama
        memory.push(state,action,next_state,reward)      

        #state güncelle
        state = next_state

        #traning
        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
            target_net.load_state_dict(target_net_state_dict)
        if done:
            episode_durations.append(t+1)
            plot_duration()
            break
print("Done")
plot_duration(show_result=True)
plt.ioff()
plt.show()




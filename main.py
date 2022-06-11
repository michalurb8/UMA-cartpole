import gym
import time
from Qlearning import Qlearning

def manual_policy(cpos, cvel, pang, pvel): #q learning here -> based on observations, decide which action to take
    action = 0
    eps = 0.1
    if abs(pvel) < eps:
        action = int(pang > 0)
    else:
        action = int(pvel > 0)
    assert action in [0,1], "Return action must be 0 or 1"
    return action

env = gym.make("CartPole-v1")
env.reset()
obs, reward, done, info = env.step(0)
for _ in range(80):
    obs, reward, done, info = env.step(manual_policy(*obs))
    env.render()
    # if done == True:
    #     break
    time.sleep(0.01)
env.close()

env = gym.make("CartPole-v1")
env.reset()
ql = Qlearning()
obs, reward, done, info = env.step(0)
for _ in range(80):
    action = ql.step(obs, reward, done, info)
    obs, reward, done, info = env.step(action)
    env.render()
    if done == True:
        break
    time.sleep(0.01)
env.close()
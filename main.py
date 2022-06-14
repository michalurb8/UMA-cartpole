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

# n_iterations = 80
# env = gym.make("CartPole-v1")
# obs, _ = env.reset()
# for _ in range(n_iterations):
#     action = manual_policy(*obs)
#     obs, reward, done, _ = env.step(action)

#     env.render()
#     if done == True:
#         break
#     time.sleep(0.01)
# env.close()

n_iterations = 10000
env = gym.make("CartPole-v1")
ql = Qlearning()
for iter in range(n_iterations):
    print(iter)
    prevObs = env.reset()
    done = False

    while done == False:
        action = ql.getAction(prevObs, iter)
        newObs, reward, done, _ = env.step(action)
        ql.update(reward, prevObs, newObs, iter, action)
        prevObs = newObs

        if iter > n_iterations*0.9:
            env.render()
            time.sleep(0.01)
        if done == True:
            break
env.close()
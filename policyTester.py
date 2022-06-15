import time
import gym 
from numpy import mean
from Qlearning import Qlearning
import matplotlib.pyplot as plt

def manual_policy(cpos, cvel, pang, pvel): #q learning here -> based on observations, decide which action to take
    action = 0
    eps = 0.1
    if abs(pvel) < eps:
        action = int(pang > 0)
    else:
        action = int(pvel > 0)
    assert action in [0,1], "Return action must be 0 or 1"
    return action

def testQLearning(n_iterations: int = 10**4):
    env = gym.make("CartPole-v1")
    ql = Qlearning()
    survivalData = []
    for iter in range(n_iterations):
        print(iter)
        prevObs = env.reset()
        done = False

        ticksSurvived = 0
        while done == False:
            action = ql.getAction(prevObs, iter)
            newObs, reward, done, info = env.step(action)
            ql.update(reward, prevObs, newObs, iter, action)
            prevObs = newObs

            if done == True:
                break
            ticksSurvived +=1
        survivalData.append(ticksSurvived)
    env.close()
    window = 50
    averagedData = [mean(survivalData[ind:ind+window]) for ind in range(len(survivalData)-window+1)]
    plt.xlabel("Liczba iteracji algorytmu QLearning", fontsize=20)
    plt.ylabel("Liczba klatek,\nprzez które wagonik\nutrzymał wahadło w pionie", fontsize=20)
    plt.plot(averagedData)
    plt.show()

def testManual(n_iterations: int = 10**4):
    env = gym.make("CartPole-v1")
    survivalData = []
    for iter in range(n_iterations):
        print(iter)
        obs = env.reset()
        done = False

        ticksSurvived = 0
        while done == False:
            action = manual_policy(*obs)
            obs, reward, done, info = env.step(action)
            if done == True:
                break
            ticksSurvived +=1
        survivalData.append(ticksSurvived)
    env.close()
    window = 50
    averagedData = [mean(survivalData[ind:ind+window]) for ind in range(len(survivalData)-window+1)]
    plt.xlabel("Liczba iteracji algorytmu QLearning", fontsize=20)
    plt.ylabel("Liczba klatek,\nprzez które wagonik\nutrzymał wahadło w pionie", fontsize=20)
    plt.plot(averagedData)
    plt.show()

def visualizeQLearning(n_iterations=1000):
    env = gym.make("CartPole-v1")
    ql = Qlearning()
    for iter in range(n_iterations):
        print(iter)
        prevObs = env.reset()
        done = False

        while done == False:
            action = ql.getAction(prevObs, iter)
            newObs, reward, done, info = env.step(action)
            ql.update(reward, prevObs, newObs, iter, action)
            prevObs = newObs

            if iter > n_iterations*0.9:
                env.render()
                time.sleep(0.01)
            if done == True:
                break
    env.close()

def visualizeManual(n_iterations=1000):
    env = gym.make("CartPole-v1")
    ql = Qlearning()
    for iter in range(n_iterations):
        obs = env.reset()
        done = False

        while done == False:
            action = manual_policy(*obs)
            obs, reward, done, info = env.step(action)

            env.render()
            time.sleep(0.01)
            if done == True:
                break
    env.close()
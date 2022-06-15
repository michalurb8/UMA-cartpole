import numpy as np

GO_LEFT = 0
GO_RIGHT = 1

INF = float('inf')

OBS_COUNT = 4

OBS_BUCKETS = [
    list(np.arange(-4.8, 4.9, 0.3)),
    [-INF, *list(np.arange(-5, 5, 0.3)), INF],
    list(np.arange(-0.42, 0.42, 0.03)),
    [-INF, *list(np.arange(-5, 5, 0.3)), INF]
]

def getState(observations):
    assert len(observations) == OBS_COUNT, "incorrect number of observations"
    assert len(OBS_BUCKETS) == OBS_COUNT, "incorrect number of buckets"
    states = []
    for index in range(OBS_COUNT):
        observation = observations[index]
        thresholds = OBS_BUCKETS[index]
        assert observation > thresholds[0], "No observation can be outside of range"
        for bucket_index, threshold in enumerate(thresholds[1:]):
            if observation < threshold:
                states.append(bucket_index)
                break
        else: states.append(bucket_index)
    return states
        
def learningRate(iteration: int, minRate: float = 0.05) -> float:
    return max(minRate, min(1.0, 1.0 - np.log10((iteration+1)/25)))

def explorationRate(iteration: int, minRate: float = 0.05) -> float:
    return max(minRate, min(1.0, 1.0 - np.log10((iteration+1)/25)))

class Qlearning:
    def __init__(self):
        state_shape = [len(thresholds)-1 for thresholds in OBS_BUCKETS]
        state_shape.append(2)
        self.Qtable = np.zeros(tuple(state_shape))

    def update(self, reward, prevObs, newObs, iteration, action, discount = 1):
        lr = learningRate(iteration)
        oldState = getState(prevObs)
        newState = getState(newObs)
        oldValue = self.Qtable[tuple(oldState)][action]

        newValue = reward + discount * max(self.Qtable[tuple(newState)])
        finalValue = (1-lr)*oldValue + lr*newValue
        self.Qtable[tuple(oldState)][action] = finalValue

    def getAction(self, obs, iteration) -> int:
        er = explorationRate(iteration)
        state = getState(obs)
        left = self.Qtable[tuple(state)][0]
        right = self.Qtable[tuple(state)][1]
        action = GO_LEFT if left > right else GO_RIGHT
        if left == right or np.random.uniform() < er:
            action = np.random.randint(2)
        return int(action)
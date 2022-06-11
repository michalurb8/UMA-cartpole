import numpy as np

INF = float('inf')

OBS_COUNT = 4
OBS_MIN = [-4.8, -INF, -0.42, -INF]
OBS_MAX = [+4.8, +INF, +0.42, +INF]

OBS_BUCKETS = [
    [-4.8, -2.4, 0, 2.4, 4.8],
    [-INF, -1, 0, 1, INF],
    [-0.42, -0.08, 0, 0.08, 0.42],
    [-INF, -1, 0, 1, INF]
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
    return states
        

class Qlearning:
    def __init__(self):
        state_shape = [len(thresholds)-1 for thresholds in OBS_BUCKETS]
        state_shape.append(2)
        print(state_shape)
        self.Qtable = np.zeros(tuple(state_shape))

    def step(self, obs, reward, done, info) -> int:
        print("OBS:", obs)
        state = getState(obs)
        print("STATE:", state)
        left = self.Qtable[tuple(state)][0]
        right = self.Qtable[tuple(state)][1]
        print("LEFT:", left)
        print("RIGHT:", right)
        return 0 if left > right else 1
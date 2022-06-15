import gym
from gym.utils import play

def callback(obs_t, obs_tp1, action, rew, done, info):
    return [done,]

if __name__ == "__main__":
    print("Try to balance the cart. Hold 'd' to drive to the right." \
          "Otherwise, the cart will drive to the left. Good luck!")

    env = gym.make("CartPole-v1")
    controls = {
        "d": 1
    }

    plotter = play.PlayPlot(callback, 300, ["Has game finished?"])
    play.play(env, keys_to_action=controls, callback=plotter.callback)

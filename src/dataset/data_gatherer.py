import argparse
import sys
import gym
from gym import wrappers, logger
from rtpt import RTPT

# Create RTPT object
rtpt = RTPT(name_initials='QD', experiment_name='TestingRTPT', max_iterations=10)

# Start the RTPT tracking
rtpt.start()


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='Pong-v0', help='Select the environment to run')
    args = parser.parse_args()

    env = gym.make(args.env_id)

    out_dir = 'data_gathered/random-agent-results/'
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        step = 0
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            plt.imshow(env.render('rgb_array'))
            plt.savefig(f"{out_dir}{args.env_id}_ep{i}_st{step}.png")
            step += 1
            if done:
                break
        rtpt.step(subtitle=f"step={i}/{episode_count}")

    env.close()

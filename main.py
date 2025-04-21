import gymnasium as gym
import Agents.dqnagent as dqn
import Environments.discreteCarRacing as denv

render = True
base_env = gym.make("CarRacing-v3", render_mode="human" if render else "rgb_array")
env = denv.DiscreteCarRacingObservations(denv.DiscreteCarRacingActions(base_env))

agent = dqn.DQN(
    state_shape=(48, 48, 3),
    n_actions=env.action_space.n,
    epsilon=0.2,
    lr=1e-4
)

agent.train(env, episodes=500, render=render)
#agent.load_model()
#agent.play(env, episodes=1000)

env.close()
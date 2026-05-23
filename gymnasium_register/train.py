import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import torch as th
import torch.nn as nn
from pathlib import Path
import tic_tac_go_env

#This class is AI code idk whats happening inside
#it just creates a custom 3X3 window
class CustomTinyCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            # First layer: 3x3 filter fits perfectly on a 6x6 board
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Second layer: 3x3 filter to capture local combinations
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape dynamically to hook up the linear layer
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
            
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

#This class is also AI
#Print out grad number with other info
class GraduationEvalCallback(EvalCallback):
    def __init__(self, graduation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graduation = graduation

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            print(f"\nGraduation {self.graduation}")

        return super()._on_step()
    

board = (("", "", "", "B", "B", "B", "B", "B"),
         ("U", "", "O", "B", "B", "B", "B", "B"),
         ("", "", "O", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"))

render_mode=None
env = gym.make("tic_tac_go_env/TicTacWorld-v0", length=3, width=3, board=board, render_mode=render_mode, reset_option=1)

obs, info = env.reset()

callbackEnv = gym.make("tic_tac_go_env/TicTacWorld-v0", length=3, width=3, board=board, render_mode=render_mode, reset_option=1)
obs, info = callbackEnv.reset()

policy_kwargs = dict(
    features_extractor_class=CustomTinyCNN,
    features_extractor_kwargs=dict(features_dim=128),
    normalize_images=False
)

model_path = Path("dqn_tic_tac_go.zip")

def learnProcess(num, threshold = 24):
    eval_episodes = 5 if num < 4 else 10

    env = gym.make("tic_tac_go_env/TicTacWorld-v0", length=6, width=6, board=board, render_mode=render_mode, reset_option=num)
    obs, info = env.reset()

    callbackEnv = gym.make("tic_tac_go_env/TicTacWorld-v0", length=6, width=6, board=board, render_mode=render_mode, reset_option=num)
    obs, info = callbackEnv.reset()

    if model_path.exists():
        model = DQN.load(model_path, env=env)
    else:
        model = DQN("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

    model.exploration_initial_eps = 0.50
    model.exploration_final_eps = 0.05

    callback_on_thresh = StopTrainingOnRewardThreshold(threshold, verbose=1)
    env_callback = GraduationEvalCallback(num,
                                          callbackEnv, 
                                          callback_after_eval=callback_on_thresh, 
                                          eval_freq=1000,
                                          n_eval_episodes=eval_episodes,
                                          verbose=1)

    model.learn(total_timesteps=999999999999999, log_interval=4, reset_num_timesteps=True, callback=env_callback)
    model.save(model_path)

#Graduation 1(Agent randomness static O)
if model_path.exists():
    model = DQN.load(model_path, env=env)
else:
    model = DQN("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

model.exploration_initial_eps = 0.50
model.exploration_final_eps = 0.05

callback_on_thresh = StopTrainingOnRewardThreshold(24, verbose=1)
eval_episodes = 5
env_callback = GraduationEvalCallback(1,
                                      callbackEnv, 
                                      callback_after_eval=callback_on_thresh,
                                      eval_freq=1000,
                                      n_eval_episodes=eval_episodes, 
                                      verbose=1)

model.learn(total_timesteps=999999999999999, log_interval=4, reset_num_timesteps=True, callback=env_callback)
model.save(model_path)


#Graduation 2 6X6(Same thing but bigger)

board = (("", "", "", "", "", "", "B", "B"),
         ("", "", "", "", "", "", "B", "B"),
         ("", "", "U", "", "", "O", "B", "B"),
         ("", "", "", "", "O", "", "B", "B"),
         ("", "", "", "", "", "", "B", "B"),
         ("", "", "", "", "", "", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"))

learnProcess(2)

#Graduation 3 Sparser Os(List of good positions for all 3 of them)
board = (("", "", "", "", "", "", "B", "B"),
         ("", "", "", "", "O", "", "B", "B"),
         ("", "", "", "", "", "", "B", "B"),
         ("", "", "", "U", "", "", "B", "B"),
         ("", "O", "", "", "", "", "B", "B"),
         ("", "", "", "", "", "", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"))

learnProcess(3)

#Graduation 4 Small Xs(Same thing but with few Xs)
board = (("", "", "", "", "X", "", "B", "B"),
         ("", "X", "", "", "O", "", "B", "B"),
         ("", "", "", "", "", "X", "B", "B"),
         ("", "", "", "U", "", "", "B", "B"),
         ("", "O", "X", "", "X", "", "B", "B"),
         ("X", "", "", "", "", "", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"))

learnProcess(4)

#Graduation 5 8X8(Same thing but bigger)
board = (("", "", "", "", "X", "", "", ""),
         ("", "X", "", "O", "", "", "", ""),
         ("", "", "", "", "", "X", "", ""),
         ("", "", "", "", "", "", "", "X"),
         ("", "", "X", "", "X", "", "", ""),
         ("X", "O", "", "", "", "U", "", ""),
         ("", "", "", "X", "", "", "", ""),
         ("X", "", "", "", "X", "", "", "X"))

learnProcess(5)

#Graduation 6 Bigger Xs (Come up with random strategy)
board = (("", "", "", "", "X", "", "", ""),
         ("", "X", "", "O", "", "", "", ""),
         ("", "", "", "", "", "X", "", ""),
         ("", "", "", "", "", "", "", "X"),
         ("", "", "X", "", "X", "", "", ""),
         ("X", "O", "", "", "", "U", "", ""),
         ("", "", "", "X", "", "", "", ""),
         ("X", "", "", "", "X", "", "", "X"))

learnProcess(6)

#Graduation 7 Blocks (Come up with random strategy)
board = (("", "", "", "", "X", "", "", ""),
         ("", "X", "", "O", "", "", "", ""),
         ("B", "B", "B", "", "", "X", "", ""),
         ("", "", "", "", "", "", "", "X"),
         ("", "", "X", "", "X", "", "", ""),
         ("X", "O", "", "B", "B", "U", "", ""),
         ("", "", "", "B", "B", "", "", ""),
         ("X", "", "", "B", "B", "", "", "X"))

learnProcess(7)

#Graduation 8 Varying Board Sizes, Real Board (Change Threshold, Find boards)
board = (("", "", "", "", "", "", "", ""),
         ("", "", "U", "X", "", "", "O", ""),
         ("", "B", "B", "", "X", "B", "B", "X"),
         ("", "", "X", "X", "", "", "X", ""),
         ("X", "X", "", "", "", "", "", ""),
         ("", "B", "B", "B", "B", "B", "B", ""),
         ("", "", "X", "", "O", "", "", "X"),
         ("", "", "", "", "", "X", "", ""))

learnProcess(8, 38)


env = gym.make("tic_tac_go_env/TicTacWorld-v0", length=len(board), width=len(board[0]), board=board, render_mode="human")

if model_path.exists():
    model = DQN.load(model_path, env=env)
else:
    model = DQN("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

obs, info = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

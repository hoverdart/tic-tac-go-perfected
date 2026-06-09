import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.utils import LinearSchedule
import torch as th
import torch.nn as nn
import numpy as np
import importlib.util
from pathlib import Path
import tic_tac_go_env
import BFStoTrainer

#This class is AI code idk whats happening inside
#it just creates a custom 3X3 window for the CNN
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


class GraduationTrainingLogCallback(BaseCallback):
    def __init__(self, graduation):
        super().__init__()
        self.graduation = graduation

    def _on_step(self) -> bool:
        self.logger.record("time/grad", self.graduation)
        return True


class StopTrainingOnMeanReward(BaseCallback):
    def __init__(self, reward_threshold, max_reward_std, verbose=0):
        super().__init__(verbose=verbose)
        self.reward_threshold = reward_threshold
        self.max_reward_std = max_reward_std
        self.threshold_reached = False

    def _on_step(self) -> bool:
        mean_reward = self.parent.last_mean_reward
        reward_std = None
        if hasattr(self.parent, "evaluations_results") and self.parent.evaluations_results:
            reward_std = float(np.std(self.parent.evaluations_results[-1]))
            self.logger.record("eval/reward_std", reward_std)
            self.logger.record("eval/reward_var", float(np.var(self.parent.evaluations_results[-1])))

        if self.verbose >= 1 and reward_std is not None:
            print(f"Eval mean={mean_reward:.2f}, std={reward_std:.2f}")

        if (
            mean_reward >= self.reward_threshold
            and reward_std is not None
            and reward_std <= self.max_reward_std
        ):
            self.threshold_reached = True
            if self.verbose >= 1:
                print(
                    f"Stopping training because the mean reward {mean_reward:.2f} "
                    f"is above the threshold {self.reward_threshold} "
                    f"and reward std {reward_std:.2f} is at or below {self.max_reward_std}"
                )
            return False

        if self.verbose >= 1 and mean_reward >= self.reward_threshold:
            if reward_std is None:
                print("Not graduating because reward std is unavailable")
            else:
                print(
                    f"Not graduating because reward std {reward_std:.2f} "
                    f"is above {self.max_reward_std}"
                )

        return True
    

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
eval_boards_path = Path(__file__).resolve().parent / "generated_eval_boards.py"

def load_eval_boards_for_grad(grad_num):
    if not eval_boards_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("generated_eval_boards", eval_boards_path)
    generated_eval_boards = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated_eval_boards)

    eval_boards = getattr(generated_eval_boards, "EVAL_BOARDS", {})
    boards = eval_boards.get(grad_num)
    if not boards:
        return None

    return {grad_num: boards}

def use_eval_boards_if_available(env, grad_num):
    eval_boards = load_eval_boards_for_grad(grad_num)
    if eval_boards is None:
        print(f"Eval grad {grad_num}: using training boards")
        return

    env.unwrapped.board_pool_override = eval_boards
    print(f"Eval grad {grad_num}: using {len(eval_boards[grad_num])} held-out eval boards")

def get_exploration_fraction(num):
    if num == 12:
        return 0.15
    if num <= 2:
        return 0.60
    if num <= 4:
        return 0.40
    if num <= 6:
        return 0.30
    return 0.20

def set_exploration_schedule(model, num):
    if num == 12:
        model.exploration_initial_eps = 0.20
    elif num == 11:
        model.exploration_initial_eps = 0.30
    else:
        model.exploration_initial_eps = 0.50
    model.exploration_final_eps = 0.05
    model.exploration_fraction = get_exploration_fraction(num)
    model.exploration_schedule = LinearSchedule(
        model.exploration_initial_eps,
        model.exploration_final_eps,
        model.exploration_fraction,
    )

def inject(model, boards):
    BFSone = BFStoTrainer.BFStoTrainer()

    model.replay_buffer.reset()

    injectCounter = 0
    injectSolve = ["RRULUUULUULDULULLDDLDDDUUUURRRRDRDRDDDLLULULLULUURDDLDRRDR", 
                   "DDDLLLLDDRURULUL",
                   "LDDRURURRDLLLDLDDRUULURRRUR",
                   "RRULDLDDRUURU",
                   "LLDDRDDUULLDRR",
                   "LLDDRUDRDDLUULU"]

    for board_to_solve in boards:
        dataset = BFSone.solve(board_to_solve, injectSolve[injectCounter])
        injectCounter += 1

        for step in dataset:
            model.replay_buffer.add(
                obs=step["observation"],
                next_obs=step["next_observation"],
                action=np.array([step["action"]]),
                reward=np.array([step["reward"]]),
                done=np.array([step["done"]]),
                infos=[{}]
            )

def learnProcess(num, threshold = 24):
    eval_episodes = 5 if num < 4 else 10
    max_reward_std = 15 if num <= 3 else 10
    threshold_reached = False

    env = gym.make("tic_tac_go_env/TicTacWorld-v0", length=6, width=6, board=board, render_mode=render_mode, reset_option=num)
    obs, info = env.reset()

    callbackEnv = gym.make("tic_tac_go_env/TicTacWorld-v0", length=6, width=6, board=board, render_mode=render_mode, reset_option=num)
    use_eval_boards_if_available(callbackEnv, num)
    obs, info = callbackEnv.reset()
    
    if model_path.exists():
        model = DQN.load(model_path, env=env)
        model.replay_buffer.reset()
    else:
        model = DQN("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

    if num == 12:
        inject(model, injection_boards)

    while not threshold_reached:
        set_exploration_schedule(model, num)

        callback_on_thresh = StopTrainingOnMeanReward(
            threshold,
            max_reward_std=max_reward_std,
            verbose=1,
        )
        env_callback = GraduationEvalCallback(num,
                                              callbackEnv, 
                                              callback_after_eval=callback_on_thresh, 
                                              eval_freq=1000,
                                              n_eval_episodes=eval_episodes,
                                              log_path=f"./eval_logs/grad_{num}",
                                              verbose=1)

        callbacks = CallbackList([
            GraduationTrainingLogCallback(num),
            env_callback,
        ])

        model.learn(total_timesteps=50000, log_interval=4, reset_num_timesteps=True, callback=callbacks)
        model.save(model_path)

        threshold_reached = callback_on_thresh.threshold_reached

injection_boards = [
    (("", "X", "", "", "X", "", "B", "B"),
     ("", "O", "", "X", "", "X", "X", "B"),
     ("", "", "B", "B", "X", "", "", ""),
     ("", "", "B", "B", "", "", "X", "X"),
     ("X", "X", "", "", "B", "X", "", ""),
     ("", "B", "X", "", "", "B", "", ""),
     ("", "B", "", "", "", "", "O", ""),
     ("", "", "X", "", "", "U", "", "")),

    (("", "", "X", "X", "", "U", "B", "B"),
     ("", "O", "", "", "X", "", "B", "B"),
     ("", "X", "X", "B", "X", "X", "B", "B"),
     ("", "", "X", "", "", "", "B", "B"),
     ("", "", "O", "X", "", "", "B", "B"),
     ("", "", "", "", "", "", "B", "B"),
     ("B", "B", "B", "B", "B", "B", "B", "B"),
     ("B", "B", "B", "B", "B", "B", "B", "B")),

    (("", "U", "", "X", "", "", "B", "B"),
     ("", "X", "X", "", "O", "", "B", "B"),
     ("X", "", "X", "B", "", "X", "B", "B"),
     ("", "O", "", "B", "X", "", "B", "B"),
     ("", "X", "", "X", "", "X", "B", "B"),
     ("", "", "", "X", "X", "", "B", "B"),
     ("B", "B", "B", "B", "B", "B", "B", "B"),
     ("B", "B", "B", "B", "B", "B", "B", "B")),

    (("", "X", "", "O", "B", "B", "B", "B"),
     ("U", "", "", "X", "B", "B", "B", "B"),
     ("", "O", "X", "", "B", "B", "B", "B"),
     ("", "X", "", "X", "B", "B", "B", "B"),
     ("B", "B", "B", "B", "B", "B", "B", "B"),
     ("B", "B", "B", "B", "B", "B", "B", "B"),
     ("B", "B", "B", "B", "B", "B", "B", "B"),
     ("B", "B", "B", "B", "B", "B", "B", "B")),

    (("", "", "X", "U", "", "", "B", "B"),
     ("X", "", "B", "", "X", "", "B", "B"),
     ("", "O", "", "X", "B", "X", "B", "B"),
     ("X", "", "", "", "O", "", "B", "B"),
     ("", "X", "X", "B", "X", "X", "B", "B"),
     ("X", "", "", "X", "", "", "B", "B"),
     ("B", "B", "B", "B", "B", "B", "B", "B"),
     ("B", "B", "B", "B", "B", "B", "B", "B")),

    (("", "", "B", "B", "B", "B", "B", "B"),
     ("X", "", "B", "B", "B", "B", "B", "B"),
     ("", "", "X", "U", "B", "B", "B", "B"),
     ("O", "", "X", "X", "B", "B", "B", "B"),
     ("", "X", "", "", "B", "B", "B", "B"),
     ("X", "", "O", "", "B", "B", "B", "B"),
     ("B", "B", "", "X", "B", "B", "B", "B"),
     ("B", "B", "", "", "B", "B", "B", "B")),
]

#1  3x3 active area, static Os, only agent randomized
learnProcess(1)


#2 6x6 active area, static Os, only agent randomized

board = (("", "", "", "", "", "", "B", "B"),
         ("", "", "", "", "", "", "B", "B"),
         ("", "", "U", "", "", "O", "B", "B"),
         ("", "", "", "", "O", "", "B", "B"),
         ("", "", "", "", "", "", "B", "B"),
         ("", "", "", "", "", "", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"))

learnProcess(2)

#3 8x8, Os only slightly misaligned, no Xs
board = (("", "", "", "", "", "", "B", "B"),
         ("", "", "", "", "O", "", "B", "B"),
         ("", "", "", "", "", "", "B", "B"),
         ("", "", "", "U", "", "", "B", "B"),
         ("", "O", "", "", "", "", "B", "B"),
         ("", "", "", "", "", "", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"))

learnProcess(3)

#4  8x8, randomized Os (spread out), no Xs
board = (("", "", "", "", "X", "", "B", "B"),
         ("", "X", "", "", "O", "", "B", "B"),
         ("", "", "", "", "", "X", "B", "B"),
         ("", "", "", "U", "", "", "B", "B"),
         ("", "O", "X", "", "X", "", "B", "B"),
         ("X", "", "", "", "", "", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"))

learnProcess(4)

#5  8x8, randomized Os + sparse non-dangerous Xs
board = (("", "", "", "", "X", "", "", ""),
         ("", "X", "", "O", "", "", "", ""),
         ("", "", "", "", "", "X", "", ""),
         ("", "", "", "", "", "", "", "X"),
         ("", "", "X", "", "X", "", "", ""),
         ("X", "O", "", "", "", "U", "", ""),
         ("", "", "", "X", "", "", "", ""),
         ("X", "", "", "", "X", "", "", "X"))

learnProcess(5, 28)

#6  8x8, medium random Xs
board = (("", "", "", "", "X", "", "", ""),
         ("", "X", "", "O", "", "", "", ""),
         ("", "", "", "", "", "X", "", ""),
         ("", "", "", "", "", "", "", "X"),
         ("", "", "X", "", "X", "", "", ""),
         ("X", "O", "", "", "", "U", "", ""),
         ("", "", "", "X", "", "", "", ""),
         ("X", "", "", "", "X", "", "", "X"))

learnProcess(6, 28)

#7  8x8, more random Xs, farther Os, agent starts somewhat far
learnProcess(7, 28)

#8  8x8, full random Xs, agent starts far
learnProcess(8, 30)

#9  8x8, dangerous near-line-threat Xs
board = (("", "", "", "", "X", "", "", ""),
         ("", "X", "", "O", "", "", "", ""),
         ("B", "B", "B", "", "", "X", "", ""),
         ("", "", "", "", "", "", "", "X"),
         ("", "", "X", "", "X", "", "", ""),
         ("X", "O", "", "B", "B", "U", "", ""),
         ("", "", "", "B", "B", "", "", ""),
         ("X", "", "", "B", "B", "", "", "X"))

learnProcess(9, 30)

#10  8x8, Xs + B blocks
learnProcess(10, 30)

#Graduation 11 Varying Board Sizes, Real Board (Change Threshold, Find boards)
board = (("", "", "", "", "", "", "", ""),
         ("", "", "U", "X", "", "", "O", ""),
         ("", "B", "B", "", "X", "B", "B", "X"),
         ("", "", "X", "X", "", "", "X", ""),
         ("X", "X", "", "", "", "", "", ""),
         ("", "B", "B", "B", "B", "B", "B", ""),
         ("", "", "X", "", "O", "", "", "X"),
         ("", "", "", "", "", "X", "", ""))

learnProcess(11, 29)

#Graduation 12 Final Training
learnProcess(12, 36)


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

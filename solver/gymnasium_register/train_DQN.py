import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.utils import LinearSchedule
import torch as th
import torch.nn as nn
import numpy as np
import importlib.util
import shutil
import random
from pathlib import Path
from datetime import datetime
import tic_tac_go_env
import BFStoTrainer
from injection_boards import (
    GRAD6_INJECTION_BOARDS,
    GRAD6_INJECTION_SOLUTIONS,
    GRAD10_INJECTION_BOARDS,
    GRAD10_INJECTION_SOLUTIONS,
    FINAL_INJECTION_BOARDS,
    FINAL_INJECTION_SOLUTIONS,
)

#This class is AI code idk whats happening inside
#Three 3x3 conv layers give each final conv cell a 7x7 receptive field.
class CustomTinyCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
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
    def __init__(self, graduation, eval_metrics, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graduation = graduation
        self.eval_metrics = eval_metrics

    def _on_step(self) -> bool:
        should_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if should_eval:
            print(f"\nGraduation {self.graduation}")

        continue_training = super()._on_step()

        if should_eval and self.evaluations_results:
            rewards = self.evaluations_results[-1]
            lengths = self.evaluations_length[-1]
            self.eval_metrics["reward_std"] = float(np.std(rewards))
            self.eval_metrics["mean_length"] = float(np.mean(lengths))
            self.eval_metrics["mean_reward"] = float(self.last_mean_reward)

        return continue_training


class GraduationTrainingLogCallback(BaseCallback):
    def __init__(self, graduation, eval_metrics):
        super().__init__()
        self.graduation = graduation
        self.eval_metrics = eval_metrics

    def _on_step(self) -> bool:
        self.logger.record("time/grad", self.graduation)
        if "reward_std" in self.eval_metrics:
            self.logger.record("rollout/eval_reward_std", self.eval_metrics["reward_std"])
            self.logger.record("rollout/eval_mean_length", self.eval_metrics["mean_length"])
            self.logger.record("rollout/eval_mean_reward", self.eval_metrics["mean_reward"])
        return True


class StopTrainingOnMeanReward(BaseCallback):
    def __init__(self, reward_threshold, max_reward_std, verbose=0):
        super().__init__(verbose=verbose)
        self.reward_threshold = reward_threshold
        self.max_reward_std = max_reward_std
        self.threshold_reached = False
        self.graduation_mean_reward = None
        self.graduation_reward_std = None

    def _on_step(self) -> bool:
        mean_reward = self.parent.last_mean_reward
        reward_std = None
        if hasattr(self.parent, "evaluations_results") and self.parent.evaluations_results:
            reward_std = float(np.std(self.parent.evaluations_results[-1]))
            self.logger.record("eval/reward_std", reward_std)

        if self.verbose >= 1 and reward_std is not None:
            print(f"Eval mean={mean_reward:.2f}, std={reward_std:.2f}")

        if (
            mean_reward >= self.reward_threshold
            and reward_std is not None
            and reward_std <= self.max_reward_std
        ):
            self.threshold_reached = True
            self.graduation_mean_reward = float(mean_reward)
            self.graduation_reward_std = float(reward_std)
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
    features_extractor_kwargs=dict(features_dim=256),
    normalize_images=False
)

model_path = Path("dqn_tic_tac_go.zip")
eval_boards_path = Path(__file__).resolve().parent / "generated_eval_boards.py"
graduation_output_dir = Path(__file__).resolve().parent / "graduation_checkpoints"
graduation_log_path = graduation_output_dir / "graduation_log.txt"
START_FROM_GRAD = 13

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def write_graduation_log(lines):
    graduation_output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(lines, str):
        lines = [lines]
    with graduation_log_path.open("a") as log_file:
        log_file.write(f"[{timestamp()}]\n")
        for line in lines:
            log_file.write(f"{line}\n")
        log_file.write("\n")

def model_training_stats(model):
    return {
        "num_timesteps": getattr(model, "num_timesteps", None),
        "n_updates": getattr(model, "_n_updates", None),
    }

def format_value(value):
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)

def save_graduation_checkpoint(model, model_path, grad_num, mean_reward, reward_std, mean_length):
    graduation_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = graduation_output_dir / f"dqn_tic_tac_go_grad_{grad_num}.zip"
    shutil.copy2(model_path, checkpoint_path)

    training_stats = model_training_stats(model)
    write_graduation_log([
        f"EVENT: END grad {grad_num} / PASSED",
        f"NEXT_GRAD: {grad_num + 1}",
        f"CHECKPOINT_FILE: {checkpoint_path.name}",
        "EVAL_STATS:",
        f"  eval_mean_reward: {format_value(mean_reward)}",
        f"  eval_reward_std: {format_value(reward_std)}",
        f"  eval_mean_length: {format_value(mean_length)}",
        "TRAINING_STATS_AT_SAVE:",
        f"  num_timesteps: {format_value(training_stats['num_timesteps'])}",
        f"  n_updates: {format_value(training_stats['n_updates'])}",
    ])
    print(f"Saved graduation checkpoint: {checkpoint_path}")

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

def use_fixed_grad7_eval_sequence(env, grad_num):
    if grad_num != 10:
        return

    eval_boards = load_eval_boards_for_grad(grad_num)
    if eval_boards is None:
        return

    boards = eval_boards[grad_num]
    rng = random.Random(7007)
    fixed_boards = rng.sample(boards, min(30, len(boards)))
    env.unwrapped.board_sequence_override = {grad_num: fixed_boards}
    env.unwrapped.board_sequence_index = 0
    print(f"Eval grad {grad_num}: fixed {len(fixed_boards)} board eval sequence")

def get_exploration_fraction(num):
    if num == 17:
        return 0.15
    if num <= 4:
        return 0.60
    if num <= 8:
        return 0.40
    if num <= 11:
        return 0.30
    return 0.20

def set_exploration_schedule(model, num):
    if num == 17:
        model.exploration_initial_eps = 0.20
    elif num == 16:
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

def inject(model, boards, solutions, current_grad):
    BFSone = BFStoTrainer.BFStoTrainer()

    if len(boards) != len(solutions):
        raise ValueError("Injection board count must match solution count")

    model.replay_buffer.reset()

    for board_to_solve, solution in zip(boards, solutions):
        dataset = BFSone.solve(
            board_to_solve,
            solution,
            current_grad=current_grad,
            terminate_on_repeated_states=True,
            repeat_termination_limit=3,
            penalize_repeated_states=True,
        )

        for step in dataset:
            model.replay_buffer.add(
                obs=step["observation"],
                next_obs=step["next_observation"],
                action=np.array([step["action"]]),
                reward=np.array([step["reward"]]),
                done=np.array([step["done"]]),
                infos=[{}]
            )

def reward_threshold_for_grad(num):
    if num <= 6:
        return 24
    if num <= 10:
        return 28
    if num <= 12:
        return 29
    return 31


def std_threshold_for_grad(num):
    if num <= 3:
        return 7
    if num == 10:
        return 15
    if num <= 10:
        return 10
    if num == 15:
        return 15
    return 13


def learnProcess(num, threshold=None):
    if threshold is None:
        threshold = reward_threshold_for_grad(num)
    eval_episodes = 5 if num < 6 else 30
    max_reward_std = std_threshold_for_grad(num)
    threshold_reached = False

    env = gym.make("tic_tac_go_env/TicTacWorld-v0", length=6, width=6, board=board, render_mode=render_mode, reset_option=num)
    env.unwrapped.terminate_on_repeated_states = True
    env.unwrapped.repeat_termination_limit = 3
    env.unwrapped.penalize_repeated_states = True
    obs, info = env.reset()

    callbackEnv = gym.make("tic_tac_go_env/TicTacWorld-v0", length=6, width=6, board=board, render_mode=render_mode, reset_option=num)
    callbackEnv.unwrapped.terminate_on_repeated_states = True
    callbackEnv.unwrapped.repeat_termination_limit = 5
    callbackEnv.unwrapped.penalize_repeated_states = False
    use_eval_boards_if_available(callbackEnv, num)
    use_fixed_grad7_eval_sequence(callbackEnv, num)
    obs, info = callbackEnv.reset()
    
    if model_path.exists():
        model = DQN.load(model_path, env=env)
        model.replay_buffer.reset()
    else:
        model = DQN("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

    training_stats = model_training_stats(model)
    write_graduation_log([
        f"EVENT: START grad {num}",
        "THRESHOLDS:",
        f"  reward_threshold: {threshold}",
        f"  max_reward_std: {max_reward_std}",
        "EVAL_CONFIG:",
        f"  eval_episodes: {eval_episodes}",
        "TRAINING_STATS_AT_START:",
        f"  num_timesteps: {format_value(training_stats['num_timesteps'])}",
        f"  n_updates: {format_value(training_stats['n_updates'])}",
    ])

    if num == 6:
        inject(model, GRAD6_INJECTION_BOARDS, GRAD6_INJECTION_SOLUTIONS, current_grad=num)

    if num == 10:
        inject(
            model,
            GRAD10_INJECTION_BOARDS,
            GRAD10_INJECTION_SOLUTIONS,
            current_grad=num,
        )

    if num == 17:
        inject(model, FINAL_INJECTION_BOARDS, FINAL_INJECTION_SOLUTIONS, current_grad=num)

    while not threshold_reached:
        set_exploration_schedule(model, num)
        eval_metrics = {}

        callback_on_thresh = StopTrainingOnMeanReward(
            threshold,
            max_reward_std=max_reward_std,
            verbose=1,
        )
        env_callback = GraduationEvalCallback(num,
                                              eval_metrics,
                                              callbackEnv, 
                                              callback_after_eval=callback_on_thresh, 
                                              eval_freq=1000,
                                              n_eval_episodes=eval_episodes,
                                              log_path=f"./eval_logs/grad_{num}",
                                              verbose=1)

        callbacks = CallbackList([
            GraduationTrainingLogCallback(num, eval_metrics),
            env_callback,
        ])

        model.learn(total_timesteps=50000, log_interval=4, reset_num_timesteps=True, callback=callbacks)
        model.save(model_path)

        threshold_reached = callback_on_thresh.threshold_reached
        if threshold_reached:
            mean_reward = callback_on_thresh.graduation_mean_reward
            reward_std = callback_on_thresh.graduation_reward_std
            mean_length = eval_metrics.get("mean_length")
            save_graduation_checkpoint(
                model,
                model_path,
                num,
                mean_reward,
                reward_std,
                mean_length,
            )

def run_grad(num, threshold=None):
    if num < START_FROM_GRAD:
        print(f"Skipping grad {num}; START_FROM_GRAD={START_FROM_GRAD}")
        return

    learnProcess(num, threshold)

#1  3x3 active area, adjacent Os, agent randomized
run_grad(1)


#2 6x6 active area, adjacent Os, agent randomized
run_grad(2)


#3 6x6 active area, close line Os, agent randomized

board = (("", "", "", "", "", "", "B", "B"),
         ("", "", "", "", "", "", "B", "B"),
         ("", "", "U", "", "", "O", "B", "B"),
         ("", "", "", "", "O", "", "B", "B"),
         ("", "", "", "", "", "", "B", "B"),
         ("", "", "", "", "", "", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"))

run_grad(3)

#4 8x8 active area, adjacent Os, no Xs
run_grad(4)

#5 8x8, Os only slightly misaligned, no Xs
board = (("", "", "", "", "", "", "B", "B"),
         ("", "", "", "", "O", "", "B", "B"),
         ("", "", "", "", "", "", "B", "B"),
         ("", "", "", "U", "", "", "B", "B"),
         ("", "O", "", "", "", "", "B", "B"),
         ("", "", "", "", "", "", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"))

run_grad(5)

#6  8x8, randomized Os (spread out), no Xs
board = (("", "", "", "", "X", "", "B", "B"),
         ("", "X", "", "", "O", "", "B", "B"),
         ("", "", "", "", "", "X", "B", "B"),
         ("", "", "", "U", "", "", "B", "B"),
         ("", "O", "X", "", "X", "", "B", "B"),
         ("X", "", "", "", "", "", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"),
         ("B", "B", "B", "B", "B", "B", "B", "B"))

run_grad(6)

#7  8x8, no Xs, uncapped O distance, agent distance <= 10
run_grad(7)

#8  8x8, 1 random X, uncapped O distance, agent distance <= 10
run_grad(8)

#9  8x8, 2 random Xs, uncapped O distance, agent distance <= 10
run_grad(9)

#10  8x8, randomized Os + sparse non-dangerous Xs
board = (("", "", "", "", "X", "", "", ""),
         ("", "X", "", "O", "", "", "", ""),
         ("", "", "", "", "", "X", "", ""),
         ("", "", "", "", "", "", "", "X"),
         ("", "", "X", "", "X", "", "", ""),
         ("X", "O", "", "", "", "U", "", ""),
         ("", "", "", "X", "", "", "", ""),
         ("X", "", "", "", "X", "", "", "X"))

run_grad(10)

#11  8x8, medium random Xs
board = (("", "", "", "", "X", "", "", ""),
         ("", "X", "", "O", "", "", "", ""),
         ("", "", "", "", "", "X", "", ""),
         ("", "", "", "", "", "", "", "X"),
         ("", "", "X", "", "X", "", "", ""),
         ("X", "O", "", "", "", "U", "", ""),
         ("", "", "", "X", "", "", "", ""),
         ("X", "", "", "", "X", "", "", "X"))

run_grad(11)

#12  8x8, more random Xs, farther Os, agent starts somewhat far
run_grad(12)

#13  8x8, full random Xs, agent starts far
run_grad(13)

#14  8x8, dangerous near-line-threat Xs
board = (("", "", "", "", "X", "", "", ""),
         ("", "X", "", "O", "", "", "", ""),
         ("B", "B", "B", "", "", "X", "", ""),
         ("", "", "", "", "", "", "", "X"),
         ("", "", "X", "", "X", "", "", ""),
         ("X", "O", "", "B", "B", "U", "", ""),
         ("", "", "", "B", "B", "", "", ""),
         ("X", "", "", "B", "B", "", "", "X"))

run_grad(14)

#15  8x8, Xs + B blocks
run_grad(15)

#Graduation 16 Varying Board Sizes, Real Board (Change Threshold, Find boards)
board = (("", "", "", "", "", "", "", ""),
         ("", "", "U", "X", "", "", "O", ""),
         ("", "B", "B", "", "X", "B", "B", "X"),
         ("", "", "X", "X", "", "", "X", ""),
         ("X", "X", "", "", "", "", "", ""),
         ("", "B", "B", "B", "B", "B", "B", ""),
         ("", "", "X", "", "O", "", "", "X"),
         ("", "", "", "", "", "X", "", ""))

run_grad(16)

#Graduation 17 Final Training
run_grad(17)


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

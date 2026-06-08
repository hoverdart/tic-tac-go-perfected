from gymnasium.envs.registration import register

register(
    id="tic_tac_go_env/TicTacWorld-v0",
    entry_point="tic_tac_go_env.envs:TicTacWorldEnv",
    max_episode_steps=200
)

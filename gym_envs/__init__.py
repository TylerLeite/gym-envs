from gym.envs.registration import register

register(
    id='kranch-v0',
    entry_point='gym_envs.envs:KranchEnv',
)

register(
    id='skg-v0',
    entry_point='gym_envs.envs:SkgEnv',
)

register(
    id='t3-v0',
    entry_point='gym_envs.envs:TicTacToeEnv',
)

# register(
#     id='t3r-v0',
#     entry_point='gym_envs.envs:TicTacToeRoppEnv',
# )

register(
    id='f3-v0',
    entry_point='gym_envs.envs:FicFacFoeEnv',
)

register(
    id='simp-v0',
    entry_point='gym_envs.envs:SimpleEnv',
)

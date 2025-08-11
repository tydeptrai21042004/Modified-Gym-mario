"""Registration code of Gym environments in this package."""
import gym

# 5-tuple adapter (classic -> Gym>=0.26)
from .wrappergym import V0toV26

# Import concrete env classes for callable factories
from .smb_env import SuperMarioBrosEnv
from .smb_random_stages_env import SuperMarioBrosRandomStagesEnv


def _register_mario_env(id, is_random=False, **kwargs):
    """
    Register a Super Mario Bros. (1/2) environment with OpenAI Gym.

    Args:
        id (str): id for the env to register
        is_random (bool): whether to use the random levels environment
        kwargs (dict): keyword arguments for the SuperMarioBrosEnv initializer
    """
    env_cls = SuperMarioBrosEnv

    # Factory so we can wrap BEFORE Gym's wrappers (e.g., TimeLimit)
    def _make_env(**_ignored):
        env = env_cls(**kwargs)
        return V0toV26(env)

    gym.envs.registration.register(
        id=id,
        entry_point=_make_env,          # callable factory returns adapted env
        max_episode_steps=9_999_999,
        reward_threshold=9_999_999,
        nondeterministic=True,
    )


def _register_mario_stage_env(id, **kwargs):
    """
    Register a Super Mario Bros. (1/2) single-stage environment with Gym.

    Args:
        id (str): id for the env to register
        kwargs (dict): keyword arguments for the SuperMarioBrosEnv initializer
    """
    def _make_env(**_ignored):
        env = SuperMarioBrosEnv(**kwargs)
        return V0toV26(env)

    gym.envs.registration.register(
        id=id,
        entry_point=_make_env,          # callable factory returns adapted env
        max_episode_steps=9_999_999,
        reward_threshold=9_999_999,
        nondeterministic=True,
    )


# ---------- Family registrations ----------

# Super Mario Bros.
_register_mario_env('SuperMarioBros-v0', rom_mode='vanilla')
_register_mario_env('SuperMarioBros-v1', rom_mode='downsample')
_register_mario_env('SuperMarioBros-v2', rom_mode='pixel')
_register_mario_env('SuperMarioBros-v3', rom_mode='rectangle')

# Super Mario Bros. Random Levels
_register_mario_env('SuperMarioBrosRandomStages-v0', is_random=True, rom_mode='vanilla')
_register_mario_env('SuperMarioBrosRandomStages-v1', is_random=True, rom_mode='downsample')
_register_mario_env('SuperMarioBrosRandomStages-v2', is_random=True, rom_mode='pixel')
_register_mario_env('SuperMarioBrosRandomStages-v3', is_random=True, rom_mode='rectangle')

# Super Mario Bros. 2 (Lost Levels)
_register_mario_env('SuperMarioBros2-v0', lost_levels=True, rom_mode='vanilla')
_register_mario_env('SuperMarioBros2-v1', lost_levels=True, rom_mode='downsample')


# ---------- Per-stage registrations ----------

# a template for making individual stage environments
_ID_TEMPLATE = 'SuperMarioBros{}-{}-{}-v{}'
# A list of ROM modes for each level environment
_ROM_MODES = ['vanilla', 'downsample', 'pixel', 'rectangle']

# iterate over all the ROM modes, worlds (1-8), and stages (1-4)
for version, rom_mode in enumerate(_ROM_MODES):
    for world in range(1, 9):
        for stage in range(1, 5):
            target = (world, stage)
            env_id = _ID_TEMPLATE.format('', world, stage, version)
            _register_mario_stage_env(env_id, rom_mode=rom_mode, target=target)


# Use Gym's make directly (adapter is already applied inside registration)
make = gym.make

__all__ = [make.__name__]

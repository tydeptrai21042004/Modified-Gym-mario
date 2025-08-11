"""Test cases for the gym registered environments."""
from unittest import TestCase
from .._registration import make

def _reset_compat(env, seed=None):
    """Normalize reset() across Gym (obs) and Gym>=0.26/Gymnasium ((obs, info))."""
    out = env.reset(seed=seed) if seed is not None else env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]   # (obs, info) -> obs
    return out          # obs

def _step_compat(env, action):
    """Normalize step() to a 4‑tuple (obs, reward, done, info)."""
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, reward, done, info
    elif isinstance(out, tuple) and len(out) == 4:
        return out
    else:
        raise RuntimeError(f"Unexpected step() return: {type(out)} len={len(out) if hasattr(out,'__len__') else 'n/a'}")

class ShouldMakeEnv:
    """A test case for making an arbitrary environment."""
    # the number of coins at the start
    coins = 0
    # whether flag get is thrown
    flag_get = False
    # the number of lives left
    life = 2
    # the current world
    world = 1
    # the current score
    score = 0
    # the current stage
    stage = 1
    # the amount of time left
    time = 400
    # the x position of Mario
    x_pos = 40
    # the environments ID
    env_id = None
    # the random seed to apply
    seed = None
    # the subset of stages to sample from
    stages = None

    def _test_env(self, env_id, stages):
        if stages is not None:
            env = make(env_id, stages=stages)
        else:
            env = make(env_id)

        _reset_compat(env, seed=self.seed)   # was: env.reset(seed=self.seed)
        s, r, d, i = _step_compat(env, 0)    # was: s, r, d, i = env.step(0)

        self.assertEqual(self.coins, i['coins'])
        self.assertEqual(self.flag_get, i['flag_get'])
        self.assertEqual(self.life, i['life'])
        self.assertEqual(self.world, i['world'])
        self.assertEqual(self.score, i['score'])
        self.assertEqual(self.stage, i['stage'])
        self.assertEqual(self.time, i['time'])
        self.assertEqual(self.x_pos, i['x_pos'])
        env.close()

    def test(self):
        if isinstance(self.env_id, str):
            self._test_env(self.env_id, self.stages)
        elif isinstance(self.env_id, list):
            for env_id in self.env_id:
                self._test_env(env_id, self.stages)






class ShouldMakeSuperMarioBrosLostLevels(ShouldMakeEnv, TestCase):
    # the amount of time left
    time = 400
    # the environments ID for all versions of Super Mario Bros
    env_id = ['SuperMarioBros2-v{}'.format(v) for v in range(2)]


class ShouldMakeSuperMarioBros_1_1(ShouldMakeEnv, TestCase):
    # the current world
    world = 1
    # the current stage
    stage = 1
    # the environments ID
    env_id = ['SuperMarioBros-1-1-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_1_2(ShouldMakeEnv, TestCase):
    # the current world
    world = 1
    # the current stage
    stage = 2
    # the environments ID
    env_id = ['SuperMarioBros-1-2-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_1_3(ShouldMakeEnv, TestCase):
    # the amount of time left
    time = 300
    # the current world
    world = 1
    # the current stage
    stage = 3
    # the environments ID
    env_id = ['SuperMarioBros-1-3-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_1_4(ShouldMakeEnv, TestCase):
    # the amount of time left
    time = 300
    # the current world
    world = 1
    # the current stage
    stage = 4
    # the environments ID
    env_id = ['SuperMarioBros-1-4-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_2_1(ShouldMakeEnv, TestCase):
    # the current world
    world = 2
    # the current stage
    stage = 1
    # the environments ID
    env_id = ['SuperMarioBros-2-1-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_2_2(ShouldMakeEnv, TestCase):
    # the current world
    world = 2
    # the current stage
    stage = 2
    # the environments ID
    env_id = ['SuperMarioBros-2-2-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_2_3(ShouldMakeEnv, TestCase):
    # the amount of time left
    time = 300
    # the current world
    world = 2
    # the current stage
    stage = 3
    # the environments ID
    env_id = ['SuperMarioBros-2-3-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_2_4(ShouldMakeEnv, TestCase):
    # the amount of time left
    time = 300
    # the current world
    world = 2
    # the current stage
    stage = 4
    # the environments ID
    env_id = ['SuperMarioBros-2-4-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_3_1(ShouldMakeEnv, TestCase):
    # the current world
    world = 3
    # the current stage
    stage = 1
    # the environments ID
    env_id = ['SuperMarioBros-3-1-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_3_2(ShouldMakeEnv, TestCase):
    # the amount of time left
    time = 300
    # the current world
    world = 3
    # the current stage
    stage = 2
    # the environments ID
    env_id = ['SuperMarioBros-3-2-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_3_3(ShouldMakeEnv, TestCase):
    # the amount of time left
    time = 300
    # the current world
    world = 3
    # the current stage
    stage = 3
    # the environments ID
    env_id = ['SuperMarioBros-3-3-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_3_4(ShouldMakeEnv, TestCase):
    # the amount of time left
    time = 300
    # the current world
    world = 3
    # the current stage
    stage = 4
    # the environments ID
    env_id = ['SuperMarioBros-3-4-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_4_1(ShouldMakeEnv, TestCase):
    # the current world
    world = 4
    # the current stage
    stage = 1
    # the environments ID
    env_id = ['SuperMarioBros-4-1-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_4_2(ShouldMakeEnv, TestCase):
    # the current world
    world = 4
    # the current stage
    stage = 2
    # the environments ID
    env_id = ['SuperMarioBros-4-2-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_4_3(ShouldMakeEnv, TestCase):
    # the amount of time left
    time = 300
    # the current world
    world = 4
    # the current stage
    stage = 3
    # the environments ID
    env_id = ['SuperMarioBros-4-3-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_4_4(ShouldMakeEnv, TestCase):
    # the current world
    world = 4
    # the current stage
    stage = 4
    # the environments ID
    env_id = ['SuperMarioBros-4-4-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_5_1(ShouldMakeEnv, TestCase):
    # the amount of time left
    time = 300
    # the current world
    world = 5
    # the current stage
    stage = 1
    # the environments ID
    env_id = ['SuperMarioBros-5-1-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_5_2(ShouldMakeEnv, TestCase):
    # the current world
    world = 5
    # the current stage
    stage = 2
    # the environments ID
    env_id = ['SuperMarioBros-5-2-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_5_3(ShouldMakeEnv, TestCase):
    # the amount of time left
    time = 300
    # the current world
    world = 5
    # the current stage
    stage = 3
    # the environments ID
    env_id = ['SuperMarioBros-5-3-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_5_4(ShouldMakeEnv, TestCase):
    # the amount of time left
    time = 300
    # the current world
    world = 5
    # the current stage
    stage = 4
    # the environments ID
    env_id = ['SuperMarioBros-5-4-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_6_1(ShouldMakeEnv, TestCase):
    # the current world
    world = 6
    # the current stage
    stage = 1
    # the environments ID
    env_id = ['SuperMarioBros-6-1-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_6_2(ShouldMakeEnv, TestCase):
    # the current world
    world = 6
    # the current stage
    stage = 2
    # the environments ID
    env_id = ['SuperMarioBros-6-2-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_6_3(ShouldMakeEnv, TestCase):
    # the amount of time left
    time = 300
    # the current world
    world = 6
    # the current stage
    stage = 3
    # the environments ID
    env_id = ['SuperMarioBros-6-3-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_6_4(ShouldMakeEnv, TestCase):
    # the amount of time left
    time = 300
    # the current world
    world = 6
    # the current stage
    stage = 4
    # the environments ID
    env_id = ['SuperMarioBros-6-4-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_7_1(ShouldMakeEnv, TestCase):
    # the current world
    world = 7
    # the current stage
    stage = 1
    # the environments ID
    env_id = ['SuperMarioBros-7-1-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_7_2(ShouldMakeEnv, TestCase):
    # the current world
    world = 7
    # the current stage
    stage = 2
    # the environments ID
    env_id = ['SuperMarioBros-7-2-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_7_3(ShouldMakeEnv, TestCase):
    # the amount of time left
    time = 300
    # the current world
    world = 7
    # the current stage
    stage = 3
    # the environments ID
    env_id = ['SuperMarioBros-7-3-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_7_4(ShouldMakeEnv, TestCase):
    # the current world
    world = 7
    # the current stage
    stage = 4
    # the environments ID
    env_id = ['SuperMarioBros-7-4-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_8_1(ShouldMakeEnv, TestCase):
    # the amount of time left
    time = 300
    # the current world
    world = 8
    # the current stage
    stage = 1
    # the environments ID
    env_id = ['SuperMarioBros-8-1-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_8_2(ShouldMakeEnv, TestCase):
    # the current world
    world = 8
    # the current stage
    stage = 2
    # the environments ID
    env_id = ['SuperMarioBros-8-2-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_8_3(ShouldMakeEnv, TestCase):
    # the amount of time left
    time = 300
    # the current world
    world = 8
    # the current stage
    stage = 3
    # the environments ID
    env_id = ['SuperMarioBros-8-3-v{}'.format(v) for v in range(4)]


class ShouldMakeSuperMarioBros_8_4(ShouldMakeEnv, TestCase):
    # the current world
    world = 8
    # the current stage
    stage = 4
    # the environments ID
    env_id = ['SuperMarioBros-8-4-v{}'.format(v) for v in range(4)]



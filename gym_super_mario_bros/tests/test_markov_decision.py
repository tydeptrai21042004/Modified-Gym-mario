import unittest
import numpy as np

from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT



def bucket_xpos(x, bin_size=64, max_bins=16):
    b = int(x // bin_size)
    return b if b < max_bins else max_bins - 1


def api_reset(env):
    r = env.reset()
    # Gym >=0.26: (obs, info); older Gym: obs
    if isinstance(r, tuple) and len(r) == 2:
        obs, info = r
        return obs, info
    return r, {}  # no info available on old Gym reset

def api_step(env, a):
    r = env.step(a)
    # Gym >=0.26: (obs, reward, terminated, truncated, info)
    if isinstance(r, tuple) and len(r) == 5:
        obs, reward, terminated, truncated, info = r
        done = bool(terminated) or bool(truncated)
        return obs, reward, done, info
    # Old Gym: (obs, reward, done, info)
    obs, reward, done, info = r
    return obs, reward, bool(done), info
# -----------------------------------------------------------------------------


class TestMDPOnMario(unittest.TestCase):
    def test_learned_mdp_prefers_right(self):
        env = SuperMarioBrosEnv()
        env = JoypadSpace(env, COMPLEX_MOVEMENT)

        try:
            nA = env.action_space.n
            nS = 16

            # --- stats with Laplace smoothing
            N_STEPS = 2000
            alpha = 1.0
            trans_counts = np.full((nS, nA, nS), alpha, dtype=np.float64)
            rew_sums     = np.zeros((nS, nA), dtype=np.float64)
            rew_counts   = np.full((nS, nA), alpha, dtype=np.float64)

            rng = np.random.default_rng(0)

            # reset and get an initial x_pos
            obs, info = api_reset(env)
            # take one step to ensure info has x_pos
            obs, r, done, info = api_step(env, 0)
            if done:
                obs, info = api_reset(env)
                obs, r, done, info = api_step(env, 0)

            x_prev = float(info.get('x_pos', 0.0))
            s      = bucket_xpos(x_prev)

            for _ in range(N_STEPS):
                a = int(rng.integers(nA))  # random exploration

                obs, _, done, info = api_step(env, a)
                x_next = float(info.get('x_pos', x_prev))
                s_next = bucket_xpos(x_next)

                r_dx = x_next - x_prev

                trans_counts[s, a, s_next] += 1.0
                rew_sums[s, a]  += r_dx
                rew_counts[s, a] += 1.0

                s, x_prev = s_next, x_next

                if done:
                    obs, info = api_reset(env)
                    obs, _, done, info = api_step(env, 0)
                    x_prev = float(info.get('x_pos', 0.0))
                    s      = bucket_xpos(x_prev)

            P = trans_counts / trans_counts.sum(axis=2, keepdims=True)
            R = rew_sums / np.maximum(rew_counts, 1e-6)

            # value iteration
            gamma = 0.95
            V = np.zeros(nS, dtype=np.float64)
            for _ in range(200):
                V_new = np.empty_like(V)
                for si in range(nS):
                    q_vals = [R[si, ai] + gamma * (P[si, ai] @ V) for ai in range(nA)]
                    V_new[si] = max(q_vals)
                if np.max(np.abs(V_new - V)) < 1e-6:
                    V = V_new
                    break
                V = V_new

            # greedy policy
            pi = np.zeros(nS, dtype=int)
            for si in range(nS):
                q_vals = [R[si, ai] + gamma * (P[si, ai] @ V) for ai in range(nA)]
                pi[si] = int(np.argmax(q_vals))

            # assertions
            for si in range(0, 4):  # early x bins
                move = COMPLEX_MOVEMENT[pi[si]]
                self.assertTrue('right' in move, f"Expected rightward action in state {si}, got {move}")

            self.assertTrue(np.allclose(P.sum(axis=2), 1.0, atol=1e-6))
            self.assertTrue(np.isfinite(R).all())

        finally:
            env.close()

"""
Exact Metropolis–Hastings tests.

We implement a textbook MH kernel:
- Target: user-supplied logpi(a)  (π up to normalization)
- Proposal: user-supplied sampler q(a'|a) and log q(a'|a)
- Accept: α = min(1, [π(a') q(a|a')] / [π(a) q(a'|a)])

Includes:
  (A) Detailed-balance sanity check on a tiny discrete space.
  (B) Mario smoke test with a fixed, hand-crafted π over actions.
"""
import unittest
import math
import numpy as np

# ---- Generic, exact MH kernel ------------------------------------------------
class MetropolisHastings:
    def __init__(self, n_actions, logpi_fn, prop_sampler_fn, logq_fn, seed=0, a0=None):
        """
        n_actions: int
        logpi_fn(a): returns log π(a) (unnormalized ok)
        prop_sampler_fn(a_cur, rng)->a_prop: sample from q(a'|a_cur)
        logq_fn(a_from, a_to): returns log q(a_to | a_from)
        """
        self.nA = int(n_actions)
        self.logpi_fn = logpi_fn
        self.prop_sampler_fn = prop_sampler_fn
        self.logq_fn = logq_fn
        self.rng = np.random.default_rng(seed)
        self.a = int(a0) if a0 is not None else int(self.rng.integers(self.nA))

    def step(self):
        a_cur = self.a
        a_prop = int(self.prop_sampler_fn(a_cur, self.rng))

        logpi_cur  = float(self.logpi_fn(a_cur))
        logpi_prop = float(self.logpi_fn(a_prop))
        logq_cur_given_prop  = float(self.logq_fn(a_prop, a_cur))
        logq_prop_given_cur  = float(self.logq_fn(a_cur, a_prop))

        # α = min(1, exp( logπ' - logπ + logq_back - logq_fwd ))
        log_alpha = (logpi_prop - logpi_cur) + (logq_cur_given_prop - logq_prop_given_cur)
        accept = (math.log(self.rng.random()) < min(0.0, log_alpha))
        if accept:
            self.a = a_prop
        return self.a, accept

# ---- (A) Detailed-balance test ----------------------------------------------
class TestExactMH_DetailedBalance(unittest.TestCase):
    def test_stationary_matches_target_uniform_q(self):
        # Target over 3 actions (unnormalized OK)
        p = np.array([0.1, 0.3, 0.6], dtype=float)
        logpi = np.log(p)

        def logpi_fn(a): return logpi[a]

        # Uniform proposal q(a'|a)=1/n
        nA = 3
        def prop_sampler(a, rng): return int(rng.integers(nA))
        def logq(a_from, a_to): return -math.log(nA)

        mh = MetropolisHastings(n_actions=nA, logpi_fn=logpi_fn,
                                prop_sampler_fn=prop_sampler,
                                logq_fn=logq, seed=123, a0=0)

        # Burn-in, then sample
        for _ in range(2000):
            mh.step()
        counts = np.zeros(nA, dtype=int)
        N = 20000
        for _ in range(N):
            a, _ = mh.step()
            counts[a] += 1

        est = counts / counts.sum()
        # Expect close to true p (loose tolerance to keep test robust)
        np.testing.assert_allclose(est, p / p.sum(), atol=0.02, rtol=0.05)

# ---- (B) Mario smoke test with fixed π ---------------------------------------
class TestExactMH_Mario(unittest.TestCase):
    def test_runs_with_fixed_target(self):
        import gym_super_mario_bros
        from nes_py.wrappers import JoypadSpace
        from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

        # Build env
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, COMPLEX_MOVEMENT)

        # --- helpers to be API-agnostic -----------------------------------
        def api_reset(e):
            r = e.reset()
            # v0.26+: (obs, info), old gym: obs
            if isinstance(r, tuple) and len(r) == 2:
                return r[0]
            return r

        def api_step(e, a):
            r = e.step(a)
            # v0.26+: (obs, reward, terminated, truncated, info)
            if isinstance(r, tuple) and len(r) == 5:
                obs, reward, terminated, truncated, info = r
                done = bool(terminated) or bool(truncated)
                return obs, reward, done, info
            # old gym: (obs, reward, done, info)
            obs, reward, done, info = r
            return obs, reward, bool(done), info
        # -------------------------------------------------------------------

        try:
            assert hasattr(env.action_space, 'n')
            nA = env.action_space.n

            # Fixed target π over actions: prefer moves containing "right"
            beta = 1.0
            moves = COMPLEX_MOVEMENT
            pref = np.array([1.0 if ('right' in m) else 0.0 for m in moves], dtype=float)

            def logpi_fn(a): return beta * float(pref[a])

            # Symmetric uniform proposal
            def prop_sampler(a, rng): return int(rng.integers(nA))
            def logq(a_from, a_to): return -math.log(nA)

            mh = MetropolisHastings(
                n_actions=nA,
                logpi_fn=logpi_fn,
                prop_sampler_fn=prop_sampler,
                logq_fn=logq,
                seed=7
            )

            obs = api_reset(env)
            steps, accepts = 0, 0
            for _ in range(60):
                a_prev = mh.a
                a, acc = mh.step()
                if acc:
                    accepts += 1
                obs, reward, done, info = api_step(env, a)
                steps += 1
                if done:
                    break

            self.assertGreater(steps, 0)
            self.assertGreaterEqual(accepts, 0)
        finally:
            env.close()


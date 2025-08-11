import unittest
import numpy as np

from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT


def api_reset(env):
    r = env.reset()
    if isinstance(r, tuple) and len(r) == 2:
        obs, info = r
        return obs, info
    return r, {}


def api_step(env, a):
    r = env.step(a)
    if isinstance(r, tuple) and len(r) == 5:
        obs, reward, terminated, truncated, info = r
        done = bool(terminated) or bool(truncated)
        return obs, reward, done, info
    obs, reward, done, info = r
    return obs, reward, bool(done), info


def action_label(i):
    return '+'.join(COMPLEX_MOVEMENT[i])


class TestMDPOnMario(unittest.TestCase):
    def test_learned_mdp_prefers_right(self):
        # ----------------- CONFIG -----------------
        N_STEPS    = 2000         # exploration steps
        GAMMA      = 0.95
        ALPHA      = 1.0            # Laplace smoothing
        RNG_SEED   = 0
        SHOW_TOP_K = 5

        # Use env-provided discrete state, but **project** to a small subset
        # to keep tabular model size sane.
        # Feature indices (as defined in your env.state_features()):
        # 0:x_bin, 1:y_bin, 2:vx_bin, 3:vy_bin, 4:status_bin, 5:pstate_bin,
        # 6:life_bin, 7:coins_bin, 8:time_bin, 9:world_bin, 10:stage_bin,
        # 11:area_bin, 12:busy_bin, 13:flag_bin
        SELECT_FEATS = (0, 2, 4)    # (x_bin, vx_bin, status_bin)
        # ------------------------------------------

        env = SuperMarioBrosEnv()
        env = JoypadSpace(env, COMPLEX_MOVEMENT)

        try:
            nA = env.action_space.n

            # --- Prime env & grab dims/first feat ---
            obs, info = api_reset(env)
            # ensure info is populated
            obs, _, _, info = api_step(env, 0)

            state_dims_full = tuple(info.get('state_dims', (16,)))   # fallback
            state_feat      = tuple(info.get('state_feat', (0,)))    # fallback

            sel_dims = tuple(state_dims_full[i] for i in SELECT_FEATS)

            def encode(feats_tuple):
                """mixed-radix encode on selected features."""
                idx = 0
                base = 1
                for f, d in zip(feats_tuple, sel_dims):
                    if f < 0: f = 0
                    if f >= d: f = d - 1
                    idx += f * base
                    base *= d
                return int(idx)

            def decode(idx):
                """inverse of encode."""
                vals = []
                for d in sel_dims:
                    vals.append(idx % d)
                    idx //= d
                return tuple(vals)

            nS = int(np.prod(sel_dims))

            # --- Debug header ---
            print("\n========== DEBUG: Mario MDP Learner (Env State) ==========")
            print(f"Full state_dims     : {state_dims_full}")
            print(f"Selected feat idx   : {SELECT_FEATS}")
            print(f"Selected dims       : {sel_dims}  -> nS={nS}")
            print(f"Example state_feat  : {state_feat}")
            sel_example = tuple(state_feat[i] for i in SELECT_FEATS)
            print(f"Example selected    : {sel_example} -> enc={encode(sel_example)}")

            print(f"\nNum actions         : {nA}")
            print("\nAction space (index -> buttons):")
            for i, m in enumerate(COMPLEX_MOVEMENT):
                print(f"  {i:2d}: {m}")
            print(f"\nSanity: action 0 is {COMPLEX_MOVEMENT[0]}\n")

            # --- stats with Laplace smoothing ---
            trans_counts = np.full((nS, nA, nS), ALPHA, dtype=np.float64)
            rew_sums     = np.zeros((nS, nA), dtype=np.float64)
            rew_counts   = np.full((nS, nA), ALPHA, dtype=np.float64)

            rng = np.random.default_rng(RNG_SEED)
            obs, info = api_reset(env)
            # take one step to ensure info has x_pos
            obs, r, done, info = api_step(env, 0)
            if done:
                obs, info = api_reset(env)
                obs, r, done, info = api_step(env, 0)
            # initialize s & x
            x_prev = float(info.get('x_pos', 0.0))
            s_prev_full = tuple(info.get('state_feat', (0,)))
            s_prev_sel  = tuple(s_prev_full[i] for i in SELECT_FEATS)
            s           = encode(s_prev_sel)

            # --- random exploration loop ---
            for _ in range(N_STEPS):
                a = int(rng.integers(nA))  # random action

                # get env's shaped reward
                obs, r, done, info = api_step(env, a)          # CHANGED: capture r
                x_next = float(info.get('x_pos', x_prev))      # (still fine to keep for debug)

                s_full = tuple(info.get('state_feat', s_prev_full))
                s_sel  = tuple(s_full[i] for i in SELECT_FEATS)
                s_next = encode(s_sel)

                r_env = float(r)                               # CHANGED: use env reward
                trans_counts[s, a, s_next] += 1.0
                rew_sums[s, a]  += r_env                       # CHANGED
                rew_counts[s, a] += 1.0

                s, x_prev, s_prev_full = s_next, x_next, s_full

                if done:
                    obs, info = api_reset(env)
                    obs, _, done, info = api_step(env, 0)
                    x_prev = float(info.get('x_pos', 0.0))
                    s_prev_full = tuple(info.get('state_feat', (0,)))
                    s_prev_sel  = tuple(s_prev_full[i] for i in SELECT_FEATS)
                    s           = encode(s_prev_sel)

            # Build model P, R
            P = trans_counts / trans_counts.sum(axis=2, keepdims=True)
            R = rew_sums / np.maximum(rew_counts, 1e-6)

            # --- Value Iteration ---
            V = np.zeros(nS, dtype=np.float64)
            for _ in range(200):
                V_new = np.empty_like(V)
                for si in range(nS):
                    q_vals = [R[si, ai] + GAMMA * (P[si, ai] @ V) for ai in range(nA)]
                    V_new[si] = max(q_vals)
                if np.max(np.abs(V_new - V)) < 1e-6:
                    V = V_new
                    break
                V = V_new

            # Greedy policy
            pi = np.zeros(nS, dtype=int)
            for si in range(nS):
                q_vals = [R[si, ai] + GAMMA * (P[si, ai] @ V) for ai in range(nA)]
                pi[si] = int(np.argmax(q_vals))

            # ----------------- DEBUG helpers -----------------
            def dump_q_state(si):
                q_vals = np.array([R[si, ai] + GAMMA * (P[si, ai] @ V) for ai in range(nA)])
                order  = np.argsort(q_vals)[::-1]
                best   = order[0]
                sel    = decode(si)
                print(f"\nState {si} sel={sel}: greedy -> {action_label(best)} (Q={q_vals[best]:.6f})")
                print(f"Top-{SHOW_TOP_K} actions:")
                for k in order[:SHOW_TOP_K]:
                    print(f"  {action_label(k):>14s}  Q={q_vals[k]: .6f}")

            # Pick sample states where x_bin in {0,1,2,3} and vx/status = 0 bins
            # so we have a reproducible "early" slice to inspect & assert on.
            sample_states = []
            for xb in range(min(sel_dims[0], 4)):      # x_bin 0..3
                for vxb in [1]:                        # vx_bin == 1 (zero velocity)
                    for sb in [0, 1, 2]:               # status small/tall/fireball
                        if vxb < sel_dims[1] and sb < sel_dims[2]:
                            si = encode((xb, vxb, sb))
                            sample_states.append(si)

            # --- DEBUG DUMPS ---
            # Show R and P for one canonical early state (x_bin=0, vx=0, status=small if valid)
            probe = None
            if sel_dims[0] > 0:
                xb0 = 0
                vxb0 = 1 if sel_dims[1] > 1 else 0
                sb0  = 0 if sel_dims[2] > 0 else 0
                if xb0 < sel_dims[0] and vxb0 < sel_dims[1] and sb0 < sel_dims[2]:
                    probe = encode((xb0, vxb0, sb0))

            if probe is not None:
                print("\n--- DEBUG: R[probe, :] (avg dx per action) ---")
                for ai in range(nA):
                    print(f"  {action_label(ai):>14s}: R[{probe},{ai}]={R[probe, ai]: .6f}  (n={int(rew_counts[probe, ai])})")

                print("\n--- DEBUG: P[probe, a, :] mass on next states (top) ---")
                for ai in range(nA):
                    top_next = np.argsort(P[probe, ai])[::-1][:SHOW_TOP_K]
                    mass = ", ".join([f"s{j}:{P[probe,ai,j]:.3f}" for j in top_next])
                    print(f"  {action_label(ai):>14s}: {mass}")

            print("\n--- DEBUG: Q-values for a few early selected states ---")
            for si in sample_states[:4]:
                dump_q_state(si)

            # ----------------- ASSERTIONS -----------------
            # Require a rightward action for *any* state whose x_bin in {0..3}
            # (vx, status arbitrary) – we test a small set to keep runtime sane.
            for si in sample_states:
                move = COMPLEX_MOVEMENT[pi[si]]
                xb, vxb, sb = decode(si)
                self.assertTrue('right' in move, f"Expected rightward action at x_bin={xb}, vx_bin={vxb}, status_bin={sb}; got {move}")

            self.assertTrue(np.allclose(P.sum(axis=2), 1.0, atol=1e-6))
            self.assertTrue(np.isfinite(R).all())

        finally:
            env.close()

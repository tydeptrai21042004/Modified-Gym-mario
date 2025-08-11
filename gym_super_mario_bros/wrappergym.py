import gym

class V0toV26(gym.Wrapper):
    """Adapt classic 4‑tuple envs to Gym>=0.26 5‑tuple API."""

    def reset(self, *, seed=None, options=None):
        try:
            out = self.env.reset(seed=seed, options=options)
        except TypeError:
            # old envs ignore keyword seed/options
            if seed is not None and hasattr(self.env, "seed"):
                self.env.seed(seed)
            out = self.env.reset()
        
        # Pass through Gymnasium-style (obs, info)
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}  # old gym: return (obs, info)

    def step(self, action):
        out = self.env.step(action)
        # Pass through if already 5-tuple
        if isinstance(out, tuple) and len(out) == 5:
            return out
        # Old gym 4-tuple -> adapt
        obs, reward, done, info = out
        terminated = bool(done)
        truncated = False
        return obs, reward, terminated, truncated, info

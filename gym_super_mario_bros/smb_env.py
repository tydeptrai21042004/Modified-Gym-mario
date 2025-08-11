"""An OpenAI Gym environment for Super Mario Bros. and Lost Levels."""
from collections import defaultdict
from nes_py import NESEnv
import numpy as np
from ._roms import decode_target
from ._roms import rom_path


# create a dictionary mapping value of status register to string names
_STATUS_MAP = defaultdict(lambda: 'fireball', {0:'small', 1: 'tall'})
reward_range = (-20, 20)


# a set of state values indicating that Mario is "busy"
_BUSY_STATES = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07]


# RAM addresses for enemy types on the screen
_ENEMY_TYPE_ADDRESSES = [0x0016, 0x0017, 0x0018, 0x0019, 0x001A]


# enemies whose context indicate that a stage change will occur (opposed to an
# enemy that implies a stage change wont occur -- i.e., a vine)
# Bowser = 0x2D
# Flagpole = 0x31
_STAGE_OVER_ENEMIES = np.array([0x2D, 0x31])
# ===== Physics-related RAM addresses (SMB PRG0) =====
# Docs: Data Crystal SMB RAM map / notes
#   0x0709: current gravity applied while rising
#   0x070A: gravity used while falling
#   0x0456: player max velocity to the right (0x1C walk, 0x30 run)
#   0x0450: player max velocity to the left  (0xE4 walk, 0xD8 run; two’s complement)
#   0x0701: friction "adder high" (1 = strong braking)
#   0x0705: X-MoveForce (accel when holding left/right)
#   0x0057: current horizontal speed (signed, informational)
#   0x009F: current vertical velocity (signed, informational)
#   0x0433: vertical fractional velocity (informational)
# Source: Data Crystal SMB RAM map. See citations below.
GRAVITY_RISE_ADDR = 0x0709
GRAVITY_FALL_ADDR = 0x070A
MAX_VEL_RIGHT_ADDR = 0x0456
MAX_VEL_LEFT_ADDR  = 0x0450
FRICTION_ADDR      = 0x0701
X_MOVEFORCE_ADDR   = 0x0705

class SuperMarioBrosEnv(NESEnv):
    """An environment for playing Super Mario Bros with OpenAI Gym."""

    # the legal range of rewards for each step
    reward_range = (-15, 15)

    def __init__(
        self,
        rom_mode='vanilla',
        lost_levels=False,
        target=None,
        *,
        gravity_rise=None,   # e.g., 0x30 (still), 0x2D (walk), 0x38 (run)
        gravity_fall=None,   # e.g., 0xA8 (still), 0x90 (walk), 0xD0 (run)
        max_vel_right=None,  # e.g., 0x1C (walk), 0x30 (run)
        max_vel_left=None,   # e.g., 0xE4 (walk), 0xD8 (run)
        friction=None,       # 0 or 1 typically; 1 feels like strong braking
        x_moveforce=None     # accel when holding left/right (experiment)
    ):
        """
        Initialize a new Super Mario Bros environment.

        Args:
            rom_mode (str): the ROM mode to use when loading ROMs from disk
            lost_levels (bool): whether to load the ROM with lost levels.
                - False: load original Super Mario Bros.
                - True: load Super Mario Bros. Lost Levels
            target (tuple): a tuple of the (world, stage) to play as a level

        Returns:
            None

        """
        # decode the ROM path based on mode and lost levels flag
        rom = rom_path(lost_levels, rom_mode)
        # initialize the super object with the ROM path
        super(SuperMarioBrosEnv, self).__init__(rom)
        # set the target world, stage, and area variables
        target = decode_target(target, lost_levels)
        self._target_world, self._target_stage, self._target_area = target
        self._x_position_last = 0
        # setup a variable to keep track of the last frames time
        self._time_last = 0
        self._coins_last = 0
        self._status_last = 'small'
        self._score_last = 0
        self._stuck_frames = 0

        # setup a variable to keep track of the last frames x position
        self._coin_rate_ema = 0.0             # EMA of “got coin” events
        self._coin_rate_alpha = 0.10          # smoothing factor (EMA)
        # weights/values for new terms (tune later)
        self._living_cost_value = -0.01       # constant per-step penalty
        self._life_loss_penalty_value = -10.0 # extra hit when life decreases
        self._coin_smooth_weight = 5.0        # bonus * EMA(got_coin)
        self._phys = dict(
            gravity_rise=gravity_rise,
            gravity_fall=gravity_fall,
            max_vel_right=max_vel_right,
            max_vel_left=max_vel_left,
            friction=friction,
            x_moveforce=x_moveforce,
        )
        # reset the emulator
        self.reset()
        # skip the start screen
        self._skip_start_screen()
        # create a backup state to restore from on subsequent calls to reset
        self._backup()
    def _apply_custom_physics(self):
        """Write custom physics values (if provided) to SMB RAM."""
        ram = self.ram
        p = self._phys
        # gravity applied while rising
        if p["gravity_rise"] is not None:
            ram[GRAVITY_RISE_ADDR] = np.uint8(p["gravity_rise"])
        # gravity applied while falling (copied into 0x0709 by the game when falling)
        if p["gravity_fall"] is not None:
            ram[GRAVITY_FALL_ADDR] = np.uint8(p["gravity_fall"])
        # horizontal speed caps
        if p["max_vel_right"] is not None:
            ram[MAX_VEL_RIGHT_ADDR] = np.uint8(p["max_vel_right"])
        if p["max_vel_left"] is not None:
            ram[MAX_VEL_LEFT_ADDR] = np.uint8(p["max_vel_left"])
        # friction and horizontal acceleration
        if p["friction"] is not None:
            ram[FRICTION_ADDR] = np.uint8(p["friction"])
        if p["x_moveforce"] is not None:
            ram[X_MOVEFORCE_ADDR] = np.uint8(p["x_moveforce"])
    @property
    def is_single_stage_env(self):
        """Return True if this environment is a stage environment."""
        return self._target_world is not None and self._target_area is not None

    # MARK: Memory access

    def _read_mem_range(self, address, length):
        """
        Read a range of bytes where each byte is a 10's place figure.

        Args:
            address (int): the address to read from as a 16 bit integer
            length: the number of sequential bytes to read

        Note:
            this method is specific to Mario where three GUI values are stored
            in independent memory slots to save processing time
            - score has 6 10's places
            - coins has 2 10's places
            - time has 3 10's places

        Returns:
            the integer value of this 10's place representation

        """
        return int(''.join(map(str, self.ram[address:address + length])))

    @property
    def _level(self):
        """Return the level of the game."""
        return self.ram[0x075f] * 4 + self.ram[0x075c]

    @property
    def _world(self):
        """Return the current world (1 to 8)."""
        return self.ram[0x075f] + 1

    @property
    def _stage(self):
        """Return the current stage (1 to 4)."""
        return self.ram[0x075c] + 1

    @property
    def _area(self):
        """Return the current area number (1 to 5)."""
        return self.ram[0x0760] + 1

    @property
    def _score(self):
        """Return the current player score (0 to 999990)."""
        # score is represented as a figure with 6 10's places
        return self._read_mem_range(0x07de, 6)

    @property
    def _time(self):
        """Return the time left (0 to 999)."""
        # time is represented as a figure with 3 10's places
        return self._read_mem_range(0x07f8, 3)

    @property
    def _coins(self):
        """Return the number of coins collected (0 to 99)."""
        # coins are represented as a figure with 2 10's places
        return self._read_mem_range(0x07ed, 2)

    @property
    def _life(self):
        """Return the number of remaining lives."""
        return self.ram[0x075a]

    @property
    def _x_position(self):
        """Return the current horizontal position."""
        # add the current page 0x6d to the current x
        return self.ram[0x6d] * 0x100 + self.ram[0x86]

    @property
    def _left_x_position(self):
        """Return the number of pixels from the left of the screen."""
        """Done why it still have TODO"""
        # TODO: resolve RuntimeWarning: overflow encountered in ubyte_scalars
        # subtract the left x position 0x071c from the current x 0x86
        # return (self.ram[0x86] - self.ram[0x071c]) % 256
        return np.uint8(int(self.ram[0x86]) - int(self.ram[0x071c])) % 256

    @property
    def _y_pixel(self):
        """Return the current vertical position."""
        return self.ram[0x03b8]

    @property
    def _y_viewport(self):
        """
        Return the current y viewport.

        Note:
            1 = in visible viewport
            0 = above viewport
            > 1 below viewport (i.e. dead, falling down a hole)
            up to 5 indicates falling into a hole

        """
        return self.ram[0x00b5]

    @property
    def _y_position(self):
        """Return the current vertical position."""
        # check if Mario is above the viewport (the score board area)
        if self._y_viewport < 1:
            # y position overflows so we start from 255 and add the offset
            return 255 + (255 - self._y_pixel)
        # invert the y pixel into the distance from the bottom of the screen
        return 255 - self._y_pixel

    @property
    def _player_status(self):
        """Return the player status as a string."""
        return _STATUS_MAP[self.ram[0x0756]]

    @property
    def _player_state(self):
        """
        Return the current player state.

        Note:
            0x00 : Leftmost of screen
            0x01 : Climbing vine
            0x02 : Entering reversed-L pipe
            0x03 : Going down a pipe
            0x04 : Auto-walk
            0x05 : Auto-walk
            0x06 : Dead
            0x07 : Entering area
            0x08 : Normal
            0x09 : Cannot move
            0x0B : Dying
            0x0C : Palette cycling, can't move

        """
        return self.ram[0x000e]

    @property
    def _is_dying(self):
        """Return True if Mario is in dying animation, False otherwise."""
        return self._player_state == 0x0b or self._y_viewport > 1

    @property
    def _is_dead(self):
        """Return True if Mario is dead, False otherwise."""
        return self._player_state == 0x06

    @property
    def _is_game_over(self):
        """Return True if the game has ended, False otherwise."""
        # the life counter will get set to 255 (0xff) when there are no lives
        # left. It goes 2, 1, 0 for the 3 lives of the game
        return self._life == 0xff

    @property
    def _is_busy(self):
        """Return boolean whether Mario is busy with in-game garbage."""
        return self._player_state in _BUSY_STATES

    @property
    def _is_world_over(self):
        """Return a boolean determining if the world is over."""
        # 0x0770 contains GamePlay mode:
        # 0 => Demo
        # 1 => Standard
        # 2 => End of world
        return self.ram[0x0770] == 2

    @property
    def _is_stage_over(self):
        """Return a boolean determining if the level is over."""
        # iterate over the memory addresses that hold enemy types
        for address in _ENEMY_TYPE_ADDRESSES:
            # check if the byte is either Bowser (0x2D) or a flag (0x31)
            # this is to prevent returning true when Mario is using a vine
            # which will set the byte at 0x001D to 3
            if self.ram[address] in _STAGE_OVER_ENEMIES:
                # player float state set to 3 when sliding down flag pole
                return self.ram[0x001D] == 3

        return False

    @property
    def _flag_get(self):
        """Return a boolean determining if the agent reached a flag."""
        return self._is_world_over or self._is_stage_over

    # MARK: RAM Hacks

    def _write_stage(self):
        """Write the stage data to RAM to overwrite loading the next stage."""
        self.ram[0x075f] = self._target_world - 1
        self.ram[0x075c] = self._target_stage - 1
        self.ram[0x0760] = self._target_area - 1

    def _runout_prelevel_timer(self):
        """Force the pre-level timer to 0 to skip frames during a death."""
        self.ram[0x07A0] = 0

    def _skip_change_area(self):
        """Skip change area animations by by running down timers."""
        change_area_timer = self.ram[0x06DE]
        if change_area_timer > 1 and change_area_timer < 255:
            self.ram[0x06DE] = 1

    def _skip_occupied_states(self):
        """Skip occupied states by running out a timer and skipping frames."""
        while self._is_busy or self._is_world_over:
            self._runout_prelevel_timer()
            self._frame_advance(0)

    def _skip_start_screen(self):
        """Press and release start to skip the start screen."""
        # press and release the start button
        self._frame_advance(8)
        self._frame_advance(0)
        # Press start until the game starts
        while self._time == 0:
            # press and release the start button
            self._frame_advance(8)
            # if we're in the single stage, environment, write the stage data
            if self.is_single_stage_env:
                self._write_stage()
            self._frame_advance(0)
            # run-out the prelevel timer to skip the animation
            self._runout_prelevel_timer()
        # set the last time to now
        self._time_last = self._time
        # after the start screen idle to skip some extra frames
        while self._time >= self._time_last:
            self._time_last = self._time
            self._frame_advance(8)
            self._frame_advance(0)

    def _skip_end_of_world(self):
        """Skip the cutscene that plays at the end of a world."""
        if self._is_world_over:
            # get the current game time to reference
            time = self._time
            # loop until the time is different
            while self._time == time:
                # frame advance with NOP
                self._frame_advance(0)
    def _progress_reward(self):
        """Potential-based shaping using x as potential (safe)."""
        dx = self._x_position - self._x_position_last
        self._x_position_last = self._x_position
        # ignore teleport-like jumps after death/warp
        if dx < -5 or dx > 5:
            dx = 0
        # saturate small per-frame reward to avoid exploding gradients
        r = float(np.clip(dx, -1, 1))
        # stuck counter (no rightward movement)
        if dx <= 0:
            self._stuck_frames += 1
        else:
            self._stuck_frames = 0
        return r

    def _coin_reward(self, w=5.0):
        dc = self._coins - self._coins_last
        self._coins_last = self._coins
        return w * max(dc, 0)

    def _status_reward(self, w_gain=10.0, w_loss=-10.0):
        rank = {'small': 0, 'tall': 1, 'fireball': 2}
        now, last = rank.get(self._player_status, 0), rank.get(self._status_last, 0)
        self._status_last = self._player_status
        if now > last:
            return w_gain * (now - last)      # mushroom / flower
        if now < last:
            return w_loss                      # got hit
        return 0.0

    def _flag_bonus(self):
        # sparse success bonus; include tiny time-left term to prefer faster clears
        return 50.0 + 0.1 * self._time if self._flag_get else 0.0

    def _score_reward(self, w=0.01):
        # very small weight; score rises for stomps, bricks, coins, etc.
        ds = self._score - self._score_last
        self._score_last = self._score
        return w * max(ds, 0)

    def _stuck_penalty(self):
        # discourage dithering: after ~0.75s (45 frames) without progress, nudge
        return -0.5 if self._stuck_frames >= 45 else 0.0
    def _kill_mario(self):
        """Skip a death animation by forcing Mario to death."""
        # force Mario's state to dead
        self.ram[0x000e] = 0x06
        # step forward one frame
        self._frame_advance(0)

    # MARK: Reward Function

    @property
    def _x_reward(self):
        """Return the reward based on left right movement between steps."""
        _reward = self._x_position - self._x_position_last
        self._x_position_last = self._x_position
        # TODO: check whether this is still necessary
        # resolve an issue where after death the x position resets. The x delta
        # is typically has at most magnitude of 3, 5 is a safe bound
        if _reward < -5 or _reward > 5:
            return 0

        return _reward

    @property
    def _time_penalty(self):
        """Return the reward for the in-game clock ticking."""
        _reward = self._time - self._time_last
        self._time_last = self._time
        # time can only decrease, a positive reward results from a reset and
        # should default to 0 reward
        if _reward > 0:
            return 0

        return _reward

    @property
    def _death_penalty(self):
        """Return the reward earned by dying."""
        if self._is_dying or self._is_dead:
            return -15

        return 0

    # MARK: nes-py API calls

    def _will_reset(self):
        """Handle and RAM hacking before a reset occurs."""
        self._time_last = 0
        self._x_position_last = 0
        self._coins_last = self._coins
        self._status_last = self._player_status
        self._score_last = self._score
        self._stuck_frames = 0
        self._life_last = self._life
        self._coin_rate_ema = 0.0
    def _did_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        self._time_last = self._time
        self._x_position_last = self._x_position
        self._life_last = self._life

        self._apply_custom_physics()
    def _did_step(self, done):
        """
        Handle any RAM hacking after a step occurs.

        Args:
            done: whether the done flag is set to true

        Returns:
            None

        """
        # if done flag is set a reset is incoming anyway, ignore any hacking
        if done:
            return
        self._apply_custom_physics()
        # if mario is dying, then cut to the chase and kill hi,
        if self._is_dying:
            self._kill_mario()
        # skip world change scenes (must call before other skip methods)
        if not self.is_single_stage_env:
            self._skip_end_of_world()
        # skip area change (i.e. enter pipe, flag get, etc.)
        self._skip_change_area()
        # skip occupied states like the black screen between lives that shows
        # how many lives the player has left
        self._skip_occupied_states()
    def _living_cost(self):
        """Small constant penalty per step to encourage efficiency."""
        return float(self._living_cost_value)

    def _life_loss_penalty(self):
        """Extra penalty whenever the life counter drops."""
        penalty = 0.0
        if self._life < self._life_last:
            penalty = float(self._life_loss_penalty_value)
        self._life_last = self._life
        return penalty

    def _coin_rate_bonus(self):
        """
        Bonus from a smoothed 'coin rate' (EMA of coin events).
        Uses dc vs. last without consuming the last-coin side effect.
        """
        dc = self._coins - self._coins_last
        got_coin = 1.0 if dc > 0 else 0.0
        self._coin_rate_ema = (
            self._coin_rate_alpha * got_coin
            + (1.0 - self._coin_rate_alpha) * self._coin_rate_ema
        )
        return float(self._coin_smooth_weight * self._coin_rate_ema)

    def _get_reward(self):
        """
        r = progress + 0.1*time_penalty + death_penalty
            + coin + status + tiny*score + anti_stuck + flag_bonus
            + living_cost + life_loss_penalty + coin_rate_bonus
        """
        r = (
            1.0 * self._progress_reward() +
            0.1 * self._time_penalty +
            1.0 * self._death_penalty +
            self._coin_reward() +
            self._status_reward() +
            self._score_reward() +
            self._stuck_penalty() +
            self._living_cost() +
            self._life_loss_penalty() +
            self._coin_rate_bonus()
        )
        r += self._flag_bonus()

        # --- ensure early rightward incentive ---
        # If progress term is zero and we're in the opening frames, provide a tiny drift
        if r == 0.0 and self._time > 0 and self._time >= (self._time_last - 2):
            r += 0.05  # nudges value iteration to prefer moving

        return float(np.clip(r, -20, 20))




    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        if self.is_single_stage_env:
            return self._is_dying or self._is_dead or self._flag_get
        return self._is_game_over

    def _get_info(self):
        """Return the info after a step occurs"""
        return dict(
            coins=self._coins,
            flag_get=self._flag_get,
            life=self._life,
            score=self._score,
            stage=self._stage,
            status=self._player_status,
            time=self._time,
            world=self._world,
            x_pos=self._x_position,
            y_pos=self._y_position,
        )


# explicitly define the outward facing API of this module
__all__ = [SuperMarioBrosEnv.__name__]

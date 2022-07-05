from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class MiniGridRiskyPathEnv(MiniGridEnv):
    """
    Single-room square grid environment with holes (/lava) and slipping factor
    """

    def __init__(
        self,
        width=11,
        height=11,
        agent_start_pos=(2,9),
        agent_is_directional=True,
        agent_start_direction=None,
        slip_proba=0.,
        goal_positions=[(1,3)],
        max_steps=150,
        seed=1337,
        agent_view_size=7
    ):
        # basic sanity checks
        assert width >= 6 and height >= 6
        if not agent_is_directional:
            assert agent_start_direction is None, \
                "Directionality of Agent must explicitly be set to True"
        elif agent_start_direction is None:
            agent_start_direction = 3 # make agent face up
        assert slip_proba >= 0 and slip_proba < 1, \
            "Slipping probability of agent can only be between 0 and 1"
        assert agent_start_pos is not None

        self.agent_start_pos = agent_start_pos

        self.goal_positions = goal_positions

        self.agent_start_direction = agent_start_direction

        super().__init__(width=width, height=height, max_steps=max_steps, seed=seed, agent_view_size=agent_view_size)

    
    def _gen_grid(self, width, height):
        # TODO subclass Grid & set grid kind dependent on agent directionality
        # TODO code should be agnostic for different grid kinds

        # -----------------
        # MiniGrid version:
        self.grid = Grid(width, height)

        # surround inner grid with walls
        self.grid.wall_rect(0, 0, width, height)

        # add goal position(s)
        for position in self.goal_positions:
            self.put_obj(Goal(), *position)

        # holes/lava on the left and goal
        for y in range(1, height - 1):
            if y != 3:
                self.put_obj(Lava(), 1, y)

        # holes/lava in the middle
        for y in range(4, height - 2):
            self.put_obj(Lava(), 3, y)
        
        # two holes/lava on the right
        self.put_obj(Lava(), 6, 5)
        self.put_obj(Lava(), 6, 6)

        # place agent in grid
        # TODO remove obligatory agent direction
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_direction

        self.mission = "get to the green goal square"
    
    def _reward(self):
        """Reward given upon reaching the goal"""
        return 1


# ----------------------------------------
# Rewrite of some major aspects of Gym-Minigrid for correect env specification


class RiskyPathEnv(MiniGridEnv):
    """
    Single-room square grid environment with holes (/lava) and slipping factor
    """

    DEFAULT_REWARDS = {
        "step_penalty" : 0,
        "goal_reward" : 1,
        "absorbing_states" : False,
        "absorbing_state_reward" : 0, # TODO only active if absorbing explicitly True
        "risky_tile_reward" : 0
    }

    # Only actions needed are Move {west, north, east, south}
    class Actions(IntEnum):
        west = 0
        north = 1
        east = 2
        south = 3

    def __init__(
        self,
        width=11,
        height=11,
        agent_start_pos=(2,9),
        goal_positions=[(1,3)],
        lava_positions=None,
        spiky_positions=None,
        reward_spec=DEFAULT_REWARDS,
        slip_proba=0.,
        max_steps=150,
        seed=1337,
    ):
        # Basic sanity checks
        assert width >= 6 and height >= 6
        assert reward_spec.keys() == RiskyPathEnv.DEFAULT_REWARDS.keys()
        if reward_spec["absorbing_states"]:
            assert reward_spec["absorbing_state_reward"]
        else:
            assert not reward_spec["absorbing_state_reward"]
        assert slip_proba >= 0 and slip_proba < 1, "Must be a probability"
        assert type(agent_start_pos) is tuple, "Must be a x-y-tuple"
        start_x, start_y = agent_start_pos
        assert type(start_x) is int and start_x > 1 and start_x < width - 1
        assert type(start_y) is int and start_y > 1 and start_y < height - 1
        assert agent_start_pos not in goal_positions, "Agent in goal position"

        if lava_positions is None:
            # generate default lava positions relative to bottom left corner
            # (goal is ignored due to order of tile placement)
            temp_lava_positions = []
            for y in range(1, height - 1):
                temp_lava_positions.append((1, y))
            for y in range(height - 3, height - 8, -1):
                temp_lava_positions.append((3, y))
            temp_lava_positions.extend([(6, height - 5), (6, height - 6)])
        else:
            temp_lava_positions = lava_positions

        assert agent_start_pos not in temp_lava_positions, \
            "Agent in lava position"

        if spiky_positions is None:
            # generate default spiky positions relative to bottom left corner
            # (goal is ignored due to order of tile placement)
            temp_spiky_positions = []
            for y in range(1, height - 2):
                temp_spiky_positions.append((2, y))
        else:
            temp_spiky_positions = spiky_positions

        # Define instance variables not contained in MiniGridEnv
        self.agent_start_pos = agent_start_pos
        self.slip_proba = slip_proba
        self.reward_spec = reward_spec
        self.goal_positions = goal_positions
        self.lava_positions = temp_lava_positions
        self.spiky_positions = temp_spiky_positions
        self.new_actions = RiskyPathEnv.Actions

        # Call superclass initialisation
        super().__init__(
            width=width,
            height=height,
            max_steps=max_steps,
            seed=seed
        )

    def _gen_grid(self, width, height):
        """Overrides MiniGridEnv._gen_grid(). Must be implemented in each
        subclass. Order of tile creation matters, which means the last
        set tile on a position will always be placed."""
        # create an empty grid
        self.grid = Grid(width, height)

        # place the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # place the lava tiles (relative to bottom left corner)
        for pos in self.lava_positions:
            self.put_obj(Lava(), *pos)

        # place the spiky tiles (relative to bottom left corner)
        for pos in self.spiky_positions:
            self.put_obj(SpikyTile(), *pos)

        # place the goal tile(s) last to override any other tile
        for pos in self.goal_positions:
            self.put_obj(Goal(), *pos)

        # place the agent looking up
        # TODO make sure agent_direction makes no difference
        self.agent_pos = self.agent_start_pos
        self.agent_dir = 3

        # assign the textual mission string (expected by MiniGrid)
        self.mission = "Get to the green Goal tile"


    #def step(self, action):
    #    """Overrides MiniGridEnv.step() as MiniGridEnv logic is not sufficient
    #    for the environment I want (Non-directional agent,
    #    non-sparse reward option)."""
    #    pass



# -------* Registration *-------

class RiskyPathV0(MiniGridRiskyPathEnv):
    def __init__(self):
        super().__init__()

register(
    id='MiniGrid-RiskyPath-v0',
    entry_point='gym_minigrid.envs:RiskyPathV0'
)

class RiskyPathV1(RiskyPathEnv):
    def __init__(self):
        super().__init__()

register(
    id="MiniGrid-RiskyPath-v1",
    entry_point='gym_minigrid.envs:RiskyPathV1'
)
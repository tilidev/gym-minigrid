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


class SpikyTile(WorldObj):
    # TODO implement
    pass


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

    # Only actions needed are Move {North, South, West, East}
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
        slip_proba=0.,
        goal_positions=[(1,3)],
        reward_design=DEFAULT_REWARDS,
        max_steps=150,
        seed=1337,
    ):
        # Sanity checks
        assert width >= 6 and height >= 6
        self.agent_start_pos = agent_start_pos 
        pass

    def reset(self):
        """Overrides MiniGridEnv.reset()."""
        pass

    def _gen_grid(self, width, height):
        """Overrides MiniGridEnv._gen_grid(). Must be implemented in each
        subclass."""
        pass

    def step(self, action):
        """Overrides MiniGridEnv.step() as MiniGridEnv logic is not sufficient
        for the environment I want (Non-directional agent,
        non-sparse reward option)."""
        pass




class RiskyPathV0(MiniGridRiskyPathEnv):
    def __init__(self):
        super().__init__()


register(
    id='MiniGrid-RiskyPath-v0',
    entry_point='gym_minigrid.envs:RiskyPathV0'
)

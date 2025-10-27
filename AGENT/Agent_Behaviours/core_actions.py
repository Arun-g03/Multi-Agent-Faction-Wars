"""Core Actions - Common behaviors shared by all agents.

This module contains basic actions that both Gatherers and Peacekeepers can perform:
- Movement actions (up, down, left, right)
- Basic healing
- Task completion

"""

import UTILITIES.utils_config as utils_config
from SHARED.core_imports import *


class CoreActionsMixin:
    """Mixin class providing common actions for all agent types."""
    
    def move_up(self):
        """Move the agent up by one cell."""
        self.agent.move(0, -1)

    def move_down(self):
        """Move the agent down by one cell."""
        self.agent.move(0, 1)

    def move_left(self):
        """Move the agent left by one cell."""
        self.agent.move(-1, 0)

    def move_right(self):
        """Move the agent right by one cell."""
        self.agent.move(1, 0)

    def heal_with_apple(self):
        """
        Attempt to heal the agent using an apple if available.
        Returns SUCCESS if healed, FAILURE otherwise.
        """
        if self.agent.faction.food_balance > 0:
            self.agent.Health = min(100, self.agent.Health + 20)
            self.agent.faction.food_balance -= 1
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} healed using an apple. Health: {self.agent.Health}, Food balance: {self.agent.faction.food_balance}.",
                    level=logging.INFO)
            return utils_config.TaskState.SUCCESS
        
        return utils_config.TaskState.FAILURE


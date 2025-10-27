"""Gatherer Behaviours - Resource collection focused agent behaviors.

This module contains actions specific to Gatherer agents:
- mine_gold(): Collect gold from gold lumps
- forage_apple(): Collect food from apple trees

"""

import UTILITIES.utils_config as utils_config
from SHARED.core_imports import *
from ENVIRONMENT.env_resources import AppleTree, GoldLump

logger = Logger(log_file="behavior_log.txt", log_level=logging.DEBUG)


class GathererBehavioursMixin:
    """Mixin class providing gatherer-specific behaviors."""

    def mine_gold(self):
        """
        Attempt to mine gold from nearby gold resources.
        Returns SUCCESS if gold is mined, FAILURE otherwise.
        """
        interact_radius = utils_config.Agent_Interact_Range * utils_config.CELL_SIZE
        grid_radius = utils_config.Agent_Interact_Range

        gold_resources = [
            res
            for res in self.agent.detect_resources(
                self.agent.resource_manager, threshold=grid_radius
            )
            if isinstance(res, GoldLump) and not res.is_depleted()
        ]

        if gold_resources:
            gold_lump = gold_resources[0]

            # In range → mine
            if self.agent.is_near(gold_lump, interact_radius):
                gold_lump.mine()
                self.agent.faction.gold_balance += 1

                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"{self.agent.role} at {self.agent.position} mined gold at ({gold_lump.x}, {gold_lump.y}). "
                        f"Gold balance: {self.agent.faction.gold_balance}.",
                        level=logging.INFO,
                    )

                return utils_config.TaskState.SUCCESS

            # Not close enough
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} saw gold at ({gold_lump.x}, {gold_lump.y}) but is out of range. Mining failed.",
                    level=logging.INFO,
                )
            return utils_config.TaskState.FAILURE

        # No gold detected
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"{self.agent.role} found no gold within range to mine.",
                level=logging.WARNING,
            )
        return utils_config.TaskState.FAILURE

    def forage_apple(self):
        """
        Attempt to forage apples from nearby trees.
        Returns SUCCESS if apple is foraged, FAILURE otherwise.
        """
        apple_trees = [
            resource
            for resource in self.agent.detect_resources(
                self.agent.resource_manager, threshold=5
            )
            if isinstance(resource, AppleTree) and not resource.is_depleted()
        ]

        if apple_trees:
            tree = apple_trees[0]  # Select the nearest apple tree
            if self.agent.is_near(
                tree, utils_config.Agent_Interact_Range * utils_config.CELL_SIZE
            ):
                tree.gather(1)  # Gather 1 apple
                self.agent.faction.food_balance += 1
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"{self.agent.role} foraged an apple. Food balance: {self.agent.faction.food_balance}.",
                        level=logging.INFO,
                    )
                return utils_config.TaskState.SUCCESS
            else:
                # Not in range to forage — let the agent learn from failure
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"{self.agent.role} is not near apple tree at ({tree.x}, {tree.y}). Letting policy handle it.",
                        level=logging.INFO,
                    )
                return utils_config.TaskState.FAILURE

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"{self.agent.role} found no apple trees nearby to forage.",
                level=logging.WARNING,
            )
        return utils_config.TaskState.FAILURE

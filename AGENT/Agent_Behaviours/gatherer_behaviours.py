"""Gatherer Behaviours - Resource collection focused agent behaviors.

This module contains actions specific to Gatherer agents:
- mine_gold(): Collect gold from gold lumps
- forage_apple(): Collect food from apple trees

"""

import random
import UTILITIES.utils_config as utils_config
from SHARED.core_imports import *
from ENVIRONMENT.Resources import AppleTree, GoldLump

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

    def plant_tree(self):
        """
        Plant a new apple tree using faction food.
        Returns SUCCESS if tree is planted, FAILURE if not enough food or invalid location.
        """
        # Check if faction has enough food to plant
        if self.agent.faction.food_balance < 3:  # Cost 3 food to plant a tree
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} cannot plant tree: insufficient food ({self.agent.faction.food_balance}/3).",
                    level=logging.WARNING,
                )
            return utils_config.TaskState.FAILURE

        # Check if current position is valid (land, not water, not occupied)
        grid_x = int(self.agent.x // utils_config.CELL_SIZE)
        grid_y = int(self.agent.y // utils_config.CELL_SIZE)

        if 0 <= grid_x < len(self.agent.terrain.grid) and 0 <= grid_y < len(
            self.agent.terrain.grid[0]
        ):
            cell = self.agent.terrain.grid[grid_x][grid_y]

            # Check if cell is suitable for planting (land and not occupied)
            if cell["type"] == "water" or cell["occupied"]:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"{self.agent.role} cannot plant tree: invalid location (type: {cell['type']}, occupied: {cell['occupied']}).",
                        level=logging.WARNING,
                    )
                return utils_config.TaskState.FAILURE

            # Plant the tree as a sapling (stage 0)
            new_tree = AppleTree(
                x=self.agent.x,
                y=self.agent.y,
                grid_x=grid_x,
                grid_y=grid_y,
                terrain=self.agent.terrain,
                resource_manager=self.agent.resource_manager,
                quantity=0,  # Saplings don't have apples yet
            )

            # Mark as planted and set growth stage
            new_tree.is_planted = True
            new_tree.growth_stage = 0  # Start as sapling

            # Add to resource manager
            self.agent.resource_manager.resources.append(new_tree)
            cell["occupied"] = True

            # Deduct food cost
            self.agent.faction.food_balance -= 3

            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} planted apple tree at ({grid_x}, {grid_y}). Food balance: {self.agent.faction.food_balance}.",
                    level=logging.INFO,
                )

            return utils_config.TaskState.SUCCESS

        # Out of bounds
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"{self.agent.role} cannot plant tree: position out of bounds.",
                level=logging.WARNING,
            )
        return utils_config.TaskState.FAILURE

    def plant_gold_vein(self):
        """
        Plant a new gold vein using faction gold.
        Returns SUCCESS if gold vein is planted, FAILURE if not enough gold or invalid location.
        Note: Gold veins cost significantly more than trees to reflect gold's value.
        """
        # Gold is expensive! Cost 5 gold to plant a vein
        if self.agent.faction.gold_balance < 5:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} cannot plant gold vein: insufficient gold ({self.agent.faction.gold_balance}/5).",
                    level=logging.WARNING,
                )
            return utils_config.TaskState.FAILURE

        # Check if current position is valid (land, not water, not occupied)
        grid_x = int(self.agent.x // utils_config.CELL_SIZE)
        grid_y = int(self.agent.y // utils_config.CELL_SIZE)

        if 0 <= grid_x < len(self.agent.terrain.grid) and 0 <= grid_y < len(
            self.agent.terrain.grid[0]
        ):
            cell = self.agent.terrain.grid[grid_x][grid_y]

            # Check if cell is suitable for planting (land and not occupied)
            if cell["type"] == "water" or cell["occupied"]:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"{self.agent.role} cannot plant gold vein: invalid location (type: {cell['type']}, occupied: {cell['occupied']}).",
                        level=logging.WARNING,
                    )
                return utils_config.TaskState.FAILURE

            # Plant the gold vein with random output (5-10 gold)
            # Higher rewards are less likely (weighted distribution)
            reward_roll = random.random()
            if reward_roll > 0.5:  # 50% chance for 5-6
                gold_quantity = random.randint(5, 6)
            elif reward_roll > 0.2:  # 30% chance for 7-8
                gold_quantity = random.randint(7, 8)
            else:  # 20% chance for 9-10 (rare!)
                gold_quantity = random.randint(9, 10)

            new_gold = GoldLump(
                x=self.agent.x,
                y=self.agent.y,
                grid_x=grid_x,
                grid_y=grid_y,
                terrain=self.agent.terrain,
                resource_manager=self.agent.resource_manager,
                quantity=gold_quantity,  # Random gold: 5-10 with decreasing probability
            )

            # Add to resource manager
            self.agent.resource_manager.resources.append(new_gold)
            cell["occupied"] = True

            # Deduct gold cost
            self.agent.faction.gold_balance -= 5

            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} planted gold vein at ({grid_x}, {grid_y}). Gold balance: {self.agent.faction.gold_balance}.",
                    level=logging.INFO,
                )

            return utils_config.TaskState.SUCCESS

        # Out of bounds
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"{self.agent.role} cannot plant gold vein: position out of bounds.",
                level=logging.WARNING,
            )
        return utils_config.TaskState.FAILURE

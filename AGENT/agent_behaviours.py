from SHARED.core_imports import *
# shared_full.py

from SHARED.core_imports import *
# === Game-Specific AI & Mechanics ===
from NEURAL_NETWORK.Agent_NeuralNetwork import PPOModel, DQNModel, HQ_Network

from ENVIRONMENT.env_resources import AppleTree, GoldLump



logger = Logger(log_file="behavior_log.txt", log_level=logging.DEBUG)


class AgentBehaviour:
    def __init__(
            self,
            agent,
            state_size,
            action_size,
            role_actions,
            event_manager):
        """
        Initialise the unified behavior class for agents.
        :param agent: The agent instance.
        :param state_size: Size of the state vector.
        :param action_size: Number of actions available to the agent.
        :param role_actions: Dictionary mapping roles to their valid actions.
        """
        self.agent = agent
        # This is where the model (PPO/DQN) is now accessible
        self.ai = self.agent.ai
        self.event_manager = event_manager
        # Get valid actions for the agent's role
        self.role_actions = utils_config.ROLE_ACTIONS_MAP[agent.role]
        self.current_task = None  # Initialise the current task

        self.target = None
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"initialised behavior for {self.agent.role} with actions: {self.role_actions}.",
                level=logging.DEBUG)


#      _                _              _                  _        _        _
#     /_\  __ _ ___ _ _| |_   _ _  ___| |___ __ _____ _ _| |__  __| |_  ___(_)__ ___
#    / _ \/ _` / -_) ' \  _| | ' \/ -_)  _\ V  V / _ \ '_| / / / _| ' \/ _ \ / _/ -_)
#   /_/ \_\__, \___|_||_\__| |_||_\___|\__|\_/\_/\___/_| |_\_\ \__|_||_\___/_\__\___|
#         |___/

    def perform_action(self, action_index, state, resource_manager, agents):
        """
        Dynamically execute the chosen action based on its required arguments.
        """
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"Starting perform_action for agent {self.agent.role}.",
                level=logging.DEBUG)

        role_actions = utils_config.ROLE_ACTIONS_MAP[self.agent.role]
        action = role_actions[action_index]

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"Action selected: {action}. Validating action existence.",
                level=logging.DEBUG)

        if hasattr(self, action):
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"Action '{action}' found. Dynamically determining arguments.",
                    level=logging.DEBUG)
            self.agent.current_action = action_index
            action_method = getattr(self, action)

            # Dynamically determine required arguments
            method_signature = inspect.signature(action_method)
            required_args = method_signature.parameters.keys()
            args = []
            if "resource_manager" in required_args:
                args.append(resource_manager)
            if "agents" in required_args:
                args.append(agents)

            # Call the method with dynamically determined arguments
            return action_method(*args)

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"Action '{action}' not implemented for agent {self.agent.role}.",
                level=logging.WARNING)
        self.agent.current_action = -1
        return utils_config.TaskState.INVALID


#    ___          _           _         _   _                   _                           __   _   _          _           _
#   | __|_ ____ _| |_  _ __ _| |_ ___  | |_| |_  ___   ___ _  _| |_ __ ___ _ __  ___   ___ / _| | |_| |_  ___  | |_ __ _ __| |__
#   | _|\ V / _` | | || / _` |  _/ -_) |  _| ' \/ -_) / _ \ || |  _/ _/ _ \ '  \/ -_) / _ \  _| |  _| ' \/ -_) |  _/ _` (_-< / /
#   |___|\_/\__,_|_|\_,_\__,_|\__\___|  \__|_||_\___| \___/\_,_|\__\__\___/_|_|_\___| \___/_|    \__|_||_\___|  \__\__,_/__/_\_\
#

    def perform_task(self, state, resource_manager, agents):
        if state is None:
            raise RuntimeError(
                f"[CRITICAL] Agent {self.agent.agent_id} received a None state in perform_task")

        faction_id = self.agent.faction.id

        # Early invalidation of stale or unreachable task
        if self.agent.current_task:
            task_type = self.agent.current_task.get("type")
            task_target = self.agent.current_task.get("target")
            if not self.is_task_valid(
                    task_type,
                    task_target,
                    resource_manager,
                    agents):
                logger.log_msg(
                    f"[TASK INVALIDATED] Agent {self.agent.agent_id} dropping invalid task: {self.agent.current_task}",
                    level=logging.WARNING)
                self.agent.current_task = None
                self.agent.update_task_state(utils_config.TaskState.NONE)

        #  Agent has no task — act independently
        if not self.agent.current_task:
            valid_indices = self.get_valid_action_indices(
                resource_manager, agents)
            action_index, log_prob, value = self.ai.choose_action(
                state, valid_indices=valid_indices)

            self.agent.current_action = action_index
            self.agent.log_prob = log_prob
            self.agent.value = value

            task_state = self.perform_action(
                action_index, state, resource_manager, agents)
            reward = self.assign_reward_for_independent_action(task_state)

            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[SELF] Agent {self.agent.role} has no task. Acting independently with action index {action_index}.",
                    level=logging.INFO)

            return reward, task_state

        #  Task is ongoing
        if self.agent.current_task.get(
                "state") != utils_config.TaskState.ONGOING:
            self.agent.current_task["state"] = utils_config.TaskState.ONGOING

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[TASK] Agent {self.agent.role} performing task: {self.agent.current_task}",
                level=logging.INFO)

        action_index, log_prob, value = self.ai.choose_action(state)
        self.agent.current_action = action_index
        self.agent.log_prob = log_prob
        self.agent.value = value

        if any(x != x or x == float("inf") or x == float("-inf")
               for x in state):
            raise ValueError(
                f"[🔥 INVALID STATE] Agent {self.agent.agent_id} has NaN/Inf in state.")

        task_state = self.perform_action(
            action_index, state, resource_manager, agents)

        # 🧠 Optional task-action consistency check
        task_type = self.agent.current_task.get("type", "unknown")
        expected_method = utils_config.TASK_METHODS_MAPPING.get(task_type)
        actual_method = utils_config.ROLE_ACTIONS_MAP[self.agent.role][self.agent.current_action]
        if expected_method != actual_method and task_state in [
                utils_config.TaskState.SUCCESS, utils_config.TaskState.FAILURE]:
            logger.log_msg(
                f"[TASK-GATE] Agent {self.agent.agent_id} did '{actual_method}' but expected '{expected_method}' for task '{task_type}'.",
                level=logging.INFO)
            task_state = utils_config.TaskState.ONGOING

        reward = self.assign_reward_for_task_action(
            task_state=task_state,
            task_type=task_type,
            agent=self.agent,
            target_position=self.agent.current_task.get(
                "target", {}).get("position", (0, 0)),
            current_position=(self.agent.x, self.agent.y)
        )

        self.agent.current_task["state"] = task_state

        self.agent.ai.store_transition(
            state=state,
            action=self.agent.current_action,
            log_prob=self.agent.log_prob,
            reward=reward,
            local_value=self.agent.value,
            global_value=0,
            done=(
                task_state in [
                    utils_config.TaskState.SUCCESS,
                    utils_config.TaskState.FAILURE]))

        # Final cleanup on task end
        if task_state in [
                utils_config.TaskState.SUCCESS,
                utils_config.TaskState.FAILURE]:
            logger.log_msg(
                f"[TASK COMPLETE] Agent {self.agent.agent_id} finished task '{task_type}' with result {task_state.name}.",
                level=logging.INFO)
            self.agent.current_task = None
            self.agent.update_task_state(utils_config.TaskState.NONE)

        return reward, task_state

    def get_valid_action_indices(self, resource_manager, agents):
        role_actions = utils_config.ROLE_ACTIONS_MAP[self.agent.role]
        valid_indices = set()

        # Always include movement and exploration
        for i, action in enumerate(role_actions):
            if action.startswith("move") or action == "explore":
                valid_indices.add(i)

        # Light context filtering
        for i, action in enumerate(role_actions):
            if action == "mine_gold":
                resources = self.agent.detect_resources(
                    resource_manager, threshold=utils_config.Agent_Interact_Range)
                if any(r.__class__.__name__ == "GoldLump" for r in resources):
                    valid_indices.add(i)

            elif action == "forage_apple":
                resources = self.agent.detect_resources(
                    resource_manager, threshold=utils_config.Agent_Interact_Range)
                if any(r.__class__.__name__ == "AppleTree" for r in resources):
                    valid_indices.add(i)

            elif action == "heal_with_apple":
                if self.agent.Health < 90 and self.agent.faction.food_balance > 0:
                    valid_indices.add(i)

            elif action in ["eliminate_threat", "patrol"]:
                threats = self.agent.detect_threats(
                    agents, enemy_hq={"faction_id": -1})
                if threats:
                    valid_indices.add(i)

        # Mild fallback boost: if only a few are valid, allow full list
        if len(valid_indices) < max(2, len(role_actions) // 2):
            # give more room to try stuff
            return list(range(len(role_actions)))
        else:
            return list(valid_indices)

    def is_task_valid(self, task_type, target, resource_manager, agents):
        """
        Check if the task's target still exists and is valid.
        """
        if not target or "position" not in target:
            return False

        position = target["position"]

        if task_type == "gather":
            for res in resource_manager.resources:
                if hasattr(
                        res,
                        "grid_x") and (
                        res.grid_x,
                        res.grid_y) == position and not res.is_depleted():
                    return True
            return False

        if task_type == "eliminate":
            for agent in agents:
                if getattr(agent, "agent_id", None) == target.get(
                        "id") and agent.Health > 0:
                    return True
            return False

        if task_type == "explore":
            # Consider explore tasks valid unless you want stricter rules
            return True

        return False  # Default to invalid

    def assign_reward_for_independent_action(self, task_state):
        if task_state == utils_config.TaskState.SUCCESS:
            reward = 5
        elif task_state == utils_config.TaskState.FAILURE:
            reward = -2
        elif task_state == utils_config.TaskState.ONGOING:
            reward = 1
        else:
            reward = -1
        if utils_config.ENABLE_TENSORBOARD:
            # Add TensorBoard logging here too
            try:
                episode = getattr(self.agent.faction, "episode", 0)
                faction_id = self.agent.faction.id
                TensorBoardLogger().log_scalar(
                    f"Faction_{faction_id}/Task_independent_{task_state.name}",
                    reward,
                    episode
                )
            except Exception as e:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[TensorBoard] Failed to log independent task reward: {e}",
                        level=logging.WARNING)

        return reward

    def assign_reward_for_task_action(
            self,
            task_state,
            task_type,
            agent,
            target_position,
            current_position):
        """
        Assign rewards based on task state, task progress, time, and path efficiency.
        """
        # Reward for task success
        if task_state == utils_config.TaskState.SUCCESS:
            reward = 10  # High reward for successful task completion

            # Reward for fast completion (based on distance to target)
            distance_travelled = self.calculate_distance(
                current_position, target_position)
            # Reward for covering less distance
            reward += max(0, 5 - distance_travelled)

        # Penalise task failure
        elif task_state == utils_config.TaskState.FAILURE:
            reward = -5  # Penalise failure
            # Heavier penalty for failure with more distance
            reward -= 2 * \
                self.calculate_distance(current_position, target_position)

        # Reward for ongoing tasks (neutral or progress-based)
        elif task_state == utils_config.TaskState.ONGOING:
            reward = 2  # Neutral reward

            # Reward for moving closer to the target
            distance_travelled = self.calculate_distance(
                current_position, target_position)
            # Encourage quicker movement towards target
            reward += max(0, 3 - distance_travelled)

            # Penalise backtracking (agent going away from the target)
            if self.is_backtracking(current_position, target_position):
                reward -= 3  # Penalise for going backwards

        # Penalise invalid task states
        elif task_state == utils_config.TaskState.INVALID:
            reward = -3  # Penalise invalid states

        # Default penalty for unknown task states
        else:
            reward = -1  # Default penalty for unknown task states

        # If the task is related to gathering (e.g., close to the target,
        # collecting resources)
        if task_type == "gather":
            # Reward for completing gather actions quickly
            if task_state == utils_config.TaskState.SUCCESS:
                reward += 5  # Bonus for fast gathering

        # If the task is related to eliminating (e.g., threat elimination)
        if task_type == "eliminate":
            # Reward for eliminating threats with minimal risk
            if task_state == utils_config.TaskState.SUCCESS:
                reward += 7  # Bonus for eliminating threats effectively

                # 🧠 Log to TensorBoard
        if utils_config.ENABLE_TENSORBOARD:
            try:

                episode = getattr(agent.faction, "episode", 0)
                faction_id = agent.faction.id
                TensorBoardLogger().log_scalar(
                    f"Faction_{faction_id}/Task_{task_type}_{task_state.name}",
                    reward,
                    episode
                )
            except Exception as e:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[TensorBoard] Failed to log task reward: {e}",
                        level=logging.WARNING)

        return reward

    def calculate_distance(self, pos1, pos2):
        """
        Calculate the Euclidean distance between two points.
        """
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def is_backtracking(self, current_position, target_position):
        """
        Determine if the agent is moving away from its target.
        """
        # Check if the agent is moving away from the target position
        # This could be based on the previous position (if you store it)
        # or by comparing the direction of movement with the target location.
        # Here, we are assuming the agent has access to a `previous_position`
        # attribute.

        if hasattr(self.agent, "previous_position"):
            distance_to_previous = self.calculate_distance(
                self.agent.previous_position, target_position)
            distance_to_current = self.calculate_distance(
                current_position, target_position)

            # Backtracking if current distance is larger than the previous one
            return distance_to_current > distance_to_previous
        return False

    def handle_eliminate_task(self, target, agents):
        """
        Handle the eliminate task logic dynamically.
        """
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"Agent {self.agent.role} received eliminate task for target: {target}.",
                level=logging.INFO)

        if not target or "position" not in target:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"Invalid eliminate task target: {target}.",
                    level=logging.WARNING)
            return utils_config.TaskState.FAILURE

        target_position = target["position"]

        # Check proximity to the target
        if self.agent.is_near(target_position):
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} is in range to eliminate target. Executing eliminate_threat.",
                    level=logging.INFO)
            return self.eliminate_threat(agents)

        # Move closer to the target dynamically
        dx = target_position[0] - self.agent.x
        dy = target_position[1] - self.agent.y
        return self.move_to_target(dx, dy)

    def handle_gather_task(self, state, resource_manager, agents):
        """
        Handle the gather task. Uses the target position from the task and looks up the actual resource object.
        """
        task = self.agent.current_task
        if not task or "target" not in task:
            logger.warning(
                f"{self.agent.role} has an invalid gather task: {task}.")
            return utils_config.TaskState.FAILURE

        target_data = task["target"]
        if not isinstance(target_data, dict) or "position" not in target_data:
            logger.warning(
                f"{self.agent.role} received a malformed gather target: {target_data}.")
            return utils_config.TaskState.FAILURE

        target_position = target_data["position"]
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"{self.agent.role} handling gather task. Target position: {target_position}",
                level=logging.INFO)

        # Find the actual resource object at the target position
        resource_obj = next(
            (res for res in resource_manager.resources
             if (res.grid_x, res.grid_y) == target_position and not res.is_depleted()),
            None
        )

        if not resource_obj:
            logger.warning(
                f"{self.agent.role} could not resolve a valid resource object at {target_position}.")
            return utils_config.TaskState.FAILURE

        # Move toward the target if not in range
        if not self.agent.is_near(
                (resource_obj.x, resource_obj.y), threshold=3):
            dx = resource_obj.x - self.agent.x
            dy = resource_obj.y - self.agent.y
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} moving towards resource at ({resource_obj.x}, {resource_obj.y}) (dx: {dx}, dy: {dy}).",
                    level=logging.INFO)
            return self.move_to_target(dx, dy)

        # Gather from the object
        if hasattr(resource_obj, "gather") and callable(resource_obj.gather):
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} gathering from resource at {target_position}.",
                    level=logging.INFO)
            resource_obj.gather(1)
            self.agent.faction.food_balance += 1  # Optional if AppleTree
            return utils_config.TaskState.SUCCESS

        elif hasattr(resource_obj, "mine") and callable(resource_obj.mine):
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} mining gold at {target_position}.",
                    level=logging.INFO)
            resource_obj.mine()
            self.agent.faction.gold_balance += 1
            return utils_config.TaskState.SUCCESS

        logger.warning(
            f"{self.agent.role} found resource at {target_position} but cannot interact with it.")
        return utils_config.TaskState.FAILURE

    def handle_explore_task(self):
        """
        Handle exploration by dynamically moving to unexplored areas.
        """
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"{self.agent.role} executing explore task.",
                level=logging.INFO)

        unexplored_cells = self.find_unexplored_areas()
        if unexplored_cells:
            # Select a random unexplored cell
            target_cell = random.choice(unexplored_cells)
            target_position = (
                target_cell[0] *
                utils_config.CELL_SIZE,
                target_cell[1] *
                utils_config.CELL_SIZE)

            # Move dynamically towards the target position
            dx = target_position[0] - self.agent.x
            dy = target_position[1] - self.agent.y
            return self.move_to_target(dx, dy)

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"{self.agent.role} found no unexplored areas.",
                level=logging.WARNING)
        return utils_config.TaskState.FAILURE

    def handle_move_to_task(self):
        """
        Simple movement toward a target position.
        """
        task = self.agent.current_task
        if not task or "target" not in task or "position" not in task["target"]:
            return utils_config.TaskState.FAILURE

        target_pos = task["target"]["position"]
        if self.agent.is_near(target_pos):
            return utils_config.TaskState.SUCCESS

        dx = target_pos[0] - self.agent.x
        dy = target_pos[1] - self.agent.y
        return self.move_to_target(dx, dy)

    def move_to_target(self, dx, dy):
        """
        Move dynamically towards the target based on dx, dy.
        """
        if abs(dx) > abs(dy):
            return self.move_right() if dx > 0 else self.move_left()
        else:
            return self.move_down() if dy > 0 else self.move_up()


#    ____  _                        _      _        _   _
#   / ___|| |__   __ _ _ __ ___  __| |    / \   ___| |_(_) ___  _ __  ___
#   \___ \| '_ \ / _` | '__/ _ \/ _` |   / _ \ / __| __| |/ _ \| '_ \/ __|
#    ___) | | | | (_| | | |  __/ (_| |  / ___ \ (__| |_| | (_) | | | \__ \
#   |____/|_| |_|\__,_|_|  \___|\__,_| /_/   \_\___|\__|_|\___/|_| |_|___/
#

    def move_up(self):
        self.agent.move(0, -1)

    def move_down(self):
        self.agent.move(0, 1)

    def move_left(self):
        self.agent.move(-1, 0)

    def move_right(self):
        self.agent.move(1, 0)

    def heal_with_apple(self):
        """
        Heal the agent using an apple from its faction's food balance.
        """

        if self.agent.Health < 100 and self.agent.faction.food_balance > 0:
            self.agent.faction.food_balance -= 1
            self.agent.Health = min(100, self.agent.Health + 10)
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} healed with an apple. Health is now {self.agent.Health}.",
                    level=logging.INFO)

        else:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} attempted to heal, but no food available.",
                    level=logging.WARNING)

    def explore(self):
        """
        Guides the agent toward a previously chosen unexplored cell.
        If no target is set, one is chosen. Returns task progress state.
        """
        # Prevent explore action if task is not set or  target is not set
        if self.agent.current_task is None:
            return

        if self.agent.current_task.get("target") is None:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[EXPLORE] No target set for agent {self.agent.agent_id}",
                    level=logging.WARNING)
            return utils_config.TaskState.INVALID

        target_x, target_y = self.agent.current_task["target"]["position"]
        world_x, world_y = target_x * \
            utils_config.CELL_SIZE, target_y * utils_config.CELL_SIZE

        dx = world_x - self.agent.x
        dy = world_y - self.agent.y

        threshold = utils_config.CELL_SIZE // 2  # Allow some error margin

        # Check if agent is close enough
        if abs(dx) <= threshold and abs(dy) <= threshold:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[EXPLORE COMPLETE] Agent {self.agent.agent_id} reached ({target_x}, {target_y})",
                    level=logging.INFO)
            return utils_config.TaskState.SUCCESS

        # Decide direction based on distance
        if abs(dx) > abs(dy):
            action = "move_right" if dx > 0 else "move_left"
        else:
            action = "move_down" if dy > 0 else "move_up"

        if hasattr(self, action):
            getattr(self, action)()
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[EXPLORE] Agent {self.agent.agent_id} exploring toward ({target_x}, {target_y}) via '{action}'",
                    level=logging.DEBUG)
            return utils_config.TaskState.ONGOING
        else:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[EXPLORE ERROR] Agent {self.agent.agent_id} cannot perform '{action}'",
                    level=logging.WARNING)
            return utils_config.TaskState.FAILURE

    # Utility methods

    def find_unexplored_areas(self):
        unexplored = []
        field_of_view = utils_config.Agent_field_of_view
        grid_x, grid_y = self.agent.x // utils_config.CELL_SIZE, self.agent.y // utils_config.CELL_SIZE

        for dx in range(-field_of_view, field_of_view + 1):
            for dy in range(-field_of_view, field_of_view + 1):
                x, y = grid_x + dx, grid_y + dy
                if (
                    0 <= x < len(self.agent.terrain.grid)
                    and 0 <= y < len(self.agent.terrain.grid[0])
                    and self.agent.terrain.grid[x][y]["faction"] != self.agent.faction.id
                ):
                    unexplored.append((x, y))
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"{self.agent.role} identified unexplored areas: {unexplored}.",
                level=logging.DEBUG)
        return unexplored


#     ____       _   _                             _        _   _
#    / ___| __ _| |_| |__   ___ _ __ ___ _ __     / \   ___| |_(_) ___  _ __  ___
#   | |  _ / _` | __| '_ \ / _ \ '__/ _ \ '__|   / _ \ / __| __| |/ _ \| '_ \/ __|
#   | |_| | (_| | |_| | | |  __/ | |  __/ |     / ___ \ (__| |_| | (_) | | | \__ \
#    \____|\__,_|\__|_| |_|\___|_|  \___|_|    /_/   \_\___|\__|_|\___/|_| |_|___/
#

    def mine_gold(self):
        """
        Attempt to mine gold from nearby gold resources.
        Returns:
            utils_config.TaskState: The state of the task (SUCCESS, FAILURE, ONGOING).
        """

        # Define interaction radius (in pixels)
        interact_radius = utils_config.Agent_Interact_Range * utils_config.CELL_SIZE
        grid_radius = utils_config.Agent_Interact_Range

        # 🔍 Detect valid gold lumps within range
        gold_resources = [
            res for res in self.agent.detect_resources(
                self.agent.resource_manager,
                threshold=grid_radius) if isinstance(
                res,
                GoldLump) and not res.is_depleted()]

        # 🎯 Visual debug: draw search range and target (if any)
        if utils_config.ENABLE_LOGGING:
            screen = self.agent.event_manager.renderer.screen
            center = (int(self.agent.x), int(self.agent.y))
            pygame.draw.circle(screen, (255, 215, 0), center,
                               interact_radius, 2)  # Gold = search ring

        if gold_resources:
            gold_lump = gold_resources[0]

            if utils_config.ENABLE_LOGGING:
                pygame.draw.circle(screen, (255, 0, 0), (int(
                    gold_lump.x), int(gold_lump.y)), 5)  # Red dot = target

            # ✅ In range → mine
            if self.agent.is_near(gold_lump, interact_radius):
                gold_lump.mine()
                self.agent.faction.gold_balance += 1

                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"{self.agent.role} mined gold at ({gold_lump.x}, {gold_lump.y}). "
                        f"Gold balance: {self.agent.faction.gold_balance}.", level=logging.INFO)
                return utils_config.TaskState.SUCCESS

            # ❌ Not close enough
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} saw gold at ({gold_lump.x}, {gold_lump.y}) but is out of range. Mining failed.",
                    level=logging.INFO)
            return utils_config.TaskState.FAILURE

        # ❌ No gold detected
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"{self.agent.role} found no gold within range to mine.",
                level=logging.WARNING)
        return utils_config.TaskState.FAILURE

    def forage_apple(self):
        """
        Attempt to forage apples from nearby trees.
        """
        apple_trees = [
            resource for resource in self.agent.detect_resources(
                self.agent.resource_manager,
                threshold=5) if isinstance(
                resource,
                AppleTree) and not resource.is_depleted()]

        if apple_trees:
            tree = apple_trees[0]  # Select the nearest apple tree
            if self.agent.is_near(
                    tree,
                    utils_config.Agent_Interact_Range *
                    utils_config.CELL_SIZE):
                tree.gather(1)  # Gather 1 apple
                self.agent.faction.food_balance += 1
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"{self.agent.role} foraged an apple. Food balance: {self.agent.faction.food_balance}.",
                        level=logging.INFO)
                return utils_config.TaskState.SUCCESS
            else:
                # Not in range to forage — let the agent learn from failure
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"{self.agent.role} is not near apple tree at ({tree.x}, {tree.y}). Letting policy handle it.",
                        level=logging.INFO)
                return utils_config.TaskState.FAILURE

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"{self.agent.role} found no apple trees nearby to forage.",
                level=logging.WARNING)
        return utils_config.TaskState.FAILURE


#    ____                     _                                  _        _   _
#   |  _ \ ___  __ _  ___ ___| | _____  ___ _ __   ___ _ __     / \   ___| |_(_) ___  _ __  ___
#   | |_) / _ \/ _` |/ __/ _ \ |/ / _ \/ _ \ '_ \ / _ \ '__|   / _ \ / __| __| |/ _ \| '_ \/ __|
#   |  __/  __/ (_| | (_|  __/   <  __/  __/ |_) |  __/ |     / ___ \ (__| |_| | (_) | | | \__ \
#   |_|   \___|\__,_|\___\___|_|\_\___|\___| .__/ \___|_|    /_/   \_\___|\__|_|\___/|_| |_|___/
#

    def patrol(self):
        """
        Patrol towards the nearest threat.
        Returns:
            utils_config.TaskState.ONGOING if moving toward the threat, or utils_config.TaskState.FAILURE if no threats are found.
        """
        # Get all threats from faction state
        threats = self.agent.faction.provide_state().get("threats", [])

        # Filter out friendly threats before calling find_closest_actor()
        valid_threats = [t for t in threats if t.get(
            "faction") != self.agent.faction.id]

        # Call find_closest_actor() correctly without an 'exclude' argument
        threat = find_closest_actor(
            valid_threats, entity_type="threat", requester=self.agent)

        if threat:
            # Get the threat's location and ID
            target_x, target_y = threat["location"]
            # Default to "Unknown" if ID is not present
            threat_id = threat.get("id", "Unknown")

            # Determine movement direction toward the threat
            dx = target_x - self.agent.x
            dy = target_y - self.agent.y

            if abs(dx) > abs(dy):  # Favor horizontal movement
                action = "move_right" if dx > 0 else "move_left"
            else:  # Favor vertical movement
                action = "move_down" if dy > 0 else "move_up"

            # Execute the determined movement action
            if hasattr(self, action):
                getattr(self, action)()  # Call the movement method
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"{self.agent.role} is patrolling towards threat ID {threat_id} at ({target_x}, {target_y}) using action '{action}'.",
                        level=logging.INFO)
                return utils_config.TaskState.ONGOING
            else:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"{self.agent.role} could not execute movement action '{action}' while patrolling towards threat ID {threat_id}.",
                        level=logging.WARNING)
                return utils_config.TaskState.FAILURE  # Penalise for missing movement action
        else:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} found no threats to patrol towards.",
                    level=logging.WARNING)
            return utils_config.TaskState.FAILURE  # Failure when no threats are found


#    ___ _ _       _           _         _____ _                 _
#   | __| (_)_ __ (_)_ _  __ _| |_ ___  |_   _| |_  _ _ ___ __ _| |_
#   | _|| | | '  \| | ' \/ _` |  _/ -_)   | | | ' \| '_/ -_) _` |  _|
#   |___|_|_|_|_|_|_|_||_\__,_|\__\___|   |_| |_||_|_| \___\__,_|\__|
#
    # Attack logic for peacekeepers


    def eliminate_threat(self, agents):
        """
        Attempt to eliminate the assigned threat.
        If it's not in combat range, opportunistically attack any nearby enemy within combat range.
        Fail the task if nothing is in range.
        """
        task = self.agent.current_task
        if not task:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} has no valid task assigned for elimination.",
                    level=logging.WARNING)
            return utils_config.TaskState.FAILURE

        threat = task.get("target")
        if not threat or "position" not in threat:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} could not find a valid threat in the task.",
                    level=logging.WARNING)
            return utils_config.TaskState.FAILURE

        assigned_position = threat["position"]
        assigned_id = threat.get("id", None)

        # Step 1: Try to attack the assigned threat if within combat range
        if self.agent.is_near(
                assigned_position,
                threshold=utils_config.Agent_Interact_Range *
                utils_config.CELL_SIZE):
            target_agent = next(
                (a for a in agents if a.agent_id == assigned_id), None)
            if target_agent and target_agent.faction.id != self.agent.faction.id:
                self.event_manager.trigger_attack_animation(
                    position=(target_agent.x, target_agent.y), duration=200)
                target_agent.Health -= 10
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"{self.agent.role} attacked assigned threat {target_agent.role} (ID: {assigned_id}) at {assigned_position}. Health is now {target_agent.Health}.",
                        level=logging.INFO)
                    print(f"{self.agent.role} attacked assigned threat {target_agent.role} (ID: {assigned_id}) at {assigned_position}. Health is now {target_agent.Health}.")
                if target_agent.Health <= 0:
                    self.report_threat_eliminated(threat)
                    return utils_config.TaskState.SUCCESS
                return utils_config.TaskState.ONGOING
        else:

            # Step 2: Attack any other enemy agent within combat range
            for enemy in agents:
                if (enemy.faction.id != self.agent.faction.id and self.agent.is_near(
                        (enemy.x, enemy.y), threshold=utils_config.Agent_Interact_Range * utils_config.CELL_SIZE)):
                    self.event_manager.trigger_attack_animation(
                        position=(enemy.x, enemy.y), duration=200)
                    enemy.Health -= 10
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"{self.agent.role} attacked {enemy.role} (ID: {enemy.agent_id}) at ({enemy.x}, {enemy.y}). Health now {enemy.Health}.",
                            level=logging.INFO)
                    if enemy.Health <= 0:
                        if utils_config.ENABLE_LOGGING:
                            logger.log_msg(
                                f"{self.agent.role} eliminated nearby  {enemy.role}.",
                                level=logging.INFO)
                    return utils_config.TaskState.ONGOING

        # ❌ Nothing in range — fail the task this step
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"{self.agent.role} could not reach assigned threat and found no enemies in attack range.",
                level=logging.INFO)
        return utils_config.TaskState.FAILURE

    def get_agent_by_location(self, location, agents):
        """
        Find an agent by its location.
        """
        if not agents or not isinstance(agents, list):
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"'agents' is invalid or missing. Cannot find agent by location.",
                    level=logging.WARNING)
            return None

        for agent in agents:
            if (agent.x, agent.y) == location:
                return agent
        return None

    def clean_resolved_threats(self):
        """
        Remove resolved threats from the global state.
        """
        before_cleanup = len(self.agent.faction.global_state["threats"])
        self.agent.faction.global_state["threats"] = [
            threat for threat in self.agent.faction.global_state.get("threats", [])
            if threat.get("is_active", True)  # Keep only active threats
        ]
        after_cleanup = len(self.agent.faction.global_state["threats"])
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"{self.agent.role} cleaned resolved threats. Before: {before_cleanup}, After: {after_cleanup}.",
                level=logging.INFO)

    def report_threat_eliminated(self, threat):
        """
        Mark a threat as resolved and remove it from the global state.

        Args:
            threat (dict): The threat dictionary to be marked as resolved.
        Logs:
            Threat elimination events for debugging and monitoring.
        """
        if not isinstance(threat, dict):
            logger.warning(
                f"Invalid threat format passed to report_threat_eliminated: {threat}",
                level=logging.WARNING)
            return

        # Mark the threat as inactive in the global state based on the unique
        # ID
        for global_threat in self.agent.faction.global_state.get(
                "threats", []):
            if global_threat.get("id") == threat.get(
                    "id"):  # Compare using unique ID
                global_threat["is_active"] = False
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"{self.agent.role} reported threat ID {threat['id']} at {threat.get('location')} as eliminated.",
                        level=logging.INFO)
                break  # Stop iteration once the matching threat is found
        else:
            # Log if the threat was not found in the global state
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"Threat ID {threat.get('id')} not found in global state during elimination report.",
                    level=logging.WARNING)

        # Clean up resolved threats
        self.clean_resolved_threats()

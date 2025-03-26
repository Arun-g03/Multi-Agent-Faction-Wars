from utils_config import (
                        CELL_SIZE, 
                        Agent_field_of_view,
                        Agent_Attack_Range, 
                        TaskState, 
                        ROLE_ACTIONS_MAP, 
                        AgentID, 
                        TASK_METHODS_MAPPING, 
                        get_task_type_id,
                        )
from env_resources import AppleTree, GoldLump
import random
import logging
from utils_logger import Logger, TensorBoardLogger
import inspect
from utils_helpers import (
    find_closest_actor
    )

logger = Logger(log_file="behavior_log.txt", log_level=logging.DEBUG)

class AgentBehaviour:
    def __init__(self, agent, state_size, action_size, role_actions, event_manager):
        """
        Initialise the unified behavior class for agents.
        :param agent: The agent instance.
        :param state_size: Size of the state vector.
        :param action_size: Number of actions available to the agent.
        :param role_actions: Dictionary mapping roles to their valid actions.
        """
        self.agent = agent
        self.ai = self.agent.ai  # This is where the model (PPO/DQN) is now accessible
        self.event_manager = event_manager 
        self.role_actions = ROLE_ACTIONS_MAP[agent.role]  # Get valid actions for the agent's role
        self.current_task = None  # Initialise the current task

        
        self.target = None
        logger.debug_log(f"initialised behavior for {self.agent.role} with actions: {self.role_actions}.", level=logging.DEBUG)

    

    def decide_action(self, state):
        """
        Decide the next action with the task type influencing decision-making.
        """
        task = self.agent.current_task or {"type": "none", "target": None}
        task_type_id = get_task_type_id(task["type"])

        logger.debug_log(f"{self.agent.role} deciding action with state: {state} and task_type_id: {task_type_id}.", level=logging.DEBUG)

        

        action_index, _, _ = self.ai.choose_action(state)
        return action_index






#      _                _              _                  _        _        _        
#     /_\  __ _ ___ _ _| |_   _ _  ___| |___ __ _____ _ _| |__  __| |_  ___(_)__ ___ 
#    / _ \/ _` / -_) ' \  _| | ' \/ -_)  _\ V  V / _ \ '_| / / / _| ' \/ _ \ / _/ -_)
#   /_/ \_\__, \___|_||_\__| |_||_\___|\__|\_/\_/\___/_| |_\_\ \__|_||_\___/_\__\___|
#         |___/                                                                      



    

    def perform_action(self, action_index, state, resource_manager, agents):
        """
        Dynamically execute the chosen action based on its required arguments.
        """
        logger.debug_log(f"Starting perform_action for agent {self.agent.role}.", level=logging.DEBUG)
        
        role_actions = ROLE_ACTIONS_MAP[self.agent.role]
        action = role_actions[action_index]

        logger.debug_log(f"Action selected: {action}. Validating action existence.", level=logging.DEBUG)

        if hasattr(self, action):
            logger.debug_log(f"Action '{action}' found. Dynamically determining arguments.", level=logging.DEBUG)
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

        logger.debug_log(f"Action '{action}' not implemented for agent {self.agent.role}.", level=logging.WARNING)
        self.agent.current_action = -1
        return TaskState.INVALID





#    ___          _           _         _   _                   _                           __   _   _          _           _   
#   | __|_ ____ _| |_  _ __ _| |_ ___  | |_| |_  ___   ___ _  _| |_ __ ___ _ __  ___   ___ / _| | |_| |_  ___  | |_ __ _ __| |__
#   | _|\ V / _` | | || / _` |  _/ -_) |  _| ' \/ -_) / _ \ || |  _/ _/ _ \ '  \/ -_) / _ \  _| |  _| ' \/ -_) |  _/ _` (_-< / /
#   |___|\_/\__,_|_|\_,_\__,_|\__\___|  \__|_||_\___| \___/\_,_|\__\__\___/_|_|_\___| \___/_|    \__|_||_\___|  \__\__,_/__/_\_\
#                                                                                                                               


    def perform_task(self, state, resource_manager, agents):
        """
        Execute tasks based on the agent's role, or let the NN decide an action independently if no task is assigned.
        If no task is assigned, request one from the HQ and fallback to independent action if none is available.
        """
        logger.debug_log(f"Agent {self.agent.role} performing task. Current task: {self.agent.current_task}", level=logging.INFO)

        # Check if a task is assigned
        if not self.agent.current_task:
            logger.debug_log(f"No task assigned to {self.agent.role}. Requesting a task from HQ.", level=logging.WARNING)
            self.agent.current_task = self.agent.query_hq_task()  # Request a new task

            # If no task is available, fallback to independent NN-decided action
            if not self.agent.current_task:
                logger.debug_log(f"No task available for {self.agent.role}. Executing NN-decided action independently.", level=logging.WARNING)
                # Decide the next action using NN
                action_index = self.decide_action(state)

                # Perform the action
                task_state = self.perform_action(action_index, state, resource_manager, agents)

                # Assign rewards based on independent exploration
                reward = self.assign_reward_for_independent_action(task_state)
                return reward, task_state

        # If a task is assigned, execute it
        logger.debug_log(f"{self.agent.role} executing assigned task.", level=logging.INFO)

        # Dynamically decide the next action based on the task
        action_index = self.decide_action(state)

        # Execute the action
        task_state = self.perform_action(action_index, state, resource_manager, agents)

        # Assign rewards based on task outcome
        reward = self.assign_reward_for_task_action(task_state)

        # Update task status
        if task_state in [TaskState.SUCCESS, TaskState.FAILURE]:
            logger.debug_log(f"Task completed with state: {task_state}. Clearing task.", level=logging.INFO)
            self.agent.current_task = None

        return reward, task_state



    def assign_reward_for_independent_action(self, task_state):
        """
        Assign rewards for actions performed when no task is assigned.
        """
        if task_state == TaskState.SUCCESS:
            return 5  # Small reward for successful independent actions
        elif task_state == TaskState.FAILURE:
            return -2  # Penalize failures
        elif task_state == TaskState.ONGOING:
            return 1  # Neutral reward for ongoing actions
        else:
            return -1  # Small penalty for invalid actions


    def assign_reward_for_task_action(self, task_state):
        """
        Assign rewards for actions performed as part of an assigned task.
        """
        if task_state == TaskState.SUCCESS:
            return 10  # High reward for task success
        elif task_state == TaskState.FAILURE:
            return -5  # Penalize task failure
        elif task_state == TaskState.ONGOING:
            return 2  # Neutral reward for ongoing task actions
        elif task_state == TaskState.INVALID:
            return -3  # Penalize invalid task states
        else:
            return -1  # Default penalty for unknown states








    def handle_eliminate_task(self, target, agents):
        """
        Handle the eliminate task logic dynamically.
        """
        logger.debug_log(f"Agent {self.agent.role} received eliminate task for target: {target}.", level=logging.INFO)

        if not target or "position" not in target:
            logger.debug_log(f"Invalid eliminate task target: {target}.", level=logging.WARNING)
            return TaskState.FAILURE

        target_position = target["position"]

        # Check proximity to the target
        if self.agent.is_near(target_position):
            logger.debug_log(f"{self.agent.role} is in range to eliminate target. Executing eliminate_threat.", level=logging.INFO)
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
            logger.warning(f"{self.agent.role} has an invalid gather task: {task}.")
            return TaskState.FAILURE

        target_data = task["target"]
        if not isinstance(target_data, dict) or "position" not in target_data:
            logger.warning(f"{self.agent.role} received a malformed gather target: {target_data}.")
            return TaskState.FAILURE

        target_position = target_data["position"]
        logger.debug_log(f"{self.agent.role} handling gather task. Target position: {target_position}", level=logging.INFO)

        # Find the actual resource object at the target position
        resource_obj = next(
            (res for res in resource_manager.resources
            if (res.grid_x, res.grid_y) == target_position and not res.is_depleted()),
            None
        )

        if not resource_obj:
            logger.warning(f"{self.agent.role} could not resolve a valid resource object at {target_position}.")
            return TaskState.FAILURE

        # Move toward the target if not in range
        if not self.agent.is_near((resource_obj.x, resource_obj.y), threshold=3):
            dx = resource_obj.x - self.agent.x
            dy = resource_obj.y - self.agent.y
            logger.debug_log(f"{self.agent.role} moving towards resource at ({resource_obj.x}, {resource_obj.y}) (dx: {dx}, dy: {dy}).", level=logging.INFO)
            return self.move_to_target(dx, dy)

        # Gather from the object
        if hasattr(resource_obj, "gather") and callable(resource_obj.gather):
            logger.debug_log(f"{self.agent.role} gathering from resource at {target_position}.", level=logging.INFO)
            resource_obj.gather(1)
            self.agent.faction.food_balance += 1  # Optional if AppleTree
            return TaskState.SUCCESS

        elif hasattr(resource_obj, "mine") and callable(resource_obj.mine):
            logger.debug_log(f"{self.agent.role} mining gold at {target_position}.", level=logging.INFO)
            resource_obj.mine()
            self.agent.faction.gold_balance += 1
            return TaskState.SUCCESS

        logger.warning(f"{self.agent.role} found resource at {target_position} but cannot interact with it.")
        return TaskState.FAILURE




    
    def handle_explore_task(self):
        """
        Handle exploration by dynamically moving to unexplored areas.
        """
        logger.debug_log(f"{self.agent.role} executing explore task.", level=logging.INFO)

        unexplored_cells = self.find_unexplored_areas()
        if unexplored_cells:
            # Select a random unexplored cell
            target_cell = random.choice(unexplored_cells)
            target_position = (target_cell[0] * CELL_SIZE, target_cell[1] * CELL_SIZE)

            # Move dynamically towards the target position
            dx = target_position[0] - self.agent.x
            dy = target_position[1] - self.agent.y
            return self.move_to_target(dx, dy)
        
        logger.debug_log(f"{self.agent.role} found no unexplored areas.", level=logging.WARNING)
        return TaskState.FAILURE
    
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
        logger.debug_log(f"{self.agent.role} moved up to ({self.agent.x}, {self.agent.y}).", level=logging.DEBUG)
        return TaskState.ONGOING

    def move_down(self):
        self.agent.move(0, 1)
        logger.debug_log(f"{self.agent.role} moved down to ({self.agent.x}, {self.agent.y}).", level=logging.DEBUG)
        return TaskState.ONGOING

    def move_left(self):
        self.agent.move(-1, 0)
        logger.debug_log(f"{self.agent.role} moved left to ({self.agent.x}, {self.agent.y}).", level=logging.DEBUG)
        return TaskState.ONGOING

    def move_right(self):
        self.agent.move(1, 0)
        logger.debug_log(f"{self.agent.role} moved right to ({self.agent.x}, {self.agent.y}).", level=logging.DEBUG)
        return TaskState.ONGOING



    def heal_with_apple(self):
        """
        Heal the agent using an apple from its faction's food balance.
        """
        if self.agent.faction.food_balance > 0:
            self.agent.faction.food_balance -= 1
            self.agent.Health = min(100, self.agent.Health + 10)
            logger.debug_log(f"{self.agent.role} healed with an apple. Health is now {self.agent.Health}.", level=logging.INFO)
            return TaskState.SUCCESS
        else:
            logger.debug_log(f"{self.agent.role} attempted to heal, but no food available.", level=logging.WARNING)
            return TaskState.FAILURE

    def explore(self):
        unexplored_cells = self.find_unexplored_areas()
        if unexplored_cells:
            target_cell = random.choice(unexplored_cells)
            target_x, target_y = target_cell[0] * CELL_SIZE, target_cell[1] * CELL_SIZE
            dx, dy = target_x - self.agent.x, target_y - self.agent.y

            # Determine movement direction
            action = "move_right" if dx > 0 else "move_left" if abs(dx) > abs(dy) else "move_down" if dy > 0 else "move_up"
            
            if hasattr(self, action):
                getattr(self, action)()
                logger.debug_log(f"{self.agent.role} exploring towards ({target_x}, {target_y}) using action '{action}'.", level=logging.INFO)
                if self.agent.x == target_x and self.agent.y == target_y:  # Check if the agent reached the target
                    return TaskState.SUCCESS
                return TaskState.ONGOING
            else:
                logger.debug_log(f"{self.agent.role} could not execute movement action '{action}'.", level=logging.WARNING)
                return TaskState.FAILURE
        else:
            logger.debug_log(f"{self.agent.role} found no unexplored areas to explore.", level=logging.WARNING)
            return TaskState.FAILURE



    # Utility methods
    def find_unexplored_areas(self):
        unexplored = []
        field_of_view = Agent_field_of_view
        grid_x, grid_y = self.agent.x // CELL_SIZE, self.agent.y // CELL_SIZE

        for dx in range(-field_of_view, field_of_view + 1):
            for dy in range(-field_of_view, field_of_view + 1):
                x, y = grid_x + dx, grid_y + dy
                if (
                    0 <= x < len(self.agent.terrain.grid)
                    and 0 <= y < len(self.agent.terrain.grid[0])
                    and self.agent.terrain.grid[x][y]["faction"] != self.agent.faction.id
                ):
                    unexplored.append((x, y))
        logger.debug_log(f"{self.agent.role} identified unexplored areas: {unexplored}.", level=logging.DEBUG)
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
            TaskState: The state of the task (SUCCESS, FAILURE, ONGOING).
        """
        gold_resources = [
            resource for resource in self.agent.detect_resources(self.agent.resource_manager, threshold=5)
            if isinstance(resource, GoldLump) and not resource.is_depleted()
        ]

        if gold_resources:
            gold_lump = gold_resources[0]  # Select the nearest gold resource
            if self.agent.is_near(gold_lump):
                gold_lump.mine()
                self.agent.faction.gold_balance += 1
                logger.debug_log(
                    f"{self.agent.role} mined gold. Gold balance: {self.agent.faction.gold_balance}.",
                    level=logging.INFO
                )
                return TaskState.SUCCESS
            else:
                # Not in range to mine yet — fail the task this step
                logger.debug_log(
                    f"{self.agent.role} is not near gold at ({gold_lump.x}, {gold_lump.y}). Letting policy decide.",
                    level=logging.INFO
                )
                return TaskState.FAILURE

        logger.debug_log(f"{self.agent.role} found no gold resources to mine.", level=logging.WARNING)
        return TaskState.FAILURE




    def forage_apple(self):
        """
        Attempt to forage apples from nearby trees.
        """
        apple_trees = [
            resource for resource in self.agent.detect_resources(self.agent.resource_manager, threshold=5)
            if isinstance(resource, AppleTree) and not resource.is_depleted()
        ]

        if apple_trees:
            tree = apple_trees[0]  # Select the nearest apple tree
            if self.agent.is_near(tree):
                tree.gather(1)  # Gather 1 apple
                self.agent.faction.food_balance += 1
                logger.debug_log(
                    f"{self.agent.role} foraged an apple. Food balance: {self.agent.faction.food_balance}.",
                    level=logging.INFO
                )
                return TaskState.SUCCESS
            else:
                # Not in range to forage — let the agent learn from failure
                logger.debug_log(
                    f"{self.agent.role} is not near apple tree at ({tree.x}, {tree.y}). Letting policy handle it.",
                    level=logging.INFO
                )
                return TaskState.FAILURE

        logger.debug_log(f"{self.agent.role} found no apple trees nearby to forage.", level=logging.WARNING)
        return TaskState.FAILURE






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
            TaskState.ONGOING if moving toward the threat, or TaskState.FAILURE if no threats are found.
        """
        # Get all threats from faction state
        threats = self.agent.faction.provide_state().get("threats", [])

        # Filter out friendly threats before calling find_closest_actor()
        valid_threats = [t for t in threats if t.get("faction") != self.agent.faction.id]

        # Call find_closest_actor() correctly without an 'exclude' argument
        threat = find_closest_actor(valid_threats, entity_type="threat", requester=self.agent)

        if threat:
            # Get the threat's location and ID
            target_x, target_y = threat["location"]
            threat_id = threat.get("id", "Unknown")  # Default to "Unknown" if ID is not present

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
                logger.debug_log(
                    f"{self.agent.role} is patrolling towards threat ID {threat_id} at ({target_x}, {target_y}) using action '{action}'.",
                    level=logging.INFO
                )
                return TaskState.ONGOING
            else:
                logger.debug_log(
                    f"{self.agent.role} could not execute movement action '{action}' while patrolling towards threat ID {threat_id}.",
                    level=logging.WARNING
                )
                return TaskState.FAILURE  # Penalize for missing movement action
        else:
            logger.debug_log(f"{self.agent.role} found no threats to patrol towards.", level=logging.WARNING)
            return TaskState.FAILURE  # Failure when no threats are found




#    ___ _         _   _____ _                 _   
#   | __(_)_ _  __| | |_   _| |_  _ _ ___ __ _| |_ 
#   | _|| | ' \/ _` |   | | | ' \| '_/ -_) _` |  _|
#   |_| |_|_||_\__,_|   |_| |_||_|_| \___\__,_|\__|
#                                                  

    

    

#    ___ _ _       _           _         _____ _                 _   
#   | __| (_)_ __ (_)_ _  __ _| |_ ___  |_   _| |_  _ _ ___ __ _| |_ 
#   | _|| | | '  \| | ' \/ _` |  _/ -_)   | | | ' \| '_/ -_) _` |  _|
#   |___|_|_|_|_|_|_|_||_\__,_|\__\___|   |_| |_||_|_| \___\__,_|\__|
#                                                                    



    def eliminate_threat(self, agents):
        """
        Attempt to eliminate the assigned threat.
        If it's not in combat range, opportunistically attack any nearby enemy within combat range.
        Fail the task if nothing is in range.
        """
        task = self.agent.current_task
        if not task:
            logger.debug_log(f"{self.agent.role} has no valid task assigned for elimination.", level=logging.WARNING)
            return TaskState.FAILURE

        threat = task.get("target")
        if not threat or "position" not in threat:
            logger.debug_log(f"{self.agent.role} could not find a valid threat in the task.", level=logging.WARNING)
            return TaskState.FAILURE

        assigned_position = threat["position"]
        assigned_id = threat.get("id", None)

        # ✅ Step 1: Try to attack the assigned threat if within combat range
        if self.agent.is_near(assigned_position, threshold=Agent_Attack_Range):
            target_agent = next((a for a in agents if a.agent_id == assigned_id), None)
            if target_agent and target_agent.faction.id != self.agent.faction.id:
                self.event_manager.trigger_attack_animation(position=(target_agent.x, target_agent.y), duration=200)
                target_agent.Health -= 10
                logger.debug_log(
                    f"{self.agent.role} attacked assigned threat {target_agent.role} (ID: {assigned_id}) at {assigned_position}. Health is now {target_agent.Health}.",
                    level=logging.INFO
                )
                if target_agent.Health <= 0:
                    self.report_threat_eliminated(threat)
                    return TaskState.SUCCESS
                return TaskState.ONGOING
        else:

        # 🔄 Step 2: Attack any other enemy agent within combat range
            for enemy in agents:
                if (
                    enemy.faction.id != self.agent.faction.id and
                    self.agent.is_near((enemy.x, enemy.y), threshold=Agent_Attack_Range)
                ):
                    self.event_manager.trigger_attack_animation(position=(enemy.x, enemy.y), duration=200)
                    enemy.Health -= 10
                    logger.debug_log(
                        f"{self.agent.role} attacked {enemy.role} (ID: {enemy.agent_id}) at ({enemy.x}, {enemy.y}). Health now {enemy.Health}.",
                        level=logging.INFO
                    )
                    if enemy.Health <= 0:
                        logger.debug_log(
                            f"{self.agent.role} eliminated nearby  {enemy.role}.",
                            level=logging.INFO
                        )
                    return TaskState.ONGOING

        # ❌ Nothing in range — fail the task this step
        logger.debug_log(f"{self.agent.role} could not reach assigned threat and found no enemies in attack range.", level=logging.INFO)
        return TaskState.FAILURE






    def get_agent_by_location(self, location, agents):
        """
        Find an agent by its location.
        """
        if not agents or not isinstance(agents, list):
            logger.debug_log(f"'agents' is invalid or missing. Cannot find agent by location.", level=logging.WARNING)
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
        logger.debug_log(
            f"{self.agent.role} cleaned resolved threats. Before: {before_cleanup}, After: {after_cleanup}.",
            level=logging.INFO
        )


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
                level=logging.WARNING
            )
            return

        # Mark the threat as inactive in the global state based on the unique ID
        for global_threat in self.agent.faction.global_state.get("threats", []):
            if global_threat.get("id") == threat.get("id"):  # Compare using unique ID
                global_threat["is_active"] = False
                logger.debug_log(
                    f"{self.agent.role} reported threat ID {threat['id']} at {threat.get('location')} as eliminated.",
                    level=logging.INFO
                )
                break  # Stop iteration once the matching threat is found
        else:
            # Log if the threat was not found in the global state
            logger.debug_log(
                f"Threat ID {threat.get('id')} not found in global state during elimination report.",
                level=logging.WARNING
            )

        # Clean up resolved threats
        self.clean_resolved_threats()

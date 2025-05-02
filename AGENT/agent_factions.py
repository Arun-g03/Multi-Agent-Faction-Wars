"""Common Imports"""
from SHARED.core_imports import *

"""File Specific Imports"""
from NEURAL_NETWORK.PPO_Agent_Network import PPOModel
from NEURAL_NETWORK.DQN_Model import DQNModel
from NEURAL_NETWORK.HQ_Network import HQ_Network
from NEURAL_NETWORK.Common import Training_device, get_hq_input_size_from_checkpoint
from AGENT.agent_communication import CommunicationSystem
import UTILITIES.utils_config as utils_config



logger = Logger(log_file="agent_factions.txt", log_level=logging.DEBUG)


class Faction():
    def __init__(
            self,
            game_manager,
            name,
            colour,
            id,
            resource_manager,
            agents,
            state_size,
            action_size,
            role_size,
            local_state_size,
            global_state_size,
            network_type="HQNetwork",
            mode: str = "train"):
        try:
            # Initialise Faction-specific attributes
            self.name = name
            self.colour = colour
            self.id = id
            self.agents = agents  # List of agents
            self.resource_manager = resource_manager  # Reference to the resource manager
            self.gold_balance = 0
            self.food_balance = 0
            self.current_strategy = None
            self.experience_buffer = []
            self.resources = []  # Initialise known resources
            self.threats = []  # Initialise known threats
            self.task_counter = 0
            self.assigned_tasks = {}
            

            self.unvisited_cells = set()
            self.reports = []
            self.strategy_history = []  # Track strategies chosen

            self.create_task = utils_config.create_task
            self.mode = mode

            # Initialise home_base with default values
            self.home_base = {
                "position": (0, 0),  # To be set during initialisation
                "size": 50,  # Default size of the base
                "colour": colour  # Match faction colour
            }

            self.game_manager = game_manager

            self.global_state = {
                key: None for key in utils_config.STATE_FEATURES_MAP["global_state"]}
            # Populate the initial global state
            self.global_state.update({
                "HQ_health": 100,  # Default HQ health
                "gold_balance": 0,  # Starting gold
                "food_balance": 0,  # Starting food
                "resource_count": 0,  # Total resources count
                "threat_count": 0,  # Total threats count
            })
            try:
                self.network_type = network_type
                self.network = self.initialise_network(
                    network_type,
                    state_size,
                    action_size,
                    role_size,
                    local_state_size,
                    global_state_size,
                    global_state=self.global_state)

                if self.network is None:
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"[ERROR] Network failed to Initialise for faction {self.id} (Type: {network_type})",
                            level=logging.ERROR)
                        e 
                    raise AttributeError(
                        f"[ERROR] Network failed to Initialise for faction {self.id}: {str(e)}")
                else:
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"[DEBUG] Faction {self.id}: Successfully initialised {type(self.network).__name__}",
                            level=logging.INFO)

            except Exception as e:
                import traceback
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[ERROR] Failed to Initialise network for faction {self.id}: {e}",
                        level=logging.ERROR)
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(traceback.format_exc(), level=logging.ERROR)

            self.communication_system = CommunicationSystem(
                self.agents, [self])

            # Populate unvisited cells with land tiles from the terrain grid
            for x in range(len(self.resource_manager.terrain.grid)):
                for y in range(len(self.resource_manager.terrain.grid[0])):
                    cell = self.resource_manager.terrain.grid[x][y]
                    if cell['type'] == 'land' and not cell['occupied']:
                        # Convert to pixel coordinates
                        self.unvisited_cells.add(
                            (x * utils_config.CELL_SIZE, y * utils_config.CELL_SIZE))

            self.health = 100  # Initial health

            # Dynamically calculate the input size for the critic
            if len(self.agents) > 0:
                critic_state = self.agents[0].get_state(
                    self.resource_manager, self.agents, self)
                input_size = len(critic_state)
            else:
                #print("No agents available for dynamic input size calculation. Using fallback input size.")
                input_size = 14  # Default fallback size

            # Now self.hq_network and self.critic are properly initialised via
            # network

            # Initialise the optimiser for the critic (if needed)
            if self.network is None:
                raise RuntimeError(
                    f"[FATAL] Faction {self.id} could not initialise a network. Check input sizes and network_type.")
            else:
                self.optimizer = torch.optim.Adam(
                    self.network.parameters(), lr=1e-3)

            self.strategy_update_interval = 100
            self.needs_strategy_retest = True
            self.current_step = 0
            self.hq_step_rewards = []

            # Final confirmation print
            print(f"{self.name} created with ID {self.id} and colour {self.colour}")

        except Exception as e:
            print(f"An error occurred in Faction class __init__: {e}")

            traceback.print_exc()  # This will print the full traceback to the console

    def initialise_network(
            self,
            network_type,
            state_size,
            action_size,
            role_size,
            local_state_size,
            global_state_size,
            global_state):
        import traceback
        try:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[DEBUG] Faction {self.id}: Attempting to Initialise {network_type}...",
                    level=logging.DEBUG)

            if any(x is None or x == 0 for x in [state_size, action_size]):
                logger.log_msg(
                    f"[ERROR] Invalid network inputs: state_size={state_size}, action_size={action_size}",
                    level=logging.ERROR)
                return None

            if network_type == "PPOModel":
                logger.log_msg(
                    f"[DEBUG] Initialising PPOModel...", level=logging.DEBUG)
                return PPOModel(state_size, action_size)

            elif network_type == "DQNModel":
                logger.log_msg(
                    f"[DEBUG] Initialising DQNModel...", level=logging.DEBUG)
                return DQNModel(state_size, action_size)

            elif network_type == "HQNetwork":
                try:
                    logger.log_msg(f"[DEBUG] Initialising HQNetwork...", level=logging.DEBUG)

                    # Check if we're loading a saved HQ model
                    if hasattr(self, "load_existing") and self.load_existing and hasattr(self, "models") and "HQ" in self.models:
                        from NEURAL_NETWORK.Common import get_hq_input_size_from_checkpoint  # if not already imported
                        input_size_from_ckpt = get_hq_input_size_from_checkpoint(self.models["HQ"])

                        # Deduct known components to infer state_size
                        role_size = utils_config.ROLE_VECTOR_SIZE       # or 5 if hardcoded
                        local_state_size = 5
                        global_state_size = 5
                        state_size = input_size_from_ckpt - (role_size + local_state_size + global_state_size)

                        if utils_config.ENABLE_LOGGING:
                            logger.log_msg(f"[HQ LOAD MODE] Using inferred state_size={state_size} from checkpoint input size={input_size_from_ckpt}",
                                        level=logging.INFO)
                    

                    return HQ_Network(
                        state_size=state_size,
                        action_size=action_size,
                        role_size=role_size,
                        local_state_size=local_state_size,
                        global_state_size=global_state_size,
                        device=Training_device,
                        global_state=self.global_state
                    )
                except Exception as e:
                    logger.log_msg(f"[ERROR] Failed to initialise HQNetwork: {e}", level=logging.ERROR)
                    raise
                    

        except Exception as e:
            logger.log_msg(
                f"[ERROR] Network initialisation failed for Faction {self.id} (Type: {network_type}): {e}",
                level=logging.ERROR)
            logger.log_msg(traceback.format_exc(), level=logging.ERROR)
            return None

    def update(self, resource_manager, agents, current_step):
        self.clean_global_state()
        self.calculate_territory(resource_manager.terrain)
        self.aggregate_faction_state()
        self.current_step = current_step

        # Only update HQ strategy every N steps
        if current_step % self.strategy_update_interval == 0 or self.needs_strategy_retest:
            new_strategy = self.choose_HQ_Strategy()
            if new_strategy != self.current_strategy:
                print(f"\033[94mFaction {self.id} has changed HQ from {self.current_strategy} to {new_strategy}.\033[0m\n")
                self.perform_HQ_Strategy(new_strategy)
            else:
                print(f"\033[93m Faction {self.id} maintained HQ strategy: {self.current_strategy}\033[0m")
                self.current_strategy = new_strategy
                self.perform_HQ_Strategy(self.current_strategy)
            self.needs_strategy_retest = False

        self.update_tasks(agents)

        self.assign_high_level_tasks()

    

    def remove_agent(self, agent):
        """
        Safely remove an agent from the faction's agent list.
        """
        if agent in self.agents:
            self.agents.remove(agent)
            print(f"Agent {agent.role} removed from faction {self.id}.")
        else:
            print(
                f"Warning: Agent {agent.role} not found in faction {self.id}.")

    def receive_experience(self, experience):
        """
        Store experience for centralised training.
        """
        self.experience_buffer.append(experience)

    def clear_experience_buffer(self):
        """
        Clear the experience buffer after training.
        """
        self.experience_buffer = []


#       _                                    _                                   _          __                                              _
#      / \   __ _  __ _ _ __ ___  __ _  __ _| |_ ___   _ __ ___ _ __   ___  _ __| |_ ___   / _|_ __ ___  _ __ ___     __ _  __ _  ___ _ __ | |_ ___
#     / _ \ / _` |/ _` | '__/ _ \/ _` |/ _` | __/ _ \ | '__/ _ \ '_ \ / _ \| '__| __/ __| | |_| '__/ _ \| '_ ` _ \   / _` |/ _` |/ _ \ '_ \| __/ __|
#    / ___ \ (_| | (_| | | |  __/ (_| | (_| | ||  __/ | | |  __/ |_) | (_) | |  | |_\__ \ |  _| | | (_) | | | | | | | (_| | (_| |  __/ | | | |_\__ \
#   /_/   \_\__, |\__, |_|  \___|\__, |\__,_|\__\___| |_|  \___| .__/ \___/|_|   \__|___/ |_| |_|  \___/|_| |_| |_|  \__,_|\__, |\___|_| |_|\__|___/
#    _      |___/ |___/       _  |___/_           _       _    |_| _                                                       |___/
#   (_)_ __ | |_ ___     __ _| | ___ | |__   __ _| |  ___| |_ __ _| |_ ___
#   | | '_ \| __/ _ \   / _` | |/ _ \| '_ \ / _` | | / __| __/ _` | __/ _ \
#   | | | | | || (_) | | (_| | | (_) | |_) | (_| | | \__ \ || (_| | ||  __/
#   |_|_| |_|\__\___/   \__, |_|\___/|_.__/ \__,_|_| |___/\__\__,_|\__\___|
#                       |___/

    def aggregate_faction_state(self):
        """
        Aggregate faction-wide state, ensuring all required features exist.
        """

        #  Ensure required fields exist before processing
        required_keys = [
            "HQ_health",
            "gold_balance",
            "food_balance",
            "resource_count",
            "threat_count",
            "nearest_threat",
            "nearest_resource",
            "friendly_agent_count",
            "enemy_agent_count",
            "agent_density",
            "total_agents"]

        for key in required_keys:
            if key not in self.global_state:
                self.global_state[key] = 0  # Default missing values to zero

        #  Ensure HQ position exists before using it
        if "position" not in self.home_base or self.home_base["position"] == (
                0,
                0):
            print(
                f"[WARNING] HQ position for Faction {self.id} is missing! Assigning default location.")
            self.home_base["position"] = (random.randint(
                0, 100), random.randint(0, 100))  # Assign a random position

        hq_x, hq_y = self.home_base["position"]

        # Ensure `nearest_threat` and `nearest_resource` are structured
        # correctly
        self.global_state["nearest_threat"] = self.global_state.get(
            "nearest_threat", {"location": (-1, -1)})
        self.global_state["nearest_resource"] = self.global_state.get(
            "nearest_resource", {"location": (-1, -1)})

        #  Fetch agents correctly
        all_agents = self.game_manager.agents  # Game-wide agents
        enemy_agents = [agent for agent in all_agents if agent.faction != self]

        #  Compute agent-related metrics
        self.global_state["friendly_agent_count"] = len(
            self.agents)  # Use faction-level agents
        self.global_state["enemy_agent_count"] = len(
            enemy_agents)  # Use game-wide agents
        self.global_state["total_agents"] = len(
            all_agents)  # Use game-wide agents

        #  Compute agent density near HQ
        nearby_agents = [agent for agent in self.agents if (
            (agent.x - hq_x)**2 + (agent.y - hq_y)**2) ** 0.5 < 50]
        self.global_state["agent_density"] = len(nearby_agents)

        #  Ensure `agent_states` are properly formatted
        self.global_state["agent_states"] = [
            agent.get_state(self.resource_manager, self.agents,
                            self, self.global_state)
            for agent in self.agents
        ]

        #  Debug Log to verify correct agent count
        # if utils_config.ENABLE_LOGGING: logger.log_msg(f"[DEBUG] Faction {self.id} State: {self.global_state}")

        return self.global_state

    def receive_report(self, report):
        """Process reports received from agents."""
        if "type" not in report or "data" not in report:
            logger.warning(
                f"Invalid report format received by Faction {self.id}: {report}")
            return

        report_type = report["type"]
        data = report["data"]

        if report_type == "threat":
            threat_id = data.get("id")
            location = data.get("location")

            if "threats" not in self.global_state:
                self.global_state["threats"] = []

            # Use string comparison fallback if utils_config.AgentIDStruc not
            # hashable
            existing_threat = next(
                (t for t in self.global_state["threats"]
                 if str(t.get("id")) == str(threat_id)),
                None
            )

            if existing_threat:
                if existing_threat["location"] != location:
                    existing_threat["location"] = location
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"Faction {self.id} updated threat ID {threat_id} to location {location}.")

            else:
                self.global_state["threats"].append(data)
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"Faction {self.id} added new threat: {data['type']} ID {threat_id} at {location}.")

        elif report_type == "resource":
            # Extract relevant data from the resource object
            if hasattr(
                data,
                "grid_x") and hasattr(
                data,
                "grid_y") and hasattr(
                data,
                    "__class__"):
                resource_data = {
                    "location": (data.grid_x, data.grid_y),
                    "type": data.__class__.__name__,
                }

                # Check if the resource already exists
                existing_resource = next(
                    (res for res in self.global_state["resources"]
                     if res["location"] == resource_data["location"]),
                    None
                )

                if existing_resource:
                    pass  # Do nothing if the resource already exists
                else:
                    self.global_state["resources"].append(resource_data)
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"Faction {self.id} added resource: {resource_data}.")
            else:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"Invalid resource object in report for Faction {self.id}: {data}")

        else:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"Unknown report type '{report_type}' received by Faction {self.id}: {report}")

    def provide_state(self):
        """
        Provide the faction's global state to a requester.
        """
        return self.global_state


#     ____ _                    _   _                  _       _           _       _        _
#    / ___| | ___  __ _ _ __   | |_| |__   ___    __ _| | ___ | |__   __ _| |  ___| |_ __ _| |_ ___
#   | |   | |/ _ \/ _` | '_ \  | __| '_ \ / _ \  / _` | |/ _ \| '_ \ / _` | | / __| __/ _` | __/ _ \
#   | |___| |  __/ (_| | | | | | |_| | | |  __/ | (_| | | (_) | |_) | (_| | | \__ \ || (_| | ||  __/
#    \____|_|\___|\__,_|_| |_|  \__|_| |_|\___|  \__, |_|\___/|_.__/ \__,_|_| |___/\__\__,_|\__\___|
#                                                |___/

    def clean_global_state(self):
        """
        Clean outdated entries in the global state and ensure required features exist.
        This includes validating resources and threats against the actual environment.
        """
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[DEBUG] Cleaning global state for Faction {self.id} BEFORE reset: {self.global_state}",
                level=logging.DEBUG)

        # ========== RESOURCE CLEANUP ==========
        original_resources = len(self.global_state.get("resources", []))
        valid_resources = []

        for res in self.global_state.get("resources", []):
            loc = res.get("location")
            if not loc:
                continue

            match = next(
                (r for r in self.resource_manager.resources
                 if hasattr(r, "grid_x") and hasattr(r, "grid_y")
                 and (r.grid_x, r.grid_y) == loc and not r.is_depleted()),
                None
            )
            if match:
                valid_resources.append(res)

        self.global_state["resources"] = valid_resources
        self.global_state["resource_count"] = len(valid_resources)

        # ========== THREAT CLEANUP ==========
        original_threats = len(self.global_state.get("threats", []))
        valid_threats = []

        for threat in self.global_state.get("threats", []):
            tid = threat.get("id")

            if threat["type"] == "Faction HQ":
                # Keep HQ threats unless you add HQ destruction logic
                valid_threats.append(threat)
                continue

            if isinstance(tid, utils_config.AgentIDStruc):
                is_alive = any(
                    agent for agent in self.game_manager.agents if getattr(
                        agent, "agent_id", None) == tid and getattr(
                        agent, "Health", 1) > 0)
                if is_alive:
                    valid_threats.append(threat)

        self.global_state["threats"] = valid_threats
        self.global_state["threat_count"] = len(valid_threats)

        # ========== NEAREST PLACEHOLDER FIELDS ==========
        self.global_state["nearest_threat"] = self.global_state.get(
            "nearest_threat", {"location": (-1, -1)})
        self.global_state["nearest_resource"] = self.global_state.get(
            "nearest_resource", {"location": (-1, -1)})

        # ========== AGENT STATS ==========
        self.global_state["friendly_agent_count"] = len(self.agents)

        enemy_agents = [
            agent for agent in self.game_manager.agents if agent.faction != self]
        self.global_state["enemy_agent_count"] = len(enemy_agents)

        hq_x, hq_y = self.home_base["position"]
        nearby_agents = [agent for agent in self.agents if (
            (agent.x - hq_x) ** 2 + (agent.y - hq_y) ** 2) ** 0.5 < 50]
        self.global_state["agent_density"] = len(nearby_agents)

        self.global_state["total_agents"] = len(self.game_manager.agents)

        # ========== BASE STATE FIELDS ==========
        self.global_state["HQ_health"] = self.global_state.get(
            "HQ_health", 100)
        self.global_state["gold_balance"] = self.global_state.get(
            "gold_balance", 0)
        self.global_state["food_balance"] = self.global_state.get(
            "food_balance", 0)

        # ========== LOGGING ==========
        removed_resources = original_resources - len(valid_resources)
        removed_threats = original_threats - len(valid_threats)
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[CLEAN] Pruned {removed_resources} resources, {removed_threats} threats.",
                level=logging.DEBUG)

            state = self.global_state
            resource_strs = [r.get("location", "?")
                             for r in state.get("resources", [])]
            threat_strs = [
                f"{t.get('type', '?')}@{t.get('location', '?')}" for t in state.get("threats", [])]

            formatted_state = (
                f"\n[GLOBAL STATE] Faction {self.id}\n"
                f"  Resources     ({state.get('resource_count', 0)}): {resource_strs}\n"
                f"  Threats       ({state.get('threat_count', 0)}): {threat_strs}\n"
                f"  Nearest Resource: {state.get('nearest_resource', {}).get('location')}\n"
                f"  Nearest Threat : {state.get('nearest_threat', {}).get('location')}\n"
                f"  Friendly Agents: {state.get('friendly_agent_count', 0)}\n"
                f"  Enemy Agents   : {state.get('enemy_agent_count', 0)}\n"
                f"  Agent Density  : {state.get('agent_density', 0)} (near HQ)\n"
                f"  Total Agents   : {state.get('total_agents', 0)}\n"
                f"  HQ Health      : {state.get('HQ_health', 100)}\n"
                f"  Gold           : {state.get('gold_balance', 0)}\n"
                f"  Food           : {state.get('food_balance', 0)}\n")
            logger.log_msg(formatted_state, level=logging.DEBUG)

    """


    #       _    ____ ____ ___ ____ _   _   _   _ ___ ____ _   _       _     _______     _______ _       _____  _    ____  _  __
    #      / \\  / ___/ ___|_ _/ ___| \\ | | | | | |_ _/ ___| | | |     | |   | ____\\ \\   / / ____| |     |_   _|/ \\  / ___|| |/ /
    #     / _ \\ \\___ \\___ \\| | |  _|  \\| | | |_| || | |  _| |_| |_____| |   |  _|  \\ \\ / /|  _| | |       | | / _ \\ \\___ \\| ' /
    #    / ___ \\ ___) |__) | | |_| | |\\  | |  _  || | |_| |  _  |_____| |___| |___  \\ V / | |___| |___    | |/ ___ \\ ___) | . \
    #   /_/   \\_\\____/____/___\\____|_| \\_| |_| |_|___\\____|_| |_|     |_____|_____|  \\_/  |_____|_____|   |_/_/   \\_\\____/|_|\\_\
    #

    ========================================================================================================
    """

    def assign_high_level_tasks(self):
        """
        HQ network assigns new high-level tasks to idle agents based on strategy.
        """
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(f"[HQ] Faction {self.id} assigning high-level tasks...", level=logging.INFO)

        for agent in self.agents:
            # === 1. Check if agent is idle ===
            agent_idle = False

            if agent.current_task is None:
                agent_idle = True

            elif agent.current_task_state in [utils_config.TaskState.SUCCESS, utils_config.TaskState.FAILURE]:
                agent_idle = True

            elif agent.current_task.get("type", "none") == "none":
                agent_idle = True

            # === 2. Clear completed tasks if needed ===
            if agent_idle:
                if agent.current_task:
                    self.complete_task(agent.current_task.get("id"), agent, agent.current_task_state)
                agent.current_task = None
                agent.update_task_state(utils_config.TaskState.NONE)

            # === 3. Skip agents still busy ===
            if agent.current_task:
                continue  # Still busy with a valid task

            # === 4. Assign a new task ===
            task = self.assign_task(agent)
            if task:
                agent.current_task = task
                agent.update_task_state(utils_config.TaskState.ONGOING)

                self.task_counter += 1
                task_counter_id = self.task_counter

                self.assigned_tasks[task_counter_id] = {
                    "type": task["type"],                     # task type: 'explore', 'mine', etc.
                    "task_id": task["id"],                     # e.g., "Explore-(19,17)"
                    "start_step": self.current_step,           # when assigned
                    "end_step": None,                          # filled when completed
                    "agents": {
                        str(agent.agent_id): utils_config.TaskState.ONGOING
                    }
                }

                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[TASK ASSIGNED] {agent.agent_id} => {task['type']} at {task['target'].get('position')}",
                        level=logging.INFO)
                    logger.log_msg(
                        f"[DEBUG] {agent.agent_id} has task: {agent.current_task}",
                        level=logging.DEBUG)


    """

    #       _            _               _            _          _                                 _
    #      / \\   ___ ___(_) __ _ _ __   | |_ __ _ ___| | _____  | |_ ___     __ _  __ _  ___ _ __ | |_ ___
    #     / _ \\ / __/ __| |/ _` | '_ \\  | __/ _` / __| |/ / __| | __/ _ \\   / _` |/ _` |/ _ \\ '_ \\| __/ __|
    #    / ___ \\__ \\__ \\ | (_| | | | | | || (_| \\__ \\   <\\__ \\ | || (_) | | (_| | (_| |  __/ | | | |_\\__ \
    #   /_/   \\_\\___/___/_|\\__, |_| |_|  \\__\\__,_|___/_|\\_\\___/  \\__\\___/   \\__,_|\\__, |\\___|_| |_|\\__|___/
    #                      |___/                                                  |___/
    ========================================================================================================
    """

    # This is where tasks are created

    def assign_task(self, agent) -> Optional[dict]:
        role = getattr(agent, "role", None)
        strategy = self.current_strategy or "NO_PRIORITY"

        # Determine threats
        threats = [
            t for t in self.global_state.get("threats", [])
            if t["id"].faction_id != self.id
        ]
        has_threats = len(threats) > 0


        if role not in ["gatherer", "peacekeeper"]:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[WARN] Unknown role {agent.agent_id}: {role}",
                    level=logging.WARNING)
            return None

        if role == "gatherer":
            current = agent.current_task
            current_type = current["type"] if current else "none"

            if isinstance(current_type, str):
                current_type_id = utils_config.TASK_TYPE_MAPPING.get(current_type, utils_config.TASK_TYPE_MAPPING["none"])
            else:
                current_type_id = current_type

            is_resource_task = (current_type_id == utils_config.TASK_TYPE_MAPPING["gather"])

            if is_resource_task:
                return current

            if current:
                agent.current_task = None
                agent.update_task_state(utils_config.TaskState.NONE)

            # === Main logic for gatherers ===
            if strategy in ["COLLECT_GOLD", "COLLECT_FOOD"]:
                # Directly assign based on strategy
                if strategy == "COLLECT_GOLD":
                    return self.assign_mining_task(agent)
                else:
                    return self.assign_forage_task(agent)

            elif strategy == "ATTACK_THREATS":
                # If war happening, gatherers still focus on food/gold
                options = list(filter(None, [
                    self.assign_forage_task(agent),
                    self.assign_mining_task(agent)
                ]))
                if options:
                    return random.choice(options)

            # Default to exploration
            return self.assign_explore_task(agent)


        elif role == "peacekeeper":
            current = agent.current_task
            current_type = current["type"] if current else "none"

            if isinstance(current_type, str):
                current_type_id = utils_config.TASK_TYPE_MAPPING.get(current_type, utils_config.TASK_TYPE_MAPPING["none"])
            else:
                current_type_id = current_type

            if strategy == "ATTACK_THREATS" and has_threats:
                threat = min(threats, key=lambda t: self.calculate_threat_weight(agent, t))
                already_eliminating = (
                    current_type_id == utils_config.TASK_TYPE_MAPPING["eliminate"]
                )

                if not already_eliminating:
                    if current:
                        agent.current_task = None
                        agent.update_task_state(utils_config.TaskState.NONE)
                    return self.assign_eliminate_task(agent, threat["id"], threat["type"], threat["location"])

            elif strategy == "DEFEND_HQ":
                hq_position = self.home_base["position"]
                if not utils_config.SUB_TILE_PRECISION:
                    # If we are using grid precision, convert pixel HQ to grid
                    hq_position = (
                        int(hq_position[0] // utils_config.CELL_SIZE),
                        int(hq_position[1] // utils_config.CELL_SIZE)
                    )
                already_defending = (
                    current_type_id == utils_config.TASK_TYPE_MAPPING["move_to"] and
                    current and current.get("target", {}).get("position") == hq_position
                )

                if not already_defending:
                    if current:
                        agent.current_task = None
                        agent.update_task_state(utils_config.TaskState.NONE)
                    return self.assign_move_to_task(agent, hq_position, label="DefendHQ")

            elif strategy in ["COLLECT_GOLD", "COLLECT_FOOD", "NO_PRIORITY"]:
                # Peacekeepers fallback to exploration
                is_explore_move = (
                    current_type_id == utils_config.TASK_TYPE_MAPPING["move_to"] and
                    current and current.get("id", "").startswith("Explore-")
                )
                if is_explore_move and agent.current_task_state == utils_config.TaskState.ONGOING:
                    return current
                return self.assign_explore_task(agent)

            return current  # fallback





            

        else:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[WARN] - assign_task - Unknown role {agent.agent_id}: {role}",
                    level=logging.WARNING)
            return None

    """

            #    _____  _    ____  _  __  _   _    _    _   _ ____  _     _____ ____  ____
            #   |_   _|/ \\  / ___|| |/ / | | | |  / \\  | \\ | |  _ \\| |   | ____|  _ \\/ ___|
            #     | | / _ \\ \\___ \\| ' /  | |_| | / _ \\ |  \\| | | | | |   |  _| | |_) \\___ \
            #     | |/ ___ \\ ___) | . \\  |  _  |/ ___ \\| |\\  | |_| | |___| |___|  _ < ___) |
            #     |_/_/   \\_\\____/|_|\\_\\ |_| |_/_/   \\_\\_| \\_|____/|_____|_____|_| \\_\\____/
            #
    ================================================================================================

    """

    def assign_move_to_task(self, agent, position, label=None) -> Optional[dict]:
        """
        Assigns a simple move_to task to the given agent toward a position.
        Ensures the agent sticks to this task until it reaches the target.
        """
        if not position or not isinstance(position, tuple) or len(position) != 2:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[MOVE_TO] Invalid target position for agent {agent.agent_id}: {position}",
                    level=logging.WARNING)
            return None

        # Format task_id consistently like mining tasks
        task_id = label or f"MoveTo-{position}"
        target = {
            "position": position,
            "type": "Location"
        }

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"Created task: {task_id} for agent {agent.agent_id}",
                level=logging.INFO)

        task = utils_config.create_task(self, "move_to", target, task_id)

        # Mark the task as ongoing
        agent.current_task = task
        agent.update_task_state(utils_config.TaskState.ONGOING)

        return task



    def assign_eliminate_task(
            self,
            agent,
            target_id,
            target_type,
            position) -> Optional[dict]:
        try:
            task_id = f"Eliminate-{target_id}"
            target = {
                "id": target_id,
                "type": target_type,
                "position": position
            }
            logger.log_msg(
                f"Created task: {task_id} for agent {agent.agent_id}",
                level=logging.INFO)
            return utils_config.create_task(self, "eliminate", target, task_id)
        except Exception as e:
            logger.log_msg(
                f"Failed to create eliminate task: {e}", level=logging.ERROR)
            return None

    def assign_forage_task(self, agent) -> Optional[dict]:
        resources = [r for r in self.global_state.get(
            "resources", []) if r["type"] == "AppleTree"]
        if not resources:
            return None
        nearest = min(
            resources, key=lambda r: self.calculate_resource_weight(agent, r))
        task_id = f"Forage-{nearest['location']}"
        target = {"position": nearest["location"], "type": "AppleTree"}
        logger.log_msg(
            f"Created task: {task_id} for agent {agent.agent_id}",
            level=logging.INFO)
        return utils_config.create_task(self, "gather", target, task_id)

    def assign_mining_task(self, agent) -> Optional[dict]:
        resources = [r for r in self.global_state.get(
            "resources", []) if r["type"] == "GoldLump"]
        if not resources:
            return None
        nearest = min(
            resources, key=lambda r: self.calculate_resource_weight(agent, r))
        task_id = f"Mine-{nearest['location']}"
        target = {"position": nearest["location"], "type": "GoldLump"}
        logger.log_msg(
            f"Created task: {task_id} for agent {agent.agent_id}",
            level=logging.INFO)
        return utils_config.create_task(self, "gather", target, task_id)

    def assign_explore_task(self, agent) -> Optional[dict]:
        terrain = self.resource_manager.terrain

        self.calculate_territory(terrain)  # track current control
        logger.log_msg(f"[EXPLORE] Faction {self.id} controls {self.territory_count} tiles.", level=logging.INFO)

        unexplored_cells = []
        for x in range(len(terrain.grid)):
            for y in range(len(terrain.grid[0])):
                cell = terrain.grid[x][y]

                # Generate the expected task_id (string)
                task_id = f"Explore-({x}, {y})"

                # Search yur assigned_tasks
                assigned_count = 0
                for task_info in self.assigned_tasks.values():
                    if task_info["task_id"] == task_id:
                        assigned_count += len(task_info["agents"])

                # Now check if can assign
                if (
                    cell["faction"] != self.id
                    and cell["type"] == "land"
                    and assigned_count < 2
                ):
                    unexplored_cells.append((x, y))


        logger.log_msg(f"[EXPLORE] Found {len(unexplored_cells)} unexplored cells for agent {agent.agent_id}", level=logging.INFO)

        if unexplored_cells:
            cell_x, cell_y = random.choice(unexplored_cells)
            task_string_id = f"Explore-({cell_x}, {cell_y})"
            target = {"position": (cell_x, cell_y), "type": "Explore"}

            logger.log_msg(f"Created task: {task_string_id} for agent {agent.agent_id}", level=logging.INFO)

            # Create the task dictionary
            task = utils_config.create_task(self, "move_to", target, task_string_id)

            agent.current_task = task
            agent.update_task_state(utils_config.TaskState.ONGOING)

            #  New way: track it properly
            if not hasattr(self, "task_counter"):
                self.task_counter = 0
            self.task_counter += 1
            task_counter_id = self.task_counter

            self.assigned_tasks[task_counter_id] = {
                "type": "explore",
                "task_id": task_string_id,
                "start_step": self.current_step,
                "end_step": None,
                "agents": {
                    str(agent.agent_id): utils_config.TaskState.ONGOING
                }
            }

            return task

        logger.log_msg(f"[EXPLORE] No valid unexplored cells left for agent {agent.agent_id}", level=logging.WARNING)
        return None




    """

    #     ____      _            _       _         _            _                   _       _     _
    #    / ___|__ _| | ___ _   _| | __ _| |_ ___  | |_ __ _ ___| | __ __      _____(_) __ _| |__ | |_ ___
    #   | |   / _` | |/ __| | | | |/ _` | __/ _ \\ | __/ _` / __| |/ / \\ \\ /\\ / / _ \\ |/ _` | '_ \\| __/ __|
    #   | |__| (_| | | (__| |_| | | (_| | ||  __/ | || (_| \\__ \\   <   \\ V  V /  __/ | (_| | | | | |_\\__ \
    #    \\____\\__,_|_|\\___|\\__,_|_|\\__,_|\\__\\___|  \\__\\__,_|___/_|\\_\\   \\_/\\_/ \\___|_|\\__, |_| |_|\\__|___/
    #                                                                                 |___/
    =======================================================================================================
    """

    def calculate_resource_weight(self, agent, resource: dict) -> float:
        """
        Estimates priority weight for a resource (lower is better).
        Expects a dict with 'location' and 'quantity'.
        """
        if resource is None or "location" not in resource:
            return float("inf")

        rx, ry = resource["location"]
        ax, ay = agent.x // utils_config.CELL_SIZE, agent.y // utils_config.CELL_SIZE

        distance = ((rx - ax) ** 2 + (ry - ay) ** 2) ** 0.5
        quantity = resource.get("quantity", 0)

        weight = distance / (quantity + 1)
        return weight

    def calculate_threat_weight(self, agent, threat):
        """
        Calculate the weight of a threat task based on proximity and threat type.
        Lower weight is better (higher priority).
        """
        distance = ((threat["location"][0] - agent.x) **
                    2 + (threat["location"][1] - agent.y) ** 2) ** 0.5
        type_weight = 1  # Default weight for general threats
        if threat["type"] == "Faction HQ":
            type_weight = 0.5  # Higher priority for enemy HQs
        weight = distance * type_weight
        return weight


#         _               _      _   _            _            _                                                    _      _           _
#     ___| |__   ___  ___| | __ | |_| |__   ___  | |_ __ _ ___| | __ __      ____ _ ___    ___ ___  _ __ ___  _ __ | | ___| |_ ___  __| |
#    / __| '_ \ / _ \/ __| |/ / | __| '_ \ / _ \ | __/ _` / __| |/ / \ \ /\ / / _` / __|  / __/ _ \| '_ ` _ \| '_ \| |/ _ \ __/ _ \/ _` |
#   | (__| | | |  __/ (__|   <  | |_| | | |  __/ | || (_| \__ \   <   \ V  V / (_| \__ \ | (_| (_) | | | | | | |_) | |  __/ ||  __/ (_| |
#    \___|_| |_|\___|\___|_|\_\  \__|_| |_|\___|  \__\__,_|___/_|\_\   \_/\_/ \__,_|___/  \___\___/|_| |_| |_| .__/|_|\___|\__\___|\__,_|
#                                                                                                            |_|


    def complete_task(self, task_id_str, agent, task_state):
        """
        Mark an agent's task as completed or failed. Auto-fill end_step when task is fully finished.
        """

        agent_id_str = str(agent.agent_id)

        # Search for the task_counter that matches this task_id
        for task_counter_id, task_data in self.assigned_tasks.items():
            if task_data["task_id"] == task_id_str:
                if agent_id_str in task_data["agents"]:
                    task_data["agents"][agent_id_str] = task_state

                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"[TASK COMPLETE] {agent_id_str} marked as {task_state.name} in {task_id_str}.",
                            level=logging.INFO
                        )

                    #  Check if ALL agents finished
                    if all(
                        state in [utils_config.TaskState.SUCCESS, utils_config.TaskState.FAILURE]
                        for state in task_data["agents"].values()
                    ):
                        if task_data["end_step"] is None:
                            task_data["end_step"] = self.current_step
                            logger.log_msg(
                                f"[TASK COMPLETE] Task {task_id_str} fully completed at step {self.current_step}.",
                                level=logging.INFO
                            )
                else:
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"[WARN] Agent {agent_id_str} not found in task {task_id_str} agent list.",
                            level=logging.WARNING
                        )
                break
        else:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[WARN] Task {task_id_str} not found in assigned_tasks!",
                    level=logging.WARNING
                )



    def update_tasks(self, agents):
        for agent in agents:
            # Skip agents from other factions
            if agent.faction != self:
                continue

            # If task completed, clear it
            if agent.current_task_state in [utils_config.TaskState.SUCCESS, utils_config.TaskState.FAILURE]:
                if agent.current_task:
                    self.complete_task(agent.current_task["id"], agent, agent.current_task_state)
                    agent.current_task = None
                    agent.update_task_state(utils_config.TaskState.NONE)


    def calculate_territory(self, terrain):
        """Calculate the number of cells owned by this faction."""
        self.territory_count = sum(
            1 for row in terrain.grid for cell in row if cell['faction'] == self.id)

    ##########################################################################

    #    _  _  ___     _______                 ___ _            _
    #   | || |/ _ \   / /_   _|__ __ _ _ __   / __| |_ _ _ __ _| |_ ___ __ _ _  _
    #   | __ | (_) | / /  | |/ -_) _` | '  \  \__ \  _| '_/ _` |  _/ -_) _` | || |
    #   |_||_|\__\_\/_/   |_|\___\__,_|_|_|_| |___/\__|_| \__,_|\__\___\__, |\_, |
    #                                                                  |___/ |__/

    ##########################################################################

    def choose_HQ_Strategy(self):
        """
        HQ chooses a strategy using its neural network (or fallback),
        and stores it as the current_strategy.
        """
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[HQ STRATEGY] Faction {self.id} requesting decision from HQ network...",
                level=logging.INFO)

        strategy = "NO_PRIORITY"  # Default
        previous_strategy = self.current_strategy

        if hasattr(self, "network") and self.network:
            predicted = self.network.predict_strategy(self.global_state)

            if predicted in utils_config.HQ_STRATEGY_OPTIONS:
                strategy = predicted
                if strategy != previous_strategy:
                    logger.log_msg(
                        f"[HQ STRATEGY] Faction {self.id} network picked different strategy: {strategy}",
                        level=logging.INFO)
                else:
                    logger.log_msg(
                        f"[HQ STRATEGY] Faction {self.id} network continued with strategy: {strategy}",
                        level=logging.INFO)
            else:
                logger.log_msg(
                    f"[HQ STRATEGY] Invalid strategy returned: {predicted}. Defaulting to NO_PRIORITY.",
                    level=logging.WARNING)
        else:
            logger.log_msg(
                f"[HQ STRATEGY] No HQ network found. Falling back to NO_PRIORITY.",
                level=logging.WARNING)

        self.current_strategy = strategy
        self.strategy_history.append(strategy)  #  Track it
        return strategy


    def perform_HQ_Strategy(self, action):
        """
        HQ executes the chosen strategic action if valid.
        If invalid, it re-evaluates strategy using the HQ network.
        """
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[HQ EXECUTE] Faction {self.id} attempting strategy: {action}",
                level=logging.INFO)

        def retest_strategy():
            new_action = self.choose_HQ_Strategy()
            if new_action != action:
                logger.log_msg(
                    f"[HQ RETEST] Strategy '{action}' invalid. Retesting and switching to '{new_action}'",
                    level=logging.WARNING)
                self.perform_HQ_Strategy(new_action)
            logger.log_msg(
                f"[HQ EXECUTE] Faction {self.id} executing updated strategy: {new_action}",
                level=logging.INFO)

        

        # ========== STRATEGY: Recruit Peacekeeper ==========
        if action == "RECRUIT_PEACEKEEPER":
            Agent_cost = utils_config.Gold_Cost_for_Agent
            if self.gold_balance >= Agent_cost and len(self.agents) < utils_config.MAX_AGENTS:
                self.gold_balance -= Agent_cost
                self.recruit_agent("peacekeeper")
                print(f"Faction {self.id} bought a Peacekeeper")
                self.hq_step_rewards.append(+1.0)
            else:
                reason = "Not enough gold" if self.gold_balance < Agent_cost else "Agent limit reached"
                logger.log_msg(f"[HQ EXECUTE] Cannot recruit peacekeeper: {reason}.", level=logging.WARNING)
                self.current_strategy = None
                self.hq_step_rewards.append(-0.5)
                return retest_strategy()

        # ========== STRATEGY: Recruit Gatherer ==========
        elif action == "RECRUIT_GATHERER":
            Agent_cost = utils_config.Gold_Cost_for_Agent
            if self.gold_balance >= Agent_cost and len(self.agents) < utils_config.MAX_AGENTS:
                self.gold_balance -= Agent_cost
                self.recruit_agent("gatherer")
                print(f"Faction {self.id} bought a Gatherer")
                self.hq_step_rewards.append(+1.0)
            else:
                reason = "Not enough gold" if self.gold_balance < Agent_cost else "Agent limit reached"
                logger.log_msg(f"[HQ EXECUTE] Cannot recruit gatherer: {reason}.", level=logging.WARNING)
                self.current_strategy = None
                self.hq_step_rewards.append(-0.5)
                return retest_strategy()


        # ========== STRATEGY: Defend HQ ==========
        elif action == "DEFEND_HQ":
            DEFENSE_RADIUS = 100  # pixels
            DEFENSE_RADIUS_SQ = DEFENSE_RADIUS ** 2

            # Convert HQ position to pixel coords if needed
            hx, hy = self.home_base["position"]
            if isinstance(hx, int) and hx < utils_config.WORLD_WIDTH:
                hx *= utils_config.CELL_SIZE
                hy *= utils_config.CELL_SIZE
            hq_pos = (hx, hy)

            threats = self.global_state.get("threats", [])
            nearby_threat_found = False

            for threat in threats:
                if not isinstance(threat.get("id"), utils_config.AgentIDStruc):
                    continue
                if threat["id"].faction_id == self.id:
                    continue  # Skip own faction

                tx, ty = threat["location"]
                if isinstance(tx, int) and tx < utils_config.WORLD_WIDTH:
                    tx *= utils_config.CELL_SIZE
                    ty *= utils_config.CELL_SIZE

                dist_sq = (tx - hx) ** 2 + (ty - hy) ** 2
                if dist_sq <= DEFENSE_RADIUS_SQ:
                    nearby_threat_found = True
                    break
            self.hq_step_rewards.append(+1.0)

            if not nearby_threat_found:
                logger.log_msg(
                    f"[HQ STRATEGY] No nearby threats to defend HQ.",
                    level=logging.WARNING)
                self.hq_step_rewards.append(-0.5)
                return retest_strategy()

            # Strategy is valid — assign peacekeepers to move to HQ
            self.defensive_position = hq_pos
            logger.log_msg(
                f"[HQ STRATEGY] Nearby threat detected. Assigning peacekeepers to defend HQ at {hq_pos}.",
                level=logging.INFO)

            for agent in self.agents:
                if agent.role != "peacekeeper":
                    continue

                current = agent.current_task
                already_defending = (
                    current and current.get("type") == "move_to" and
                    current.get("target", {}).get("position") == hq_pos
                )

                if not already_defending:
                    agent.current_task = self.assign_move_to_task(
                        agent, hq_pos, label="DefendHQ")
                    agent.update_task_state(utils_config.TaskState.ONGOING)
                    logger.log_msg(
                        f"[DEFEND ASSIGN] Peacekeeper {agent.agent_id} assigned to move to HQ.",
                        level=logging.INFO)

        # ========== STRATEGY: Attack Threats ==========
        elif action == "ATTACK_THREATS":
            if self.global_state.get("threat_count", 0) == 0:
                logger.log_msg(
                    f"[HQ EXECUTE] No threats to attack.",
                    level=logging.WARNING)
                self.hq_step_rewards.append(-0.5)  # Penalise ineffective action
                self.current_strategy = None
                return retest_strategy()
            else:
                self.hq_step_rewards.append(+1.0)  # Reward valid threat engagement


        # ========== STRATEGY: Collect Gold ==========
        elif action == "COLLECT_GOLD":
            gold_sources = [
                r for r in self.global_state.get("resources", [])
                if r["type"] == "gold"
            ]
            if not gold_sources:
                logger.log_msg(
                    f"[HQ EXECUTE] No gold resources available.",
                    level=logging.WARNING)
                self.hq_step_rewards.append(-0.5)  # Penalise for no available gold
                self.current_strategy = None
                return retest_strategy()
            else:
                self.hq_step_rewards.append(+1.0)  # Reward valid economic strategy


        # ========== STRATEGY: Collect Food ==========
        elif action == "COLLECT_FOOD":
            food_sources = [
                r for r in self.global_state.get("resources", [])
                if r["type"] == "food"
            ]
            if not food_sources:
                logger.log_msg(
                    f"[HQ EXECUTE] No food resources available.",
                    level=logging.WARNING)
                self.hq_step_rewards.append(-0.5)
                self.current_strategy = None
                return retest_strategy()
            else:
                self.hq_step_rewards.append(+1.0)  # Reward viable food collection


        # ========== STRATEGY: No Priority ==========
        elif action == "NO_PRIORITY":
            logger.log_msg(
                f"[HQ EXECUTE] Faction {self.id} conserving resources.",
                level=logging.INFO)
            self.hq_step_rewards.append(-1.0)  # Discourage doing nothing
            return retest_strategy()


        # ========== Unknown Strategy ==========
        else:
            logger.log_msg(
                f"[HQ EXECUTE] Unknown strategy '{action}'. Retesting...",
                level=logging.WARNING)
            self.hq_step_rewards.append(-1.0)
            return retest_strategy()

        # Set current strategy only if it's valid and successful
        self.current_strategy = action

    def compute_hq_reward(self, victory: bool = False) -> float:
        """
        Computes the HQ reward at the end of an episode.
        This scalar reward trains the HQ strategy network.
        """
        reward = 0

        # Sum shaped rewards collected during the episode
        if hasattr(self, "hq_step_rewards") and self.hq_step_rewards:
            reward += sum(self.hq_step_rewards)

        # Existing static components
        w_gold = 0.01
        w_food = 0.01
        w_agents = 1.0
        w_tasks = 0.2
        w_threats = 0.3
        w_victory = 10.0

        reward += self.gold_balance * w_gold
        reward += self.food_balance * w_food
        reward += len(self.agents) * w_agents

        if hasattr(self, "tasks_completed"):
            reward += self.tasks_completed * w_tasks

        if hasattr(self, "threats_eliminated"):
            reward += self.threats_eliminated * w_threats

        if victory:
            reward += w_victory

        return reward



#    _  _  ___     _______                 ___ _            _                     _  _             _        _      _   _
#   | || |/ _ \   / /_   _|__ __ _ _ __   / __| |_ _ _ __ _| |_ ___ __ _ _  _    /_\| |_ ___ _ __ (_)__    /_\  __| |_(_)___ _ _  ___
#   | __ | (_) | / /  | |/ -_) _` | '  \  \__ \  _| '_/ _` |  _/ -_) _` | || |  / _ \  _/ _ \ '  \| / _|  / _ \/ _|  _| / _ \ ' \(_-<
#   |_||_|\__\_\/_/   |_|\___\__,_|_|_|_| |___/\__|_| \__,_|\__\___\__, |\_, | /_/ \_\__\___/_|_|_|_\__| /_/ \_\__|\__|_\___/_||_/__/
#                                                                  |___/ |__/


    def recruit_agent(self, role: str):
        try:
            cost = utils_config.Gold_Cost_for_Agent

            if self.gold_balance < cost:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[HQ RECRUIT] Faction {self.id} lacks gold to recruit {role}.",
                        level=logging.WARNING)
                return

            # Deduct cost
            self.gold_balance -= cost

            # Create agent
            new_agent = self.create_agent(role)
            if new_agent:
                self.agents.append(new_agent)  # Add to faction
                self.game_manager.agents.append(new_agent)  # <=== Add to global agent list

                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[HQ RECRUIT] Faction {self.id} recruited new {role} — Gold: {self.gold_balance}, Total agents: {len(self.agents)}",
                        level=logging.INFO)
            else:
                logger.log_msg(
                    f"[HQ RECRUIT] Spawn failed: No agent created.",
                    level=logging.WARNING)

        except Exception as e:
            logger.log_msg(
                f"[HQ RECRUIT] Error recruiting {role}: {str(e)}\nTraceback: {traceback.format_exc()}",
                level=logging.ERROR)
            raise


    def create_agent(self, role: str):
        """
        Spawns a new agent instance of the given role using GameManager's spawn_agent method.
        """
        try:
            spawn_x, spawn_y = self.home_base["position"]

            from AGENT.agent_base import Peacekeeper, Gatherer
            role_map = {
                "peacekeeper": Peacekeeper,
                "gatherer": Gatherer
            }

            if role not in role_map:
                raise ValueError(f"Unknown agent role: {role}")

            agent_class = role_map[role]

            agent = self.game_manager.spawn_agent(
                base_x=spawn_x,
                base_y=spawn_y,
                faction=self,
                agent_class=agent_class,
                state_size=utils_config.DEF_AGENT_STATE_SIZE,
                role_actions=utils_config.ROLE_ACTIONS_MAP,
                communication_system=self.communication_system,
                event_manager=self.game_manager.event_manager,
                network_type=self.network_type
            )

            if not agent:
                raise RuntimeError("spawn_agent returned None (no valid location found).")

            # 🛠 Initialise critical fields
            agent.current_action = None
            agent.current_task = None
            agent.current_task_state = utils_config.TaskState.NONE
            agent.mode = "train"
            agent.log_prob = None
            agent.value = None

            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[SPAWN] Created {role} for Faction {self.id} at ({agent.x}, {agent.y}).",
                    level=logging.INFO
                )

            return agent

        except Exception as e:
            logger.log_msg(
                f"[SPAWN ERROR] Failed to create agent: {str(e)}\n{traceback.format_exc()}",
                level=logging.ERROR
            )
            raise



    def is_under_attack(self) -> bool:
        """
        Returns True if enemy agents are within field of view range of the faction HQ.
        """
        hq_x, hq_y = self.home_base["position"]
        detection_radius = utils_config.Agent_Interact_Range

        # Scan known threats in the global state
        for threat in self.global_state.get("threats", []):
            threat_id = threat.get("id")
            if not isinstance(threat_id, utils_config.AgentIDStruc):
                continue

            # Ignore self threats (shouldn’t happen, but just in case)
            if threat_id.faction_id == self.id:
                continue

            tx, ty = threat.get("location", (-999, -999))
            dist_sq = (tx - hq_x) ** 2 + (ty - hq_y) ** 2

            if dist_sq <= detection_radius ** 2:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[HQ THREAT] Faction {self.id} HQ is under threat from enemy agent {threat_id} at {tx, ty}",
                        level=logging.INFO)
                return True

        return False

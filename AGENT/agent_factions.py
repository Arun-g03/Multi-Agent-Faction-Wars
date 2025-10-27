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


class Faction:
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
        mode: str = "train",
    ):
        try:
            # Initialise Faction-specific attributes
            self.name = name
            self.colour = colour
            self.id = id
            self.agents = agents  # List of agents
            self.resource_manager = (
                resource_manager  # Reference to the resource manager
            )
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
                "colour": colour,  # Match faction colour
            }

            self.game_manager = game_manager

            self.global_state = {
                key: None for key in utils_config.STATE_FEATURES_MAP["global_state"]
            }
            # Populate the initial global state
            self.global_state.update(
                {
                    "HQ_health": 100,  # Default HQ health
                    "gold_balance": 0,  # Starting gold
                    "food_balance": 0,  # Starting food
                    "resource_count": 0,  # Total resources count
                    "threat_count": 0,  # Total threats count
                }
            )
            try:
                self.network_type = network_type
                self.network = self.initialise_network(
                    network_type,
                    state_size,
                    action_size,
                    role_size,
                    local_state_size,
                    global_state_size,
                    global_state=self.global_state,
                )

                if self.network is None:
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"[ERROR] Network failed to Initialise for faction {self.id} (Type: {network_type})",
                            level=logging.ERROR,
                        )
                        e
                    raise AttributeError(
                        f"[ERROR] Network failed to Initialise for faction {self.id}: {str(e)}"
                    )
                else:
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"[DEBUG] Faction {self.id}: Successfully initialised {type(self.network).__name__}",
                            level=logging.INFO,
                        )

            except Exception as e:
                import traceback

                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[ERROR] Failed to Initialise network for faction {self.id}: {e}",
                        level=logging.ERROR,
                    )
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(traceback.format_exc(), level=logging.ERROR)

            self.communication_system = CommunicationSystem(self.agents, [self])

            # Populate unvisited cells with land tiles from the terrain grid
            for x in range(len(self.resource_manager.terrain.grid)):
                for y in range(len(self.resource_manager.terrain.grid[0])):
                    cell = self.resource_manager.terrain.grid[x][y]
                    if cell["type"] == "land" and not cell["occupied"]:
                        # Convert to pixel coordinates
                        self.unvisited_cells.add(
                            (x * utils_config.CELL_SIZE, y * utils_config.CELL_SIZE)
                        )

            self.health = 100  # Initial health

            # Dynamically calculate the input size for the critic
            if len(self.agents) > 0:
                critic_state = self.agents[0].get_state(
                    self.resource_manager, self.agents, self
                )
                input_size = len(critic_state)
            else:
                # print("No agents available for dynamic input size calculation. Using fallback input size.")
                input_size = 14  # Default fallback size

            # Now self.hq_network and self.critic are properly initialised via
            # network

            # Initialise the optimiser for the critic (if needed)
            if self.network is None:
                raise RuntimeError(
                    f"[FATAL] Faction {self.id} could not initialise a network. Check input sizes and network_type."
                )
            else:
                self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)

            self.strategy_update_interval = (
                50  # Update strategy every 50 steps instead of 100
            )
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
        global_state,
    ):
        import traceback

        try:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[DEBUG] Faction {self.id}: Attempting to Initialise {network_type}...",
                    level=logging.DEBUG,
                )

            if any(x is None or x == 0 for x in [state_size, action_size]):
                logger.log_msg(
                    f"[ERROR] Invalid network inputs: state_size={state_size}, action_size={action_size}",
                    level=logging.ERROR,
                )
                return None

            if network_type == "PPOModel":
                logger.log_msg(f"[DEBUG] Initialising PPOModel...", level=logging.DEBUG)
                return PPOModel(state_size, action_size)

            elif network_type == "DQNModel":
                logger.log_msg(f"[DEBUG] Initialising DQNModel...", level=logging.DEBUG)
                return DQNModel(state_size, action_size)

            elif network_type == "HQNetwork":
                try:
                    logger.log_msg(
                        f"[DEBUG] Initialising HQNetwork...", level=logging.DEBUG
                    )

                    # Check if we're loading a saved HQ model
                    if (
                        hasattr(self, "load_existing")
                        and self.load_existing
                        and hasattr(self, "models")
                        and "HQ" in self.models
                    ):
                        from NEURAL_NETWORK.Common import (
                            get_hq_input_size_from_checkpoint,
                        )  # if not already imported

                        input_size_from_ckpt = get_hq_input_size_from_checkpoint(
                            self.models["HQ"]
                        )

                        # Deduct known components to infer state_size
                        # Use the passed-in values, don't hardcode!
                        role_size_inferred = (
                            utils_config.ROLE_VECTOR_SIZE
                        )  # or 5 if hardcoded
                        local_state_size_inferred = 5
                        global_state_size_inferred = 5
                        state_size_inferred = input_size_from_ckpt - (
                            role_size_inferred
                            + local_state_size_inferred
                            + global_state_size_inferred
                        )

                        if utils_config.ENABLE_LOGGING:
                            logger.log_msg(
                                f"[HQ LOAD MODE] Using inferred state_size={state_size_inferred} from checkpoint input size={input_size_from_ckpt}",
                                level=logging.INFO,
                            )

                        # Use inferred values for loading
                        return HQ_Network(
                            state_size=state_size_inferred,
                            action_size=action_size,
                            role_size=role_size_inferred,
                            local_state_size=local_state_size_inferred,
                            global_state_size=global_state_size_inferred,
                            device=Training_device,
                            global_state=self.global_state,
                        )
                    else:
                        # Not loading from checkpoint - use passed-in values
                        if utils_config.ENABLE_LOGGING:
                            logger.log_msg(
                                f"[HQ INIT] Using provided parameters: state_size={state_size}, role_size={role_size}, local_state_size={local_state_size}, global_state_size={global_state_size}",
                                level=logging.INFO,
                            )

                        return HQ_Network(
                            state_size=state_size,
                            action_size=action_size,
                            role_size=role_size,
                            local_state_size=local_state_size,
                            global_state_size=global_state_size,
                            device=Training_device,
                            global_state=self.global_state,
                        )
                except Exception as e:
                    logger.log_msg(
                        f"[ERROR] Failed to initialise HQNetwork: {e}",
                        level=logging.ERROR,
                    )
                    raise

        except Exception as e:
            logger.log_msg(
                f"[ERROR] Network initialisation failed for Faction {self.id} (Type: {network_type}): {e}",
                level=logging.ERROR,
            )
            logger.log_msg(traceback.format_exc(), level=logging.ERROR)
            return None

    def update(self, resource_manager, agents, current_step):
        self.clean_global_state()
        self.calculate_territory(resource_manager.terrain)
        self.aggregate_faction_state()
        self.current_step = current_step

        # Only update HQ strategy every N steps
        if (
            current_step % self.strategy_update_interval == 0
            or self.needs_strategy_retest
        ):
            new_strategy = self.choose_HQ_Strategy()
            if new_strategy != self.current_strategy:
                if utils_config.ENABLE_LOGGING:
                    print(
                        f"\033[94mFaction {self.id} has changed HQ from {self.current_strategy} to {new_strategy}.\033[0m\n"
                    )
                self.perform_HQ_Strategy(new_strategy)
            else:
                if utils_config.ENABLE_LOGGING:
                    print(
                        f"\033[93m Faction {self.id} maintained HQ strategy: {self.current_strategy}\033[0m"
                    )
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
            print(f"Warning: Agent {agent.role} not found in faction {self.id}.")

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
            "total_agents",
        ]

        for key in required_keys:
            if key not in self.global_state:
                self.global_state[key] = 0  # Default missing values to zero

        #  Ensure HQ position exists before using it
        if "position" not in self.home_base or self.home_base["position"] == (0, 0):
            print(
                f"[WARNING] HQ position for Faction {self.id} is missing! Assigning default location."
            )
            self.home_base["position"] = (
                random.randint(0, 100),
                random.randint(0, 100),
            )  # Assign a random position

        hq_x, hq_y = self.home_base["position"]

        # Ensure `nearest_threat` and `nearest_resource` are structured
        # correctly
        self.global_state["nearest_threat"] = self.global_state.get(
            "nearest_threat", {"location": (-1, -1)}
        )
        self.global_state["nearest_resource"] = self.global_state.get(
            "nearest_resource", {"location": (-1, -1)}
        )

        #  Fetch agents correctly
        all_agents = self.game_manager.agents  # Game-wide agents
        enemy_agents = [agent for agent in all_agents if agent.faction != self]

        #  Compute agent-related metrics
        self.global_state["friendly_agent_count"] = len(
            self.agents
        )  # Use faction-level agents
        self.global_state["enemy_agent_count"] = len(
            enemy_agents
        )  # Use game-wide agents
        self.global_state["total_agents"] = len(all_agents)  # Use game-wide agents

        #  Compute agent density near HQ
        nearby_agents = [
            agent
            for agent in self.agents
            if ((agent.x - hq_x) ** 2 + (agent.y - hq_y) ** 2) ** 0.5 < 50
        ]
        self.global_state["agent_density"] = len(nearby_agents)

        # Calculate nearest resource relative to HQ
        if self.global_state["resources"]:
            nearest_res = min(
                self.global_state["resources"],
                key=lambda r: (
                    (r["location"][0] - hq_x) ** 2 + (r["location"][1] - hq_y) ** 2
                )
                ** 0.5,
            )
            distance = (
                (nearest_res["location"][0] - hq_x) ** 2
                + (nearest_res["location"][1] - hq_y) ** 2
            ) ** 0.5
            self.global_state["nearest_resource"] = {
                "location": nearest_res["location"],
                "distance": distance,
                "type": nearest_res.get("type", "unknown"),
            }
        else:
            self.global_state["nearest_resource"] = {
                "location": (-1, -1),
                "distance": float("inf"),
                "type": None,
            }

        # Calculate nearest threat relative to HQ
        if self.global_state["threats"]:
            nearest_threat = min(
                self.global_state["threats"],
                key=lambda t: (
                    (t["location"][0] - hq_x) ** 2 + (t["location"][1] - hq_y) ** 2
                )
                ** 0.5,
            )
            distance = (
                (nearest_threat["location"][0] - hq_x) ** 2
                + (nearest_threat["location"][1] - hq_y) ** 2
            ) ** 0.5
            self.global_state["nearest_threat"] = {
                "location": nearest_threat["location"],
                "distance": distance,
                "type": nearest_threat.get("type", "unknown"),
            }
        else:
            self.global_state["nearest_threat"] = {
                "location": (-1, -1),
                "distance": float("inf"),
                "type": None,
            }

        #  Ensure `agent_states` are properly formatted
        self.global_state["agent_states"] = [
            agent.get_state(self.resource_manager, self.agents, self, self.global_state)
            for agent in self.agents
        ]

        #  Debug Log to verify correct agent count
        # if utils_config.ENABLE_LOGGING: logger.log_msg(f"[DEBUG] Faction {self.id} State: {self.global_state}")

        return self.global_state

    def receive_report(self, report):
        """Process reports received from agents."""
        if "type" not in report or "data" not in report:
            logger.warning(
                f"Invalid report format received by Faction {self.id}: {report}"
            )
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
                (
                    t
                    for t in self.global_state["threats"]
                    if str(t.get("id")) == str(threat_id)
                ),
                None,
            )

            if existing_threat:
                if existing_threat["location"] != location:
                    existing_threat["location"] = location
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"Faction {self.id} updated threat ID {threat_id} to location {location}."
                        )

            else:
                self.global_state["threats"].append(data)
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"Faction {self.id} added new threat: {data['type']} ID {threat_id} at {location}."
                    )

        elif report_type == "resource":
            # Extract relevant data from the resource object
            if (
                hasattr(data, "grid_x")
                and hasattr(data, "grid_y")
                and hasattr(data, "__class__")
            ):
                resource_data = {
                    "location": (data.grid_x, data.grid_y),
                    "type": data.__class__.__name__,
                }

                # Check if the resource already exists
                existing_resource = next(
                    (
                        res
                        for res in self.global_state["resources"]
                        if res["location"] == resource_data["location"]
                    ),
                    None,
                )

                if existing_resource:
                    pass  # Do nothing if the resource already exists
                else:
                    self.global_state["resources"].append(resource_data)
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"Faction {self.id} added resource: {resource_data}."
                        )
            else:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"Invalid resource object in report for Faction {self.id}: {data}"
                    )

        else:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"Unknown report type '{report_type}' received by Faction {self.id}: {report}"
                )

    def provide_state(self):
        """
        Provide the faction's global state to a requester.
        """
        return self.global_state

    #     ____ _                    _   _                  _       _           _       _        _
    #    / ___| | ___  __ _ _ __   | |_| |__   ___    __ _| | ___ | |__   __ _| |  ___| |_ __ _| |_ ___
    #   | |   | |/ _ \/ _` | '_ \  | __| '_ \ / _ \  / _` | |/ _ \| '_ \ / _` | | / __| __/ _` | __/ _ \
    #   | |___| |  __/ (_| | | | | | |_| | | |  __/ | (_| | | (_) | |_) | (_) | |  | |_\__ \ |  _| | | (_) | | | | | | | (_| | (_| |  __/ | | | |_\__ \
    #    \____|_|\___|\__,_|_| |_|  \__|_| |_|\___|  \__, |_|\___/|_.__/ \___/|_| |___/\__\__,_|\__\___|
    #                                                |___/

    def clean_global_state(self):
        """
        Clean outdated entries in the global state and ensure required features exist.
        This includes validating resources and threats against the actual environment.
        """
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[DEBUG] Cleaning global state for Faction {self.id} BEFORE reset: {self.global_state}",
                level=logging.DEBUG,
            )

        # ========== RESOURCE CLEANUP ==========
        original_resources = len(self.global_state.get("resources", []))
        valid_resources = []

        for res in self.global_state.get("resources", []):
            loc = res.get("location")
            if not loc:
                continue

            match = next(
                (
                    r
                    for r in self.resource_manager.resources
                    if hasattr(r, "grid_x")
                    and hasattr(r, "grid_y")
                    and (r.grid_x, r.grid_y) == loc
                    and not r.is_depleted()
                ),
                None,
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
                    agent
                    for agent in self.game_manager.agents
                    if getattr(agent, "agent_id", None) == tid
                    and getattr(agent, "Health", 1) > 0
                )
                if is_alive:
                    valid_threats.append(threat)

        self.global_state["threats"] = valid_threats
        self.global_state["threat_count"] = len(valid_threats)

        # ========== NEAREST PLACEHOLDER FIELDS ==========
        self.global_state["nearest_threat"] = self.global_state.get(
            "nearest_threat", {"location": (-1, -1)}
        )
        self.global_state["nearest_resource"] = self.global_state.get(
            "nearest_resource", {"location": (-1, -1)}
        )

        # ========== AGENT STATS ==========
        self.global_state["friendly_agent_count"] = len(self.agents)

        enemy_agents = [
            agent for agent in self.game_manager.agents if agent.faction != self
        ]
        self.global_state["enemy_agent_count"] = len(enemy_agents)

        hq_x, hq_y = self.home_base["position"]
        nearby_agents = [
            agent
            for agent in self.agents
            if ((agent.x - hq_x) ** 2 + (agent.y - hq_y) ** 2) ** 0.5 < 50
        ]
        self.global_state["agent_density"] = len(nearby_agents)

        self.global_state["total_agents"] = len(self.game_manager.agents)

        # ========== BASE STATE FIELDS ==========
        self.global_state["HQ_health"] = self.global_state.get("HQ_health", 100)
        self.global_state["gold_balance"] = self.global_state.get("gold_balance", 0)
        self.global_state["food_balance"] = self.global_state.get("food_balance", 0)

        # ========== LOGGING ==========
        removed_resources = original_resources - len(valid_resources)
        removed_threats = original_threats - len(valid_threats)
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[CLEAN] Pruned {removed_resources} resources, {removed_threats} threats.",
                level=logging.DEBUG,
            )

            state = self.global_state
            resource_strs = [r.get("location", "?") for r in state.get("resources", [])]
            threat_strs = [
                f"{t.get('type', '?')}@{t.get('location', '?')}"
                for t in state.get("threats", [])
            ]

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
                f"  Food           : {state.get('food_balance', 0)}\n"
            )
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
            logger.log_msg(
                f"[HQ] Faction {self.id} assigning high-level tasks...",
                level=logging.INFO,
            )

        for agent in self.agents:
            # === 1. Check current task state ===
            task_completed = False
            task_exists = agent.current_task is not None

            if task_exists and agent.current_task_state in [
                utils_config.TaskState.SUCCESS,
                utils_config.TaskState.FAILURE,
            ]:
                task_completed = True
            elif task_exists and agent.current_task_state in [
                utils_config.TaskState.ONGOING,
                utils_config.TaskState.PENDING,
            ]:
                # Skip busy agents
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[DEBUG] Skipping {agent.role} {agent.agent_id} - busy with task in state: {agent.current_task_state}",
                        level=logging.DEBUG,
                    )
                continue

            # === 2. Clear completed tasks if needed ===
            if task_completed and task_exists:
                self.complete_task(
                    agent.current_task.get("id"), agent, agent.current_task_state
                )
                agent.current_task = None
                agent.update_task_state(utils_config.TaskState.NONE)

            # === 3. Only assign new tasks to agents who are idle ===
            agent_idle = not task_exists or task_completed

            if not agent_idle:
                continue  # Agent still has an active task

            # === 4. Assign a new task ===
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[DEBUG] Assigning task to {agent.role} {agent.agent_id} (idle: {agent_idle})",
                    level=logging.DEBUG,
                )
            task = self.assign_task(agent)
            if task:
                agent.current_task = task
                agent.update_task_state(utils_config.TaskState.ONGOING)

                self.task_counter += 1
                task_counter_id = self.task_counter

                self.assigned_tasks[task_counter_id] = {
                    "type": task["type"],  # task type: 'explore', 'mine', etc.
                    "task_id": task["id"],  # e.g., "Explore-(19,17)"
                    "start_step": self.current_step,  # when assigned
                    "end_step": None,  # filled when completed
                    "agents": {str(agent.agent_id): utils_config.TaskState.ONGOING},
                }

                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[TASK ASSIGNED] {agent.agent_id} => {task['type']} at {task['target'].get('position')}",
                        level=logging.INFO,
                    )
                    logger.log_msg(
                        f"[DEBUG] {agent.agent_id} has task: {agent.current_task}",
                        level=logging.DEBUG,
                    )
            else:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[DEBUG] No task assigned to {agent.role} {agent.agent_id}",
                        level=logging.DEBUG,
                    )

    """

    #       _            _               _            _          _                                 _
    #      / \\   ___ ___(_) __ _ _ __   | |_ __ _ ___| | _____  | |_ ___     __ _  __ _  ___ _ __ | |_ ___
    #     / _ \\ / __/ __| |/ _` | '_ \\  | __/ _` / __| |/ / __| | __/ _ \\   / _` |/ _` |/ _ \\ '_ \\| __/ __|
    #    / ___ \\__ \\__ \\ | (_| | | | | | || (_| \\__ \\   <\\__ \\ | || (_) | | (_| | (_| |  __/ | | | |_\\__ \
    #   /_/   \\_\\___/___/_|\\__, |_| |_|  \\__\\__,_|___/_|\\_\\   \\_/\\_/ \\___|_|\\__, |_| |_|\\__|___/
    #                      |___/                                                  |___/
    ========================================================================================================
    """

    # This is where tasks are created

    def assign_task(self, agent) -> Optional[dict]:
        role = getattr(agent, "role", None)
        strategy = self.current_strategy or "NO_PRIORITY"

        # Determine threats
        threats = [
            t
            for t in self.global_state.get("threats", [])
            if t["id"].faction_id != self.id
        ]
        has_threats = len(threats) > 0

        if role not in ["gatherer", "peacekeeper"]:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[WARN] Unknown role {agent.agent_id}: {role}",
                    level=logging.WARNING,
                )
            return None

        if role == "gatherer":
            current = agent.current_task
            current_type = current["type"] if current else "none"

            if isinstance(current_type, str):
                current_type_id = utils_config.TASK_TYPE_MAPPING.get(
                    current_type, utils_config.TASK_TYPE_MAPPING["none"]
                )
            else:
                current_type_id = current_type

            is_resource_task = (
                current_type_id == utils_config.TASK_TYPE_MAPPING["gather"]
            )

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
                options = list(
                    filter(
                        None,
                        [
                            self.assign_forage_task(agent),
                            self.assign_mining_task(agent),
                        ],
                    )
                )
                if options:
                    return random.choice(options)

            # Default to exploration
            return self.assign_explore_task(agent)

        elif role == "peacekeeper":
            current = agent.current_task
            current_type = current["type"] if current else "none"

            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[DEBUG] Peacekeeper {agent.agent_id} - Current task: {current}, Strategy: {strategy}, Has threats: {has_threats}",
                    level=logging.DEBUG,
                )

            if isinstance(current_type, str):
                current_type_id = utils_config.TASK_TYPE_MAPPING.get(
                    current_type, utils_config.TASK_TYPE_MAPPING["none"]
                )
            else:
                current_type_id = current_type

            if strategy == "ATTACK_THREATS" and has_threats:
                threat = min(
                    threats, key=lambda t: self.calculate_threat_weight(agent, t)
                )
                already_eliminating = (
                    current_type_id == utils_config.TASK_TYPE_MAPPING["eliminate"]
                )

                if not already_eliminating:
                    if current:
                        agent.current_task = None
                        agent.update_task_state(utils_config.TaskState.NONE)
                    return self.assign_eliminate_task(
                        agent, threat["id"], threat["type"], threat["location"]
                    )

            elif strategy == "DEFEND_HQ":
                hq_position = self.home_base["position"]
                # Always convert pixel HQ position to grid coordinates for movement
                hq_position = (
                    int(hq_position[0] // utils_config.CELL_SIZE),
                    int(hq_position[1] // utils_config.CELL_SIZE),
                )

                # Check if agent is already at HQ
                agent_grid_x = int(agent.x // utils_config.CELL_SIZE)
                agent_grid_y = int(agent.y // utils_config.CELL_SIZE)
                is_at_hq = (
                    agent_grid_x == hq_position[0] and agent_grid_y == hq_position[1]
                )

                already_defending = (
                    current_type_id == utils_config.TASK_TYPE_MAPPING["move_to"]
                    and current
                    and current.get("target", {}).get("position") == hq_position
                )

                # If already at HQ or already defending, don't assign new task
                if is_at_hq or already_defending:
                    return current

                if current:
                    agent.current_task = None
                    agent.update_task_state(utils_config.TaskState.NONE)
                return self.assign_move_to_task(agent, hq_position, label="DefendHQ")

            elif strategy in ["COLLECT_GOLD", "COLLECT_FOOD", "NO_PRIORITY"]:
                # When HQ wants to collect resources, peacekeepers should secure the area
                # by patrolling near resources or defending the HQ

                # Check if peacekeeper has an ongoing patrol or defend task
                is_patrol_or_defend = (
                    current_type_id
                    in [
                        utils_config.TASK_TYPE_MAPPING["move_to"],
                        utils_config.TASK_TYPE_MAPPING["eliminate"],
                    ]
                    and current
                    and agent.current_task_state == utils_config.TaskState.ONGOING
                )

                # Keep ongoing patrol/defend tasks
                if is_patrol_or_defend:
                    return current

                # Otherwise, assign to patrol/defend near resource areas or HQ
                hq_position = self.home_base["position"]
                hq_position = (
                    int(hq_position[0] // utils_config.CELL_SIZE),
                    int(hq_position[1] // utils_config.CELL_SIZE),
                )

                # Check if already at HQ
                agent_grid_x = int(agent.x // utils_config.CELL_SIZE)
                agent_grid_y = int(agent.y // utils_config.CELL_SIZE)
                is_at_hq = (
                    agent_grid_x == hq_position[0] and agent_grid_y == hq_position[1]
                )

                if not is_at_hq:
                    if current:
                        agent.current_task = None
                        agent.update_task_state(utils_config.TaskState.NONE)
                    return self.assign_move_to_task(
                        agent, hq_position, label="DefendHQ"
                    )
                else:
                    # Already at HQ, keep current task if valid
                    if (
                        current
                        and agent.current_task_state == utils_config.TaskState.ONGOING
                    ):
                        return current
                    # Otherwise explore nearby
                    return self.assign_explore_task(agent)

            return current  # fallback

        else:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[WARN] - assign_task - Unknown role {agent.agent_id}: {role}",
                    level=logging.WARNING,
                )
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
                    level=logging.WARNING,
                )
            return None

        # Format task_id consistently like mining tasks
        task_id = label or f"MoveTo-{position}"
        target = {"position": position, "type": "Location"}

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"Created task: {task_id} for agent {agent.agent_id}",
                level=logging.INFO,
            )

        task = utils_config.create_task(self, "move_to", target, task_id)

        # Mark the task as ongoing
        agent.current_task = task
        agent.update_task_state(utils_config.TaskState.ONGOING)

        return task

    def assign_eliminate_task(
        self, agent, target_id, target_type, position
    ) -> Optional[dict]:
        try:
            task_id = f"Eliminate-{target_id}"
            target = {"id": target_id, "type": target_type, "position": position}
            logger.log_msg(
                f"Created task: {task_id} for agent {agent.agent_id}",
                level=logging.INFO,
            )
            return utils_config.create_task(self, "eliminate", target, task_id)
        except Exception as e:
            logger.log_msg(f"Failed to create eliminate task: {e}", level=logging.ERROR)
            return None

    def assign_forage_task(self, agent) -> Optional[dict]:
        resources = [
            r
            for r in self.global_state.get("resources", [])
            if r["type"] == "AppleTree"
        ]
        if not resources:
            return None
        nearest = min(resources, key=lambda r: self.calculate_resource_weight(agent, r))
        task_id = f"Forage-{nearest['location']}"
        target = {"position": nearest["location"], "type": "AppleTree"}
        logger.log_msg(
            f"Created task: {task_id} for agent {agent.agent_id}", level=logging.INFO
        )
        return utils_config.create_task(self, "gather", target, task_id)

    def assign_mining_task(self, agent) -> Optional[dict]:
        resources = [
            r for r in self.global_state.get("resources", []) if r["type"] == "GoldLump"
        ]
        if not resources:
            return None
        nearest = min(resources, key=lambda r: self.calculate_resource_weight(agent, r))
        task_id = f"Mine-{nearest['location']}"
        target = {"position": nearest["location"], "type": "GoldLump"}
        logger.log_msg(
            f"Created task: {task_id} for agent {agent.agent_id}", level=logging.INFO
        )
        return utils_config.create_task(self, "gather", target, task_id)

    def assign_explore_task(self, agent):
        terrain = self.resource_manager.terrain

        self.calculate_territory(terrain)  # track current control
        logger.log_msg(
            f"[EXPLORE] Faction {self.id} controls {self.territory_count} tiles.",
            level=logging.INFO,
        )

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

        logger.log_msg(
            f"[EXPLORE] Found {len(unexplored_cells)} unexplored cells for agent {agent.agent_id}",
            level=logging.INFO,
        )

        if unexplored_cells:
            cell_x, cell_y = random.choice(unexplored_cells)
            task_string_id = f"Explore-({cell_x}, {cell_y})"
            target = {"position": (cell_x, cell_y), "type": "Explore"}

            logger.log_msg(
                f"Created task: {task_string_id} for agent {agent.agent_id}",
                level=logging.INFO,
            )

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
                "agents": {str(agent.agent_id): utils_config.TaskState.ONGOING},
            }

            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[DEBUG] Explore task created for {agent.role} {agent.agent_id}: {task}",
                    level=logging.DEBUG,
                )

            return task

        logger.log_msg(
            f"[EXPLORE] No valid unexplored cells left for agent {agent.agent_id}",
            level=logging.WARNING,
        )
        return None

    """

    #     ____      _            _       _         _            _                   _       _     _
    #    / ___|__ _| | ___ _   _| | __ _| |_ ___  | |_ __ _ ___| | __ __      _____(_) __ _| |__ | |_ ___
    #   | |   / _` | |/ __| | | | |/ _` | __/ _ \\ | __/ _` / __| |/ / \\ \\ /\\ / / _` / __|  / __/ _ \| '_ ` _ \| '_ \| |/ _ \ __/ _ \/ _` |
    #   | |__| (_| | | (__| |_| | | (_| | ||  __/ | || (_| \__ \   <   \\ V  V / (_| \__ \ | (_| (_) | | | | | | |_) | |  __/ ||  __/ (_| |
    #    \___|_| |_|\___|\__,_|_|\\__,_|_|_|_| |___/\__|_| \__,_|\__\___\__, |\_, | /_/ \_\__\___/_|_|_|_\__| /_/ \_\__|\__|_\___/_||_/__/
    #                                                                  |___/ |__/
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
        distance = (
            (threat["location"][0] - agent.x) ** 2
            + (threat["location"][1] - agent.y) ** 2
        ) ** 0.5
        type_weight = 1  # Default weight for general threats
        if threat["type"] == "Faction HQ":
            type_weight = 0.5  # Higher priority for enemy HQs
        weight = distance * type_weight
        return weight

    #         _               _      _   _            _            _                                                    _      _           _
    #     ___| |__   ___  ___| | __ | |_| |__   ___  | |_ __ _ ___| | __ __      ____ _ ___    ___ ___  _ __ ___  _ __ | | ___| |_ ___  __| |
    #    / __| '_ \ / _ \/ __| |/ / | __| '_ \ / _ \ | __/ _` / __| |/ / \ \ /\ / / _` / __|  / __/ _ \| '_ ` _ \| '_ \| |/ _ \ __/ _ \/ _` |
    #   | (__| | | |  __/ (__|   <  | |_| | | |  __/ | || (_| \__ \   <   \ V  V / (_| \__ \ | (_| (_) | | | | | | |_) | |  __/ ||  __/ (_| |
    #    \___|_| |_|\___|\__,_|_|\\__,_|_|_|_| |___/\__|_| \__,_|\__\___\__, |\_, | /_/ \_\__\___/_|_|_|_\__| /_/ \_\__|\__|_\___/_||_/__/
    #                                                                  |___/ |__/

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
                            level=logging.INFO,
                        )

                    #  Check if ALL agents finished
                    if all(
                        state
                        in [
                            utils_config.TaskState.SUCCESS,
                            utils_config.TaskState.FAILURE,
                        ]
                        for state in task_data["agents"].values()
                    ):
                        if task_data["end_step"] is None:
                            task_data["end_step"] = self.current_step
                            logger.log_msg(
                                f"[TASK COMPLETE] Task {task_id_str} fully completed at step {self.current_step}.",
                                level=logging.INFO,
                            )
                else:
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"[WARN] Agent {agent_id_str} not found in task {task_id_str} agent list.",
                            level=logging.WARNING,
                        )
                break
        else:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[WARN] Task {task_id_str} not found in assigned_tasks!",
                    level=logging.WARNING,
                )

    def update_tasks(self, agents):
        for agent in agents:
            # Skip agents from other factions
            if agent.faction != self:
                continue

            # If task completed, clear it
            if agent.current_task_state in [
                utils_config.TaskState.SUCCESS,
                utils_config.TaskState.FAILURE,
            ]:
                if agent.current_task:
                    self.complete_task(
                        agent.current_task["id"], agent, agent.current_task_state
                    )
                    agent.current_task = None
                    agent.update_task_state(utils_config.TaskState.NONE)

    def calculate_territory(self, terrain):
        """Calculate the number of cells owned by this faction."""
        self.territory_count = sum(
            1 for row in terrain.grid for cell in row if cell["faction"] == self.id
        )

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
                level=logging.INFO,
            )

        strategy = "NO_PRIORITY"  # Default fallback
        previous_strategy = self.current_strategy

        if hasattr(self, "network") and self.network:
            # Use enhanced state for better strategy selection
            enhanced_state = self.get_enhanced_global_state()
            predicted = self.network.predict_strategy(enhanced_state)

            if predicted in utils_config.HQ_STRATEGY_OPTIONS:
                strategy = predicted
                if strategy != previous_strategy:
                    logger.log_msg(
                        f"[HQ STRATEGY] Faction {self.id} network picked different strategy: {strategy}",
                        level=logging.INFO,
                    )
                else:
                    logger.log_msg(
                        f"[HQ STRATEGY] Faction {self.id} network continued with strategy: {strategy}",
                        level=logging.INFO,
                    )
            else:
                logger.log_msg(
                    f"[HQ STRATEGY] Invalid strategy returned: {predicted}. Defaulting to NO_PRIORITY.",
                    level=logging.WARNING,
                )
        else:
            logger.log_msg(
                f"[HQ STRATEGY] No HQ network found. Falling back to NO_PRIORITY.",
                level=logging.WARNING,
            )

        # Ensure we always have a valid strategy
        if strategy not in utils_config.HQ_STRATEGY_OPTIONS:
            strategy = "NO_PRIORITY"
            logger.log_msg(
                f"[HQ STRATEGY] Invalid strategy detected, defaulting to NO_PRIORITY.",
                level=logging.WARNING,
            )

        # Log strategy selection reasoning for debugging
        if utils_config.ENABLE_LOGGING:
            if strategy.startswith("SWAP_TO_"):
                composition = self.get_faction_composition()
                if composition and composition.get("suggestions"):
                    best_suggestion = composition["suggestions"][0]
                    logger.log_msg(
                        f"[HQ STRATEGY] Swap strategy {strategy} selected based on: {best_suggestion['reason']}",
                        level=logging.INFO,
                    )

            # Add debug logging for initial strategy assignment
            if previous_strategy is None:
                logger.log_msg(
                    f"[DEBUG] Faction {self.id} initial strategy set to: {strategy}",
                    level=logging.DEBUG,
                )

        self.current_strategy = strategy
        self.strategy_history.append(strategy)  # Track it
        return strategy

    def perform_HQ_Strategy(self, action):
        """
        HQ executes the chosen strategic action if valid.
        If invalid, it re-evaluates strategy using the HQ network.
        """

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[HQ EXECUTE] Faction {self.id} attempting strategy: {action}",
                level=logging.INFO,
            )

        def retest_strategy():
            new_action = self.choose_HQ_Strategy()
            if new_action != action:
                logger.log_msg(
                    f"[HQ RETEST] Strategy '{action}' invalid. Retesting and switching to '{new_action}'",
                    level=logging.WARNING,
                )
                self.perform_HQ_Strategy(new_action)
            logger.log_msg(
                f"[HQ EXECUTE] Faction {self.id} executing updated strategy: {new_action}",
                level=logging.INFO,
            )

        # ========== STRATEGY: Recruit Peacekeeper ==========
        if action == "RECRUIT_PEACEKEEPER":
            Agent_cost = utils_config.Gold_Cost_for_Agent
            if (
                self.gold_balance >= Agent_cost
                and len(self.agents) < utils_config.MAX_AGENTS
            ):
                new_agent = self.recruit_agent("peacekeeper")
                if new_agent:
                    print(f"Faction {self.id} bought a Peacekeeper")
                    self.hq_step_rewards.append(+1.0)
                else:
                    logger.log_msg(
                        f"[HQ EXECUTE] Failed to recruit peacekeeper: spawn failed.",
                        level=logging.WARNING,
                    )
                    self.current_strategy = None
                    self.hq_step_rewards.append(-0.5)
                    return retest_strategy()
            else:
                reason = (
                    "Not enough gold"
                    if self.gold_balance < Agent_cost
                    else "Agent limit reached"
                )
                logger.log_msg(
                    f"[HQ EXECUTE] Cannot recruit peacekeeper: {reason}.",
                    level=logging.WARNING,
                )
                self.current_strategy = None
                self.hq_step_rewards.append(-0.5)
                return retest_strategy()

        # ========== STRATEGY: Recruit Gatherer ==========
        elif action == "RECRUIT_GATHERER":
            Agent_cost = utils_config.Gold_Cost_for_Agent
            if (
                self.gold_balance >= Agent_cost
                and len(self.agents) < utils_config.MAX_AGENTS
            ):
                new_agent = self.recruit_agent("gatherer")
                if new_agent:
                    print(f"Faction {self.id} bought a Gatherer")
                    self.hq_step_rewards.append(+1.0)
                else:
                    logger.log_msg(
                        f"[HQ EXECUTE] Failed to recruit gatherer: spawn failed.",
                        level=logging.WARNING,
                    )
                    self.current_strategy = None
                    self.hq_step_rewards.append(-0.5)
                    return retest_strategy()
            else:
                reason = (
                    "Not enough gold"
                    if self.gold_balance < Agent_cost
                    else "Agent limit reached"
                )
                logger.log_msg(
                    f"[HQ EXECUTE] Cannot recruit gatherer: {reason}.",
                    level=logging.WARNING,
                )
                self.current_strategy = None
                self.hq_step_rewards.append(-0.5)
                return retest_strategy()

        # ========== STRATEGY: Swap to Gatherer ==========
        elif action == "SWAP_TO_GATHERER":
            if len(self.agents) == 0:
                logger.log_msg(
                    f"[HQ EXECUTE] Cannot swap to gatherer: No agents available.",
                    level=logging.WARNING,
                )
                self.current_strategy = None
                self.hq_step_rewards.append(-0.5)
                return retest_strategy()

            # Evaluate which agents to swap
            candidates = self.evaluate_agent_swap_candidates("gatherer")

            if candidates:
                best_candidate, score = candidates[0]
                if self.swap_agent_role(best_candidate, "gatherer"):
                    print(
                        f"Faction {self.id} swapped {best_candidate.role} to Gatherer"
                    )
                    self.hq_step_rewards.append(+1.0)
                    # Log swap statistics for monitoring
                    self.log_swap_statistics()
                else:
                    logger.log_msg(
                        f"[HQ EXECUTE] Failed to swap agent to gatherer.",
                        level=logging.WARNING,
                    )
                    self.current_strategy = None
                    self.hq_step_rewards.append(-0.5)
                    return retest_strategy()
            else:
                logger.log_msg(
                    f"[HQ EXECUTE] No suitable candidates for swapping to gatherer.",
                    level=logging.WARNING,
                )
                self.current_strategy = None
                self.hq_step_rewards.append(-0.5)
                return retest_strategy()

        # ========== STRATEGY: Swap to Peacekeeper ==========
        elif action == "SWAP_TO_PEACEKEEPER":
            if len(self.agents) == 0:
                logger.log_msg(
                    f"[HQ EXECUTE] Cannot swap to peacekeeper: No agents available.",
                    level=logging.WARNING,
                )
                self.current_strategy = None
                self.hq_step_rewards.append(-0.5)
                return retest_strategy()

            # Evaluate which agents to swap
            candidates = self.evaluate_agent_swap_candidates("peacekeeper")

            if candidates:
                best_candidate, score = candidates[0]
                if self.swap_agent_role(best_candidate, "peacekeeper"):
                    print(
                        f"Faction {self.id} swapped {best_candidate.role} to Peacekeeper"
                    )
                    self.hq_step_rewards.append(+1.0)
                    # Log swap statistics for monitoring
                    self.log_swap_statistics()
                else:
                    logger.log_msg(
                        f"[HQ EXECUTE] Failed to swap agent to peacekeeper.",
                        level=logging.WARNING,
                    )
                    self.current_strategy = None
                    self.hq_step_rewards.append(-0.5)
                    return retest_strategy()
            else:
                logger.log_msg(
                    f"[HQ EXECUTE] No suitable candidates for swapping to peacekeeper.",
                    level=logging.WARNING,
                )
                self.current_strategy = None
                self.hq_step_rewards.append(-0.5)
                return retest_strategy()

        # ========== STRATEGY: Defend HQ ==========
        elif action == "DEFEND_HQ":
            DEFENSE_RADIUS = 100  # pixels
            DEFENSE_RADIUS_SQ = DEFENSE_RADIUS**2

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
                    level=logging.WARNING,
                )
                self.hq_step_rewards.append(-0.5)
                return retest_strategy()

            # Strategy is valid — assign peacekeepers to move to HQ
            self.defensive_position = hq_pos
            logger.log_msg(
                f"[HQ STRATEGY] Nearby threat detected. Assigning peacekeepers to defend HQ at {hq_pos}.",
                level=logging.INFO,
            )

            for agent in self.agents:
                if agent.role != "peacekeeper":
                    continue

                current = agent.current_task
                already_defending = (
                    current
                    and current.get("type") == "move_to"
                    and current.get("target", {}).get("position") == hq_pos
                )

                if not already_defending:
                    agent.current_task = self.assign_move_to_task(
                        agent, hq_pos, label="DefendHQ"
                    )
                    agent.update_task_state(utils_config.TaskState.ONGOING)
                    logger.log_msg(
                        f"[DEFEND ASSIGN] Peacekeeper {agent.agent_id} assigned to move to HQ.",
                        level=logging.INFO,
                    )

        # ========== STRATEGY: Attack Threats ==========
        elif action == "ATTACK_THREATS":
            if self.global_state.get("threat_count", 0) == 0:
                logger.log_msg(
                    f"[HQ EXECUTE] No threats to attack.", level=logging.WARNING
                )
                self.hq_step_rewards.append(-0.5)  # Penalise ineffective action
                self.current_strategy = None
                return retest_strategy()
            else:
                self.hq_step_rewards.append(+1.0)  # Reward valid threat engagement

        # ========== STRATEGY: Collect Gold ==========
        elif action == "COLLECT_GOLD":
            gold_sources = [
                r for r in self.global_state.get("resources", []) if r["type"] == "gold"
            ]
            if not gold_sources:
                logger.log_msg(
                    f"[HQ EXECUTE] No gold resources available.", level=logging.WARNING
                )
                self.hq_step_rewards.append(-0.5)  # Penalise for no available gold
                self.current_strategy = None
                return retest_strategy()
            else:
                self.hq_step_rewards.append(+1.0)  # Reward valid economic strategy

        # ========== STRATEGY: Collect Food ==========
        elif action == "COLLECT_FOOD":
            food_sources = [
                r for r in self.global_state.get("resources", []) if r["type"] == "food"
            ]
            if not food_sources:
                logger.log_msg(
                    f"[HQ EXECUTE] No food resources available.", level=logging.WARNING
                )
                self.hq_step_rewards.append(-0.5)
                self.current_strategy = None
                return retest_strategy()
            else:
                self.hq_step_rewards.append(+1.0)  # Reward viable food collection

        # ========== STRATEGY: No Priority ==========
        elif action == "NO_PRIORITY":
            logger.log_msg(
                f"[HQ EXECUTE] Faction {self.id} conserving resources and waiting for opportunities.",
                level=logging.INFO,
            )
            # This is a valid strategy - no need to retest
            self.hq_step_rewards.append(0.0)  # Neutral reward for waiting

        # ========== Unknown Strategy ==========
        else:
            logger.log_msg(
                f"[HQ EXECUTE] Unknown strategy '{action}'. Retesting...",
                level=logging.WARNING,
            )
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
        """
        Recruit a new agent of the specified role.
        Returns the created agent if successful, None if failed.
        Gold is only deducted AFTER successful creation.
        """
        try:
            cost = utils_config.Gold_Cost_for_Agent

            if self.gold_balance < cost:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[HQ RECRUIT] Faction {self.id} lacks gold to recruit {role}.",
                        level=logging.WARNING,
                    )
                return None

            if len(self.agents) >= utils_config.MAX_AGENTS:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[HQ RECRUIT] Faction {self.id} cannot recruit: max agents reached ({len(self.agents)}/{utils_config.MAX_AGENTS}).",
                        level=logging.WARNING,
                    )
                return None

            # Try to create the agent first before deducting gold
            new_agent = self.create_agent(role)
            if new_agent:
                # Success! Deduct the cost and add to lists
                self.gold_balance -= cost
                self.agents.append(new_agent)  # Add to faction
                self.game_manager.agents.append(
                    new_agent
                )  # <=== Add to global agent list

                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[HQ RECRUIT] Faction {self.id} recruited new {role} — Gold: {self.gold_balance}, Total agents: {len(self.agents)}",
                        level=logging.INFO,
                    )

                return new_agent
            else:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[HQ RECRUIT] Spawn failed: No valid location found for {role}.",
                        level=logging.WARNING,
                    )
                return None

        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[HQ RECRUIT] Error recruiting {role}: {str(e)}\nTraceback: {traceback.format_exc()}",
                    level=logging.ERROR,
                )
            return None

    def create_agent(self, role: str):
        """
        Spawns a new agent instance of the given role using GameManager's spawn_agent method.
        """
        try:
            spawn_x, spawn_y = self.home_base["position"]

            from AGENT.Agent_Types import Peacekeeper, Gatherer

            role_map = {"peacekeeper": Peacekeeper, "gatherer": Gatherer}

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
                network_type=self.network_type,
            )

            if not agent:
                raise RuntimeError(
                    "spawn_agent returned None (no valid location found)."
                )

            # 🛠 Initialise critical fields
            agent.current_action = None
            agent.current_task = None
            agent.current_task_state = utils_config.TaskState.NONE
            agent.mode = self.game_manager.mode
            agent.log_prob = None
            agent.value = None

            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[SPAWN] Created {role} for Faction {self.id} at ({agent.x}, {agent.y}).",
                    level=logging.INFO,
                )
            print(f"Created {role} for Faction {self.id} at ({agent.x}, {agent.y}).")
            return agent

        except Exception as e:
            logger.log_msg(
                f"[SPAWN ERROR] Failed to create agent: {str(e)}\n{traceback.format_exc()}",
                level=logging.ERROR,
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

            # Ignore self threats (shouldn't happen, but just in case)
            if threat_id.faction_id == self.id:
                continue

            tx, ty = threat.get("location", (-999, -999))
            dist_sq = (tx - hq_x) ** 2 + (ty - hq_y) ** 2

            if dist_sq <= detection_radius**2:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[HQ THREAT] Faction {self.id} HQ is under threat from enemy agent {threat_id} at {tx, ty}",
                        level=logging.INFO,
                    )
                return True

        return False

    def evaluate_agent_swap_candidates(self, target_role: str):
        """
        Evaluate which agents should be swapped to the target role.
        Returns a list of (agent, swap_score) tuples sorted by priority.
        """
        try:
            if not self.agents:
                return []

            candidates = []

            for agent in self.agents:
                if agent.role == target_role:
                    continue  # Skip agents that are already the target role

                swap_score = 0

                # Base score for role mismatch
                if target_role == "gatherer":
                    # Prefer swapping peacekeepers to gatherers when we need resources
                    if agent.role == "peacekeeper":
                        if self.food_balance < 20 or self.gold_balance < 30:
                            swap_score += 3  # High priority for resource gathering
                        else:
                            swap_score += (
                                1  # Lower priority when resources are sufficient
                            )
                elif target_role == "peacekeeper":
                    # Prefer swapping gatherers to peacekeepers when under threat
                    if agent.role == "gatherer":
                        if self.global_state.get("threat_count", 0) > 0:
                            swap_score += 3  # High priority for defense
                        else:
                            swap_score += 1  # Lower priority when safe

                # Consider agent health and performance
                if hasattr(agent, "Health") and agent.Health < 50:
                    swap_score += 2  # Injured agents are good swap candidates

                # Consider agent position (prefer swapping agents far from HQ)
                if hasattr(agent, "x") and hasattr(agent, "y"):
                    hq_x, hq_y = self.home_base["position"]
                    distance = ((agent.x - hq_x) ** 2 + (agent.y - hq_y) ** 2) ** 0.5
                    if distance > 100:  # Far from HQ
                        swap_score += 1

                # Consider current task state
                if hasattr(agent, "current_task_state"):
                    if agent.current_task_state in [
                        utils_config.TaskState.SUCCESS,
                        utils_config.TaskState.FAILURE,
                    ]:
                        swap_score += 1  # Idle agents are good swap candidates

                if swap_score > 0:
                    candidates.append((agent, swap_score))

            # Sort by swap score (highest first)
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates

        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[SWAP EVALUATION] Error evaluating swap candidates: {e}",
                    level=logging.ERROR,
                )
            return []

    def swap_agent_role(self, agent, new_role: str):
        """
        Swap an existing agent to a new role.
        This is more cost-effective than recruiting a new agent.
        """
        try:
            swap_cost = utils_config.Gold_Cost_for_Agent_Swap

            if self.gold_balance < swap_cost:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[HQ SWAP] Faction {self.id} lacks gold to swap {agent.role} to {new_role}.",
                        level=logging.WARNING,
                    )
                return False

            # Store old role for logging
            old_role = agent.role

            # Remove agent from current lists
            if agent in self.agents:
                self.agents.remove(agent)
            if agent in self.game_manager.agents:
                self.game_manager.agents.remove(agent)

            # Create new agent with the new role
            new_agent = self.create_agent(new_role)

            if new_agent:
                # Position the new agent where the old one was
                new_agent.x = agent.x
                new_agent.y = agent.y

                # Add to lists
                self.agents.append(new_agent)
                self.game_manager.agents.append(new_agent)

                # Deduct swap cost
                self.gold_balance -= swap_cost

                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[HQ SWAP] Faction {self.id} swapped {old_role} to {new_role} at ({new_agent.x}, {new_agent.y}) — Gold: {self.gold_balance}",
                        level=logging.INFO,
                    )

                return True
            else:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[HQ SWAP] Failed to create new {new_role} during swap.",
                        level=logging.ERROR,
                    )
                return False

        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[HQ SWAP] Error swapping agent {agent.agent_id if hasattr(agent, 'agent_id') else 'unknown'}: {e}",
                    level=logging.ERROR,
                )
            return False

    def get_faction_composition(self):
        """
        Get current faction composition and suggest optimal unit distribution.
        """
        try:
            gatherer_count = len([a for a in self.agents if a.role == "gatherer"])
            peacekeeper_count = len([a for a in self.agents if a.role == "peacekeeper"])
            total_agents = len(self.agents)

            composition = {
                "gatherers": gatherer_count,
                "peacekeepers": peacekeeper_count,
                "total": total_agents,
                "gatherer_ratio": (
                    gatherer_count / total_agents if total_agents > 0 else 0
                ),
                "peacekeeper_ratio": (
                    peacekeeper_count / total_agents if total_agents > 0 else 0
                ),
            }

            # Analyze current needs
            needs_analysis = self.analyze_faction_needs()

            # Suggest optimal distribution
            suggestions = []

            if needs_analysis["resource_priority"] > needs_analysis["defense_priority"]:
                if gatherer_count < peacekeeper_count:
                    suggestions.append(
                        {
                            "action": "SWAP_TO_GATHERER",
                            "reason": "Need more resource gathering",
                            "priority": needs_analysis["resource_priority"],
                        }
                    )
            elif (
                needs_analysis["defense_priority"] > needs_analysis["resource_priority"]
            ):
                if peacekeeper_count < gatherer_count:
                    suggestions.append(
                        {
                            "action": "SWAP_TO_PEACEKEEPER",
                            "reason": "Need more defense",
                            "priority": needs_analysis["defense_priority"],
                        }
                    )

            # Balance suggestions
            if abs(gatherer_count - peacekeeper_count) > 1:
                if gatherer_count > peacekeeper_count:
                    suggestions.append(
                        {
                            "action": "SWAP_TO_PEACEKEEPER",
                            "reason": "Balance unit distribution",
                            "priority": 1.0,
                        }
                    )
                else:
                    suggestions.append(
                        {
                            "action": "SWAP_TO_GATHERER",
                            "reason": "Balance unit distribution",
                            "priority": 1.0,
                        }
                    )

            return {
                "composition": composition,
                "needs_analysis": needs_analysis,
                "suggestions": suggestions,
            }

        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[COMPOSITION] Error analyzing faction composition: {e}",
                    level=logging.ERROR,
                )
            return {}

    def analyze_faction_needs(self):
        """
        Analyze current faction needs to determine optimal unit distribution.
        """
        try:
            needs = {
                "resource_priority": 0.0,
                "defense_priority": 0.0,
                "exploration_priority": 0.0,
            }

            # Resource needs
            if self.food_balance < 20:
                needs["resource_priority"] += 2.0
            elif self.food_balance < 50:
                needs["resource_priority"] += 1.0

            if self.gold_balance < 30:
                needs["resource_priority"] += 2.0
            elif self.gold_balance < 80:
                needs["resource_priority"] += 1.0

            # Defense needs
            threat_count = self.global_state.get("threat_count", 0)
            if threat_count > 2:
                needs["defense_priority"] += 3.0
            elif threat_count > 0:
                needs["defense_priority"] += 2.0

            # Exploration needs
            if len(self.resources) < 5:
                needs["exploration_priority"] += 1.0

            # Normalize priorities
            total = sum(needs.values())
            if total > 0:
                for key in needs:
                    needs[key] = needs[key] / total

            return needs

        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[NEEDS ANALYSIS] Error analyzing faction needs: {e}",
                    level=logging.ERROR,
                )
            return {
                "resource_priority": 0.5,
                "defense_priority": 0.5,
                "exploration_priority": 0.0,
            }

    def suggest_optimal_swap(self):
        """
        Suggest the optimal swap action based on current faction state.
        """
        try:
            composition = self.get_faction_composition()
            if not composition:
                return None

            suggestions = composition.get("suggestions", [])
            if not suggestions:
                return None

            # Sort by priority and return the best suggestion
            suggestions.sort(key=lambda x: x["priority"], reverse=True)
            return suggestions[0]["action"]

        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[SWAP SUGGESTION] Error suggesting optimal swap: {e}",
                    level=logging.ERROR,
                )
            return None

    def log_swap_statistics(self):
        """
        Log current swap statistics and faction composition for monitoring.
        """
        try:
            composition = self.get_faction_composition()
            if not composition:
                return

            comp = composition["composition"]
            needs = composition["needs_analysis"]
            suggestions = composition["suggestions"]

            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[SWAP STATS] Faction {self.id} Composition: "
                    f"Gatherers: {comp['gatherers']}, Peacekeepers: {comp['peacekeepers']}, "
                    f"Total: {comp['total']}",
                    level=logging.INFO,
                )

                logger.log_msg(
                    f"[SWAP STATS] Faction {self.id} Needs: "
                    f"Resources: {needs['resource_priority']:.2f}, "
                    f"Defense: {needs['defense_priority']:.2f}, "
                    f"Exploration: {needs['exploration_priority']:.2f}",
                    level=logging.INFO,
                )

                if suggestions:
                    best_suggestion = suggestions[0]
                    logger.log_msg(
                        f"[SWAP STATS] Faction {self.id} Best Swap: "
                        f"{best_suggestion['action']} - {best_suggestion['reason']} "
                        f"(Priority: {best_suggestion['priority']:.2f})",
                        level=logging.INFO,
                    )
                else:
                    logger.log_msg(
                        f"[SWAP STATS] Faction {self.id} No swap suggestions needed",
                        level=logging.INFO,
                    )

        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[SWAP STATS] Error logging swap statistics: {e}",
                    level=logging.ERROR,
                )

    def get_swap_efficiency_metrics(self):
        """
        Calculate efficiency metrics for the swapping system.
        """
        try:
            if not hasattr(self, "strategy_history") or not self.strategy_history:
                return {}

            # Count swap strategies used
            swap_strategies = [
                s for s in self.strategy_history if s.startswith("SWAP_TO_")
            ]
            total_strategies = len(self.strategy_history)

            # Calculate efficiency metrics
            swap_usage_rate = (
                len(swap_strategies) / total_strategies if total_strategies > 0 else 0
            )

            # Analyze strategy distribution
            strategy_counts = {}
            for strategy in self.strategy_history:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

            return {
                "total_strategies": total_strategies,
                "swap_strategies_used": len(swap_strategies),
                "swap_usage_rate": swap_usage_rate,
                "strategy_distribution": strategy_counts,
                "efficiency_score": self.calculate_swap_efficiency_score(),
            }

        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[SWAP EFFICIENCY] Error calculating efficiency metrics: {e}",
                    level=logging.ERROR,
                )
            return {}

    def calculate_swap_efficiency_score(self):
        """
        Calculate a score indicating how efficiently the swapping system is being used.
        """
        try:
            if len(self.agents) == 0:
                return 0.0

            # Get current composition
            gatherer_count = len([a for a in self.agents if a.role == "gatherer"])
            peacekeeper_count = len([a for a in self.agents if a.role == "peacekeeper"])
            total_agents = len(self.agents)

            # Calculate balance score (closer to 0.5 is better)
            gatherer_ratio = gatherer_count / total_agents
            balance_score = 1.0 - abs(gatherer_ratio - 0.5) * 2  # 0.5 = perfect balance

            # Calculate resource efficiency
            resource_efficiency = min(
                1.0, (self.food_balance + self.gold_balance) / 100.0
            )

            # Calculate threat response efficiency
            threat_count = self.global_state.get("threat_count", 0)
            if threat_count == 0:
                threat_efficiency = 1.0
            else:
                # More peacekeepers when there are threats is better
                peacekeeper_ratio = peacekeeper_count / total_agents
                threat_efficiency = peacekeeper_ratio if threat_count > 0 else 1.0

            # Weighted average
            efficiency_score = (
                balance_score * 0.4
                + resource_efficiency * 0.3
                + threat_efficiency * 0.3
            )

            return max(0.0, min(1.0, efficiency_score))

        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[SWAP EFFICIENCY] Error calculating efficiency score: {e}",
                    level=logging.ERROR,
                )
            return 0.0

    def get_enhanced_global_state(self):
        """
        Get enhanced global state with additional context for better strategy selection.
        This helps the HQ network make more informed decisions about when to use swap strategies.
        """
        try:
            # Get base global state
            enhanced_state = self.global_state.copy()

            # Always add agent count information (crucial for zero-agent case!)
            enhanced_state["friendly_agent_count"] = len(self.agents)
            enhanced_state["gatherer_count"] = len(
                [a for a in self.agents if a.role == "gatherer"]
            )
            enhanced_state["peacekeeper_count"] = len(
                [a for a in self.agents if a.role == "peacekeeper"]
            )
            enhanced_state["has_agents"] = 1.0 if len(self.agents) > 0 else 0.0

            # Add cost-benefit analysis (important for zero-agent case!)
            enhanced_state["can_afford_recruit"] = (
                1.0 if self.gold_balance >= utils_config.Gold_Cost_for_Agent else 0.0
            )
            enhanced_state["can_afford_swap"] = (
                1.0
                if self.gold_balance >= utils_config.Gold_Cost_for_Agent_Swap
                else 0.0
            )
            enhanced_state["gold_balance_norm"] = min(self.gold_balance / 1000.0, 2.0)

            # Add swap-relevant context
            if len(self.agents) > 0:
                composition = self.get_faction_composition()
                if composition:
                    comp = composition["composition"]
                    needs = composition["needs_analysis"]

                    enhanced_state["unit_balance"] = abs(
                        comp["gatherers"] - comp["peacekeepers"]
                    )

                    # Add needs analysis
                    enhanced_state["resource_priority"] = needs["resource_priority"]
                    enhanced_state["defense_priority"] = needs["defense_priority"]
                    enhanced_state["exploration_priority"] = needs[
                        "exploration_priority"
                    ]

                    # Add swap opportunity indicators
                    enhanced_state["swap_to_gatherer_benefit"] = 0.0
                    enhanced_state["swap_to_peacekeeper_benefit"] = 0.0

                    if (
                        comp["gatherers"] < comp["peacekeepers"]
                        and needs["resource_priority"] > 0.6
                    ):
                        enhanced_state["swap_to_gatherer_benefit"] = needs[
                            "resource_priority"
                        ]

                    if (
                        comp["peacekeepers"] < comp["gatherers"]
                        and needs["defense_priority"] > 0.6
                    ):
                        enhanced_state["swap_to_peacekeeper_benefit"] = needs[
                            "defense_priority"
                        ]
            else:
                # No agents: high priority to recruit!
                enhanced_state["unit_balance"] = 0.0
                enhanced_state["resource_priority"] = 0.5
                enhanced_state["defense_priority"] = 0.5
                enhanced_state["exploration_priority"] = 0.0
                enhanced_state["swap_to_gatherer_benefit"] = 0.0
                enhanced_state["swap_to_peacekeeper_benefit"] = 0.0

            # Add territory change as observation (not directive)
            if self.territory_count and hasattr(self, "previous_territory_count"):
                territory_delta = self.territory_count - self.previous_territory_count
                # Normalized: -1 = losing ground, 0 = stable, +1 = gaining ground
                enhanced_state["territory_delta"] = max(
                    -1.0, min(1.0, territory_delta / 10.0)
                )
            else:
                enhanced_state["territory_delta"] = 0.0

            self.previous_territory_count = self.territory_count

            # Add raw distance observations (let network interpret significance)
            enhanced_state["nearest_threat_distance"] = self.global_state.get(
                "nearest_threat", {}
            ).get("distance", float("inf"))
            enhanced_state["nearest_resource_distance"] = self.global_state.get(
                "nearest_resource", {}
            ).get("distance", float("inf"))

            # Add efficiency metrics
            if len(self.agents) > 0:
                efficiency_metrics = self.get_swap_efficiency_metrics()
                if efficiency_metrics:
                    enhanced_state["swap_efficiency_score"] = efficiency_metrics.get(
                        "efficiency_score", 0.0
                    )
                    enhanced_state["swap_usage_rate"] = efficiency_metrics.get(
                        "swap_usage_rate", 0.0
                    )
            else:
                enhanced_state["swap_efficiency_score"] = 0.0
                enhanced_state["swap_usage_rate"] = 0.0

            return enhanced_state

        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[ENHANCED STATE] Error getting enhanced global state: {e}",
                    level=logging.ERROR,
                )
            return self.global_state

"""Common Imports"""

from SHARED.core_imports import *

"""File Specific Imports"""
from NEURAL_NETWORK.PPO_Agent_Network import PPOModel
from NEURAL_NETWORK.DQN_Model import DQNModel
from NEURAL_NETWORK.HQ_Network import HQ_Network
from NEURAL_NETWORK.Common import Training_device, get_hq_input_size_from_checkpoint
from AGENT.agent_communication import CommunicationSystem
from AGENT.hierarchical_rewards import HierarchicalRewardManager
from AGENT.learned_communication import LearnedCommunicationSystem
from AGENT.experience_sharing import ExperienceSharingSystem
from AGENT.learned_state_representation import LearnedStateRepresentationSystem
from AGENT.strategy_composition import StrategyCompositionSystem
from AGENT.meta_learning import MetaLearningSystem
from AGENT.strategy_visualization import StrategyVisualizer
import UTILITIES.utils_config as utils_config
import matplotlib.pyplot as plt


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
            self.state_size = state_size
            self.action_size = action_size
            self.role_size = role_size
            self.local_state_size = local_state_size
            self.global_state_size = global_state_size
            self.gold_balance = 0
            self.food_balance = 0
            self.current_strategy = None
            self.experience_buffer = []
            self.resources = []  # Initialise known resources
            self.threats = []  # Initialise known threats
            self.task_counter = 0
            self.assigned_tasks = {}
            self.territory_count = 0  # Track number of tiles owned by this faction
            self._territory_cache = None  # Cached territory count
            self._territory_cache_step = -1  # Step when cache was calculated

            self.unvisited_cells = set()
            self.reports = []
            self.recent_threat_reports = (
                []
            )  # Track recent threat reports with timestamps
            self.strategy_history = []  # Track strategies chosen

            self.create_task = utils_config.create_task
            self.mode = mode

            # Initialise home_base with default values
            self.home_base = {
                "position": (0, 0),  # To be set during initialisation
                "size": 50,  # Default size of the base
                "colour": colour,  # Match faction colour
                "health": 100,  # HQ health (0-100)
                "max_health": 100,  # Maximum HQ health
                "is_destroyed": False,  # Whether HQ is destroyed
            }

            self.game_manager = game_manager

            self.global_state = {
                key: None for key in utils_config.STATE_FEATURES_MAP["global_state"]
            }
            # Populate the initial global state
            self.global_state.update(
                {
                    "HQ_health": self.home_base["health"]
                    / self.home_base["max_health"],  # Normalized HQ health (0-1)
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

            # Initialize hierarchical reward manager
            self.hierarchical_reward_manager = HierarchicalRewardManager(self.id)

            # Initialize learned communication system
            self.learned_communication = LearnedCommunicationSystem(
                self.id, self.agents
            )

            # Initialize experience sharing system
            self.experience_sharing = ExperienceSharingSystem(self.id, self.agents)

            # Initialize learned state representation system
            self.learned_state_representation = LearnedStateRepresentationSystem(
                self.id, self.state_size
            )

            # Initialize strategy composition system
            self.strategy_composition = StrategyCompositionSystem(
                self.id, self.state_size
            )

            # Initialize meta-learning system
            self.meta_learning = MetaLearningSystem(self.id, self.state_size)

            # Initialize strategy visualization system
            self.strategy_visualization = StrategyVisualizer(self.id, self.state_size)

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
                            faction_id=self.id,
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
                            faction_id=self.id,
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
                # Use parametric strategy execution if parameters are available
                if (
                    hasattr(self, "current_strategy_parameters")
                    and self.current_strategy_parameters
                ):
                    self.perform_HQ_Strategy_parametric(
                        new_strategy, self.current_strategy_parameters
                    )
                else:
                    self.perform_HQ_Strategy(new_strategy)
            else:
                if utils_config.ENABLE_LOGGING:
                    print(
                        f"\033[93m Faction {self.id} maintained HQ strategy: {self.current_strategy}\033[0m"
                    )
                self.current_strategy = new_strategy
                # Use parametric strategy execution if parameters are available
                if (
                    hasattr(self, "current_strategy_parameters")
                    and self.current_strategy_parameters
                ):
                    self.perform_HQ_Strategy_parametric(
                        self.current_strategy, self.current_strategy_parameters
                    )
                else:
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

        # Update HQ health from home_base
        if "health" in self.home_base and "max_health" in self.home_base:
            if self.home_base["max_health"] > 0:
                self.global_state["HQ_health"] = (
                    self.home_base["health"] / self.home_base["max_health"]
                )
            else:
                self.global_state["HQ_health"] = 0.0

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

    def receive_report(self, report, current_step=None):
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
                # Update location if it's changed (re-detected threat)
                if existing_threat["location"] != location:
                    existing_threat["location"] = location
                    # Update last_seen with current step
                    existing_threat["last_seen"] = (
                        current_step if current_step is not None else self.current_step
                    )
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"Faction {self.id} updated threat ID {threat_id} to location {location}."
                        )
                else:
                    # Same location - just update last_seen timestamp to indicate it's still visible
                    existing_threat["last_seen"] = (
                        current_step if current_step is not None else self.current_step
                    )

            else:
                self.global_state["threats"].append(data)
                # Add timestamp for new threats
                if "last_seen" not in data:
                    data["last_seen"] = (
                        current_step if current_step is not None else self.current_step
                    )
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"Faction {self.id} added new threat: {data['type']} ID {threat_id} at {location}."
                    )

            # Track recent threat reports with timestamp
            self.recent_threat_reports.append(
                {
                    "step": (
                        current_step if current_step is not None else self.current_step
                    ),
                    "threat_id": str(threat_id),
                    "location": location,
                    "reported_by": report.get("sender_id", "unknown"),
                }
            )

            # Keep only last 50 reports to prevent memory bloat
            if len(self.recent_threat_reports) > 50:
                self.recent_threat_reports.pop(0)

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

            # Check if threat was marked as inactive (eliminated but not yet cleaned up)
            if not threat.get("is_active", True):
                # This threat was marked as eliminated, remove it
                continue

            # Check threat age - use last_seen timestamp to determine intelligence staleness
            last_seen = threat.get("last_seen", self.current_step)
            threat_age = self.current_step - last_seen if last_seen else 0
            MAX_THREAT_AGE = (
                200  # Steps before considering a "last known" threat completely stale
            )

            # Determine if threat should be kept based on its type, current state, and age
            is_keep_threat = False

            if threat["type"] == "Faction HQ":
                # Check if the enemy HQ still exists and is not destroyed
                if isinstance(tid, utils_config.AgentIDStruc) and hasattr(
                    self.game_manager, "faction_manager"
                ):
                    # Find the faction for this HQ threat
                    enemy_faction_id = tid.faction_id
                    enemy_faction = next(
                        (
                            f
                            for f in self.game_manager.faction_manager.factions
                            if f.id == enemy_faction_id
                        ),
                        None,
                    )
                    if enemy_faction:
                        # Only keep if HQ is NOT destroyed (regardless of detection range or age)
                        # HQs are strategic targets - keep their "last known" location indefinitely
                        is_keep_threat = not enemy_faction.home_base.get(
                            "is_destroyed", False
                        )
                    else:
                        # Faction not found - likely destroyed, remove the threat
                        is_keep_threat = False
                else:
                    # Can't verify - keep it as a safety measure
                    is_keep_threat = True

                if is_keep_threat:
                    valid_threats.append(threat)
                continue

            if isinstance(tid, utils_config.AgentIDStruc):
                # Check if agent still exists and is alive
                matching_agent = next(
                    (
                        agent
                        for agent in self.game_manager.agents
                        if getattr(agent, "agent_id", None) == tid
                    ),
                    None,
                )

                if matching_agent:
                    # Agent exists - keep if alive (regardless of detection range or age)
                    # Mobile threats can move, so keep "last known" location even if stale
                    is_keep_threat = getattr(matching_agent, "Health", 1) > 0
                else:
                    # Agent doesn't exist anymore - likely removed/destroyed
                    # Only remove if it's been confirmed missing for a while
                    # This prevents immediately removing threats when agents despawn
                    is_keep_threat = threat_age < MAX_THREAT_AGE

                if is_keep_threat:
                    valid_threats.append(threat)
                # Update threat age indicator for the HQ network to use
                threat["age"] = threat_age

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

                # Convert HQ position to grid coordinates for comparison
                hq_grid_x = int(hq_position[0] // utils_config.CELL_SIZE)
                hq_grid_y = int(hq_position[1] // utils_config.CELL_SIZE)
                hq_grid_position = (hq_grid_x, hq_grid_y)

                # Check if agent is already at HQ
                agent_grid_x = int(agent.x // utils_config.CELL_SIZE)
                agent_grid_y = int(agent.y // utils_config.CELL_SIZE)
                is_at_hq = agent_grid_x == hq_grid_x and agent_grid_y == hq_grid_y

                already_defending = (
                    current_type_id == utils_config.TASK_TYPE_MAPPING["move_to"]
                    and current
                    and current.get("target", {}).get("position") == hq_grid_position
                )

                # If already at HQ or already defending, don't assign new task
                if is_at_hq or already_defending:
                    return current

                if current:
                    agent.current_task = None
                    agent.update_task_state(utils_config.TaskState.NONE)
                return self.assign_move_to_task(
                    agent, hq_grid_position, label="DefendHQ"
                )

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

                # Convert to grid for comparison
                hq_grid_x = int(hq_position[0] // utils_config.CELL_SIZE)
                hq_grid_y = int(hq_position[1] // utils_config.CELL_SIZE)
                hq_grid_position = (hq_grid_x, hq_grid_y)

                # Check if already at HQ
                agent_grid_x = int(agent.x // utils_config.CELL_SIZE)
                agent_grid_y = int(agent.y // utils_config.CELL_SIZE)
                is_at_hq = agent_grid_x == hq_grid_x and agent_grid_y == hq_grid_y

                if not is_at_hq:
                    if current:
                        agent.current_task = None
                        agent.update_task_state(utils_config.TaskState.NONE)
                    return self.assign_move_to_task(
                        agent, hq_grid_position, label="DefendHQ"
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
            # Keep target in GRID coordinates (explore targets should be grid-based)
            target = {"position": (cell_x, cell_y), "type": "Explore"}

            logger.log_msg(
                f"Created task: {task_string_id} for agent {agent.agent_id} at grid ({cell_x}, {cell_y})",
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
        # Use cached value if still valid for this step
        if (
            hasattr(self, "_territory_cache_step")
            and hasattr(self, "_territory_cache")
            and self._territory_cache_step == self.current_step
            and self._territory_cache is not None
        ):
            self.territory_count = self._territory_cache
            return

        # Calculate fresh
        self.territory_count = sum(
            1
            for row in terrain.grid
            for cell in row
            if str(cell["faction"]) == str(self.id)
        )

        # Cache for this step
        self._territory_cache = self.territory_count
        self._territory_cache_step = self.current_step

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

            # Check if network supports parametric output (new networks)
            if hasattr(self.network, "predict_strategy_parametric"):
                try:
                    strategy, parameters = self.network.predict_strategy_parametric(
                        enhanced_state
                    )
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"[HQ PARAMETRIC] Faction {self.id} selected parametric strategy: {strategy} with parameters: {parameters}",
                            level=logging.INFO,
                        )
                except Exception as e:
                    logger.log_msg(
                        f"[ERROR] Parametric strategy prediction failed: {e}, falling back to standard prediction",
                        level=logging.ERROR,
                    )
                    strategy = self.network.predict_strategy(enhanced_state)
            else:
                # Fall back to standard prediction (old networks)
                strategy = self.network.predict_strategy(enhanced_state)

            if strategy in utils_config.HQ_STRATEGY_OPTIONS:
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
                    f"[HQ STRATEGY] Invalid strategy returned: {strategy}. Defaulting to NO_PRIORITY.",
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
            # Check if recruitment is logical: do we have threats?
            threat_count = self.global_state.get("threat_count", 0)
            has_threats = threat_count > 0

            if (
                self.gold_balance >= Agent_cost
                and len(self.agents) < utils_config.MAX_AGENTS
            ):
                new_agent = self.recruit_agent("peacekeeper")
                if new_agent:
                    print(f"Faction {self.id} bought a Peacekeeper")
                    # Reward higher if recruitment addresses threats
                    if has_threats:
                        self.hq_step_rewards.append(+1.5)  # Good strategic choice
                    else:
                        self.hq_step_rewards.append(+0.5)  # Recruited but no threats
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
            # Check if recruitment is logical: do we have resources to gather?
            resource_count = self.global_state.get("resource_count", 0)
            has_resources = resource_count > 0
            # Also check if we have enough gatherers relative to threats
            gatherer_count = len([a for a in self.agents if a.role == "gatherer"])
            peacekeeper_count = len([a for a in self.agents if a.role == "peacekeeper"])
            needs_more_gatherers = (
                gatherer_count <= peacekeeper_count
            )  # Balanced or skewed toward peacekeepers

            if (
                self.gold_balance >= Agent_cost
                and len(self.agents) < utils_config.MAX_AGENTS
            ):
                new_agent = self.recruit_agent("gatherer")
                if new_agent:
                    print(f"Faction {self.id} bought a Gatherer")
                    # Reward higher if resources available AND it balances composition
                    if has_resources and needs_more_gatherers:
                        self.hq_step_rewards.append(+1.5)  # Excellent strategic choice
                    elif has_resources:
                        self.hq_step_rewards.append(+1.0)  # Good choice (has resources)
                    else:
                        self.hq_step_rewards.append(+0.3)  # Recruited but no resources
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

            # Check if swap is logical: do we have resources and need more gatherers?
            resource_count = self.global_state.get("resource_count", 0)
            gatherer_count = len([a for a in self.agents if a.role == "gatherer"])
            peacekeeper_count = len([a for a in self.agents if a.role == "peacekeeper"])
            has_resources = resource_count > 0

            # If no resources at all, swapping to gatherer is pointless
            if not has_resources:
                logger.log_msg(
                    f"[HQ EXECUTE] No resources available - gatherers have no goal.",
                    level=logging.WARNING,
                )
                self.hq_step_rewards.append(-0.5)  # Penalize pointless swap
                self.current_strategy = None
                return retest_strategy()

            needs_more_gatherers = gatherer_count < peacekeeper_count

            if candidates:
                best_candidate, score = candidates[0]
                if self.swap_agent_role(best_candidate, "gatherer"):
                    print(
                        f"Faction {self.id} swapped {best_candidate.role} to Gatherer"
                    )
                    # Reward based on strategic value of the swap
                    if has_resources and needs_more_gatherers:
                        self.hq_step_rewards.append(+1.2)  # Excellent swap
                    elif has_resources:
                        self.hq_step_rewards.append(+0.8)  # Good swap
                    else:
                        self.hq_step_rewards.append(+0.3)  # Swap OK but not optimal
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

            # Check if swap is logical: do we have threats and need more peacekeepers?
            resource_count = self.global_state.get("resource_count", 0)
            threat_count = self.global_state.get("threat_count", 0)
            gatherer_count = len([a for a in self.agents if a.role == "gatherer"])
            peacekeeper_count = len([a for a in self.agents if a.role == "peacekeeper"])
            has_threats = threat_count > 0
            has_resources = resource_count > 0

            # If no resources AND we have gatherers, switching to peacekeeper is smart
            no_resources_with_gatherers = not has_resources and gatherer_count > 0

            # Normal peacekeeper need based on threats
            needs_more_peacekeepers = peacekeeper_count < gatherer_count

            if candidates:
                best_candidate, score = candidates[0]
                if self.swap_agent_role(best_candidate, "peacekeeper"):
                    print(
                        f"Faction {self.id} swapped {best_candidate.role} to Peacekeeper"
                    )
                    # Reward based on strategic value of the swap
                    if no_resources_with_gatherers:
                        self.hq_step_rewards.append(
                            +1.5
                        )  # Excellent - repurpose idle gatherers
                    elif has_threats and needs_more_peacekeepers:
                        self.hq_step_rewards.append(
                            +1.2
                        )  # Excellent - addressing threats
                    elif has_threats or needs_more_peacekeepers:
                        self.hq_step_rewards.append(+0.8)  # Good swap
                    else:
                        self.hq_step_rewards.append(+0.3)  # Swap OK but not optimal
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

            # Check if there are threats and peacekeepers to defend
            if not nearby_threat_found:
                logger.log_msg(
                    f"[HQ STRATEGY] No nearby threats to defend HQ.",
                    level=logging.WARNING,
                )
                self.hq_step_rewards.append(-0.5)  # Penalize illogical strategy
                return retest_strategy()

            # Strategy is valid — assign peacekeepers to move to HQ
            self.defensive_position = hq_pos
            logger.log_msg(
                f"[HQ STRATEGY] Nearby threat detected. Assigning peacekeepers to defend HQ at {hq_pos}.",
                level=logging.INFO,
            )

            peacekeepers_assigned = 0
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
                    peacekeepers_assigned += 1
                    logger.log_msg(
                        f"[DEFEND ASSIGN] Peacekeeper {agent.agent_id} assigned to move to HQ.",
                        level=logging.INFO,
                    )

            # Reward based on actual defensive assignment
            if peacekeepers_assigned > 0:
                self.hq_step_rewards.append(+1.0)  # Successfully defended
            else:
                self.hq_step_rewards.append(0.0)  # No peacekeepers available to defend

        # ========== STRATEGY: Attack Threats ==========
        elif action == "ATTACK_THREATS":
            threat_count = self.global_state.get("threat_count", 0)
            if threat_count == 0:
                logger.log_msg(
                    f"[HQ EXECUTE] No threats to attack.", level=logging.WARNING
                )
                self.hq_step_rewards.append(-0.5)  # Penalise illogical strategy
                self.current_strategy = None
                return retest_strategy()

            # Check if any agents are actually assigned to eliminate threats
            peacekeepers_attacking = 0
            for agent in self.agents:
                if agent.role == "peacekeeper" and agent.current_task:
                    task_type = agent.current_task.get("type", "")
                    if task_type == "eliminate":
                        peacekeepers_attacking += 1

            # Reward based on actual threat engagement
            if peacekeepers_attacking > 0:
                self.hq_step_rewards.append(+1.5)  # Actually engaging threats
            else:
                self.hq_step_rewards.append(
                    +0.5
                )  # Strategy valid but no assignments yet

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

            # Check if we have gatherers to actually collect
            gatherer_count = len([a for a in self.agents if a.role == "gatherer"])
            # Check gold balance to see if we need more gold
            needs_gold = self.gold_balance < 200  # Low gold threshold

            if gatherer_count > 0:
                # Check if agents are actually assigned to mining
                miners_assigned = 0
                for agent in self.agents:
                    if agent.role == "gatherer" and agent.current_task:
                        task_type = agent.current_task.get("type", "")
                        if task_type == "mine":
                            miners_assigned += 1

                # Reward based on actual collection and need
                if miners_assigned > 0 and needs_gold:
                    self.hq_step_rewards.append(+1.2)  # Actively collecting needed gold
                elif miners_assigned > 0:
                    self.hq_step_rewards.append(+0.8)  # Actively collecting gold
                else:
                    self.hq_step_rewards.append(
                        +0.5
                    )  # Strategy valid but no mining yet
            else:
                self.hq_step_rewards.append(0.0)  # No gatherers to collect gold

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

            # Check if we have gatherers to actually collect
            gatherer_count = len([a for a in self.agents if a.role == "gatherer"])
            # Check food balance to see if we need more food
            needs_food = self.food_balance < 200  # Low food threshold

            if gatherer_count > 0:
                # Check if agents are actually assigned to foraging
                foragers_assigned = 0
                for agent in self.agents:
                    if agent.role == "gatherer" and agent.current_task:
                        task_type = agent.current_task.get("type", "")
                        if task_type == "forage":
                            foragers_assigned += 1

                # Reward based on actual collection and need
                if foragers_assigned > 0 and needs_food:
                    self.hq_step_rewards.append(+1.2)  # Actively collecting needed food
                elif foragers_assigned > 0:
                    self.hq_step_rewards.append(+0.8)  # Actively collecting food
                else:
                    self.hq_step_rewards.append(
                        +0.5
                    )  # Strategy valid but no foraging yet
            else:
                self.hq_step_rewards.append(0.0)  # No gatherers to collect food

        # ========== STRATEGY: Plant Trees ==========
        elif action == "PLANT_TREES":
            # Check if we have enough food to plant
            if self.food_balance < 3:
                logger.log_msg(
                    f"[HQ EXECUTE] Cannot plant trees: insufficient food ({self.food_balance}/3).",
                    level=logging.WARNING,
                )
                self.hq_step_rewards.append(-0.5)
                self.current_strategy = None
                return retest_strategy()

            # Check if we have gatherers to plant
            gatherer_count = len([a for a in self.agents if a.role == "gatherer"])
            if gatherer_count == 0:
                logger.log_msg(
                    f"[HQ EXECUTE] Cannot plant trees: no gatherers available.",
                    level=logging.WARNING,
                )
                self.hq_step_rewards.append(-0.5)
                self.current_strategy = None
                return retest_strategy()

            # Check how many trees we already have
            existing_trees = len(
                [
                    r
                    for r in self.global_state.get("resources", [])
                    if r["type"] == "AppleTree"
                ]
            )

            # Check if agents are actually assigned to planting
            planters_assigned = 0
            for agent in self.agents:
                if agent.role == "gatherer" and agent.current_task:
                    task_type = agent.current_task.get("type", "")
                    if task_type == "plant":
                        planters_assigned += 1

            # Reward based on strategic value: more trees = better resource production
            # But don't plant excessively if we already have many trees
            if (
                planters_assigned > 0 and existing_trees < 20
            ):  # Actively planting and room for more
                self.hq_step_rewards.append(+1.2)  # Excellent strategic expansion
            elif planters_assigned > 0:
                self.hq_step_rewards.append(+0.8)  # Actively planting
            elif existing_trees < 10:  # Room for expansion
                self.hq_step_rewards.append(+0.5)  # Strategic planting valid
            else:
                self.hq_step_rewards.append(0.0)  # Already have many trees

        # ========== STRATEGY: Plant Gold Veins ==========
        elif action == "PLANT_GOLD_VEINS":
            # Check if we have enough gold to plant
            if self.gold_balance < 5:
                logger.log_msg(
                    f"[HQ EXECUTE] Cannot plant gold veins: insufficient gold ({self.gold_balance}/5).",
                    level=logging.WARNING,
                )
                self.hq_step_rewards.append(-0.5)
                self.current_strategy = None
                return retest_strategy()

            # Check if we have gatherers to plant
            gatherer_count = len([a for a in self.agents if a.role == "gatherer"])
            if gatherer_count == 0:
                logger.log_msg(
                    f"[HQ EXECUTE] Cannot plant gold veins: no gatherers available.",
                    level=logging.WARNING,
                )
                self.hq_step_rewards.append(-0.5)
                self.current_strategy = None
                return retest_strategy()

            # Check how many gold veins we already have
            existing_veins = len(
                [
                    r
                    for r in self.global_state.get("resources", [])
                    if r["type"] == "GoldLump"
                ]
            )

            # Check if agents are actually assigned to planting gold
            vein_planters_assigned = 0
            for agent in self.agents:
                if agent.role == "gatherer" and agent.current_task:
                    task_type = agent.current_task.get("type", "")
                    if task_type == "plant_gold":
                        vein_planters_assigned += 1

            # Gold is valuable - planting is strategic but very expensive
            # Only plant if we have extra gold and room for more veins
            has_spare_gold = self.gold_balance >= 20  # Reserve significant gold
            if vein_planters_assigned > 0 and existing_veins < 10 and has_spare_gold:
                self.hq_step_rewards.append(+1.5)  # Excellent strategic gold expansion
            elif vein_planters_assigned > 0 and has_spare_gold:
                self.hq_step_rewards.append(+1.0)  # Actively planting gold
            elif existing_veins < 5 and has_spare_gold:
                self.hq_step_rewards.append(+0.6)  # Strategic planting valid
            elif has_spare_gold:
                self.hq_step_rewards.append(0.0)  # Already have enough veins
            else:
                self.hq_step_rewards.append(-0.3)  # Don't waste precious gold!

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

    def perform_HQ_Strategy_parametric(self, strategy_type: str, parameters: dict):
        """
        Execute strategy with learned parameters for enhanced flexibility.

        Args:
            strategy_type: The base strategy type (e.g., "RECRUIT", "SWAP", "RESOURCE")
            parameters: Dictionary containing learned parameters:
                - target_role: "gatherer" or "peacekeeper"
                - priority_resource: "gold" or "food"
                - use_mission_system: bool (whether to use mission-oriented tasks)
                - aggression_level: 0.0-1.0 (defensive to offensive)
                - resource_threshold: 0.0-1.0 (when to execute)
                - urgency: 0.0-1.0 (how urgent)
                - mission_autonomy: 0.0-1.0 (how much autonomy to give agents)
                - coordination_preference: 0.0-1.0 (how much to coordinate)
                - agent_count_target: int (desired agents)
                - mission_complexity: int (1-4, complexity level)
        """
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[HQ PARAMETRIC] Faction {self.id} executing parametric strategy: {strategy_type} with parameters: {parameters}",
                level=logging.INFO,
            )

        def retest_strategy():
            new_action = self.choose_HQ_Strategy()
            if new_action != strategy_type:
                logger.log_msg(
                    f"[HQ RETEST] Parametric strategy '{strategy_type}' invalid. Retesting and switching to '{new_action}'",
                    level=logging.WARNING,
                )
                self.perform_HQ_Strategy(new_action)  # Fall back to old system
            logger.log_msg(
                f"[HQ EXECUTE] Faction {self.id} executing updated strategy: {new_action}",
                level=logging.INFO,
            )

        # Extract parameters with defaults
        target_role = parameters.get("target_role", "gatherer")
        priority_resource = parameters.get("priority_resource", "gold")
        use_mission_system = parameters.get("use_mission_system", False)
        aggression_level = parameters.get("aggression_level", 0.5)
        resource_threshold = parameters.get("resource_threshold", 0.3)
        urgency = parameters.get("urgency", 0.5)
        mission_autonomy = parameters.get("mission_autonomy", 0.5)
        coordination_preference = parameters.get("coordination_preference", 0.5)
        agent_count_target = parameters.get("agent_count_target", 3)
        mission_complexity = parameters.get("mission_complexity", 2)

        # ========== PARAMETRIC RECRUITMENT ==========
        if strategy_type in ["RECRUIT_GATHERER", "RECRUIT_PEACEKEEPER"]:
            # Use learned target_role parameter
            role_to_recruit = target_role

            Agent_cost = utils_config.Gold_Cost_for_Agent

            # Use learned resource_threshold to determine if recruitment is logical
            if role_to_recruit == "peacekeeper":
                threat_count = self.global_state.get("threat_count", 0)
                has_threats = threat_count > 0
                # Use resource_threshold to determine recruitment urgency
                should_recruit = has_threats or (
                    resource_threshold < 0.3
                )  # Low threshold = recruit even without threats
            else:  # gatherer
                resource_count = self.global_state.get("resource_count", 0)
                has_resources = resource_count > 0
                should_recruit = has_resources or (resource_threshold < 0.3)

            if (
                self.gold_balance >= Agent_cost
                and len(self.agents) < utils_config.MAX_AGENTS
                and should_recruit
            ):
                new_agent = self.recruit_agent(role_to_recruit)
                if new_agent:
                    print(f"Faction {self.id} bought a {role_to_recruit} (parametric)")
                    # Reward based on parameter effectiveness
                    base_reward = 1.0
                    urgency_bonus = urgency * 0.5  # Higher urgency = higher reward
                    self.hq_step_rewards.append(base_reward + urgency_bonus)
                else:
                    logger.log_msg(
                        f"[HQ EXECUTE] Failed to recruit {role_to_recruit}: spawn failed.",
                        level=logging.WARNING,
                    )
                    self.current_strategy = None
                    self.hq_step_rewards.append(-0.5)
                    return retest_strategy()
            else:
                reason = (
                    "Not enough gold"
                    if self.gold_balance < Agent_cost
                    else (
                        "Agent limit reached"
                        if len(self.agents) >= utils_config.MAX_AGENTS
                        else "Resource threshold not met"
                    )
                )
                logger.log_msg(
                    f"[HQ EXECUTE] Cannot recruit {role_to_recruit}: {reason}.",
                    level=logging.WARNING,
                )
                self.current_strategy = None
                self.hq_step_rewards.append(-0.5)
                return retest_strategy()

        # ========== PARAMETRIC ROLE SWAPPING ==========
        elif strategy_type in ["SWAP_TO_GATHERER", "SWAP_TO_PEACEKEEPER"]:
            if len(self.agents) == 0:
                logger.log_msg(
                    f"[HQ EXECUTE] Cannot swap to {target_role}: No agents available.",
                    level=logging.WARNING,
                )
                self.current_strategy = None
                self.hq_step_rewards.append(-0.5)
                return retest_strategy()

            # Use learned parameters for swap decision
            role_to_swap_to = target_role

            # Use resource_threshold to determine swap necessity
            if role_to_swap_to == "gatherer":
                resource_count = self.global_state.get("resource_count", 0)
                has_resources = resource_count > 0
                should_swap = has_resources or (resource_threshold < 0.4)
            else:  # peacekeeper
                threat_count = self.global_state.get("threat_count", 0)
                has_threats = threat_count > 0
                should_swap = has_threats or (resource_threshold < 0.4)

            if should_swap:
                candidates = self.evaluate_agent_swap_candidates(role_to_swap_to)
                if candidates:
                    best_candidate, score = candidates[0]
                    if self.swap_agent_role(best_candidate, role_to_swap_to):
                        print(
                            f"Faction {self.id} swapped {best_candidate.role} to {role_to_swap_to} (parametric)"
                        )
                        # Reward based on urgency and effectiveness
                        base_reward = 0.8
                        urgency_bonus = urgency * 0.4
                        self.hq_step_rewards.append(base_reward + urgency_bonus)
                    else:
                        logger.log_msg(
                            f"[HQ EXECUTE] Failed to swap agent to {role_to_swap_to}.",
                            level=logging.WARNING,
                        )
                        self.current_strategy = None
                        self.hq_step_rewards.append(-0.5)
                        return retest_strategy()
                else:
                    logger.log_msg(
                        f"[HQ EXECUTE] No suitable candidates for swapping to {role_to_swap_to}.",
                        level=logging.WARNING,
                    )
                    self.current_strategy = None
                    self.hq_step_rewards.append(-0.5)
                    return retest_strategy()
            else:
                logger.log_msg(
                    f"[HQ EXECUTE] Resource threshold not met for swapping to {role_to_swap_to}.",
                    level=logging.WARNING,
                )
                self.hq_step_rewards.append(-0.3)  # Small penalty for unnecessary swap

        # ========== PARAMETRIC RESOURCE STRATEGIES ==========
        elif strategy_type in [
            "COLLECT_GOLD",
            "COLLECT_FOOD",
            "PLANT_TREES",
            "PLANT_GOLD_VEINS",
        ]:
            # Use priority_resource parameter to determine focus
            resource_type = priority_resource

            # Use urgency parameter to determine task priority assignment
            if strategy_type.startswith("COLLECT"):
                # Adjust task assignment based on urgency
                task_priority = urgency
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[HQ PARAMETRIC] Setting {resource_type} collection priority to {task_priority:.2f}",
                        level=logging.INFO,
                    )
                # Store parametric priority for task assignment
                self.parametric_task_priority = task_priority
                self.hq_step_rewards.append(0.5)  # Base reward for setting priority
            else:  # PLANT strategies
                # Use aggression_level to determine planting strategy
                if aggression_level > 0.7:  # High aggression = plant defensively
                    logger.log_msg(
                        f"[HQ PARAMETRIC] High aggression ({aggression_level:.2f}) - defensive planting",
                        level=logging.INFO,
                    )
                self.hq_step_rewards.append(0.3)  # Base reward for planting strategy

        # ========== PARAMETRIC COMBAT STRATEGIES ==========
        elif strategy_type in ["ATTACK_THREATS", "DEFEND_HQ"]:
            # Use aggression_level parameter to determine combat behavior
            if strategy_type == "ATTACK_THREATS":
                # High aggression = more aggressive threat engagement
                engagement_distance = 50 + (aggression_level * 100)  # 50-150 range
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[HQ PARAMETRIC] Setting threat engagement distance to {engagement_distance:.1f} (aggression: {aggression_level:.2f})",
                        level=logging.INFO,
                    )
                # Store parametric engagement distance
                self.parametric_engagement_distance = engagement_distance
                self.hq_step_rewards.append(0.6)
            else:  # DEFEND_HQ
                # Use urgency to determine defense priority
                defense_priority = urgency
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[HQ PARAMETRIC] Setting defense priority to {defense_priority:.2f}",
                        level=logging.INFO,
                    )
                self.parametric_defense_priority = defense_priority
                self.hq_step_rewards.append(0.4)

        # ========== PARAMETRIC PASSIVE STRATEGY ==========
        elif strategy_type == "NO_PRIORITY":
            # Use resource_threshold to determine when to stay passive
            if resource_threshold > 0.8:  # High threshold = stay passive longer
                logger.log_msg(
                    f"[HQ PARAMETRIC] High resource threshold ({resource_threshold:.2f}) - maintaining passive strategy",
                    level=logging.INFO,
                )
            self.hq_step_rewards.append(0.1)  # Small reward for patience

            # Set current strategy
            self.current_strategy = strategy_type

            # Store parameters for use in task assignment
            self.current_strategy_parameters = parameters

        # ========== MISSION ASSIGNMENT SYSTEM ==========
        # If mission system is enabled, assign missions instead of specific tasks
        if use_mission_system:
            self.assign_missions_to_agents(parameters)

    def assign_missions_to_agents(self, parameters: dict):
        """
        Assign mission-oriented tasks to agents based on current strategy and parameters.
        This replaces specific task assignment with high-level objectives.
        """
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[MISSION SYSTEM] Faction {self.id} assigning missions to agents",
                level=logging.INFO,
            )

        mission_autonomy = parameters.get("mission_autonomy", 0.5)
        coordination_preference = parameters.get("coordination_preference", 0.5)
        mission_complexity = parameters.get("mission_complexity", 2)
        urgency = parameters.get("urgency", 0.5)

        # Determine mission priority based on urgency
        if urgency > 0.8:
            priority = utils_config.MissionPriority.CRITICAL
        elif urgency > 0.6:
            priority = utils_config.MissionPriority.HIGH
        elif urgency > 0.4:
            priority = utils_config.MissionPriority.MEDIUM
        elif urgency > 0.2:
            priority = utils_config.MissionPriority.LOW
        else:
            priority = utils_config.MissionPriority.BACKGROUND

        # Get available mission types based on complexity
        complexity_levels = ["BEGINNER", "INTERMEDIATE", "ADVANCED", "EXPERT"]
        available_missions = []
        for i in range(mission_complexity):
            level = complexity_levels[i]
            available_missions.extend(utils_config.MISSION_COMPLEXITY_LEVELS[level])

        # Assign missions to idle agents
        for agent in self.agents:
            if agent.current_task is None or agent.current_task_state in [
                utils_config.TaskState.SUCCESS,
                utils_config.TaskState.FAILURE,
                utils_config.TaskState.NONE,
            ]:
                mission = self.create_mission_for_agent(
                    agent, available_missions, parameters, priority
                )
                if mission:
                    agent.current_task = mission
                    agent.update_task_state(utils_config.TaskState.PENDING)

                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"[MISSION ASSIGNED] Agent {agent.agent_id} ({agent.role}) assigned mission: {mission['mission_type']}",
                            level=logging.INFO,
                        )

    def create_mission_for_agent(self, agent, available_missions, parameters, priority):
        """
        Create a specific mission for an agent based on their role and current strategy.
        """
        role = agent.role
        strategy = self.current_strategy

        # Determine mission type based on role and strategy
        if role == "gatherer":
            if strategy in ["COLLECT_GOLD", "COLLECT_FOOD"]:
                mission_type = utils_config.MissionType.GATHER_RESOURCES
            elif strategy == "PLANT_TREES":
                mission_type = (
                    utils_config.MissionType.SECURE_AREA
                )  # Secure area for planting
            else:
                mission_type = utils_config.MissionType.EXPLORE_TERRITORY
        else:  # peacekeeper
            if strategy == "ATTACK_THREATS":
                mission_type = utils_config.MissionType.ELIMINATE_THREATS
            elif strategy == "DEFEND_HQ":
                mission_type = utils_config.MissionType.DEFEND_POSITION
            else:
                mission_type = utils_config.MissionType.SECURE_AREA

        # Create mission parameters
        mission_params = {
            "target_area": self.get_mission_target_area(mission_type, agent),
            "resource_preference": parameters.get("priority_resource", "any"),
            "threat_tolerance": parameters.get("aggression_level", 0.5),
            "coordination_level": parameters.get("coordination_preference", 0.5),
            "urgency_factor": parameters.get("urgency", 0.5),
            "success_criteria": self.get_mission_success_criteria(
                mission_type, parameters
            ),
            "time_limit": max(
                50, int(100 * parameters.get("urgency", 0.5))
            ),  # Urgency affects time limit
            "fallback_mission": "EXPLORE_TERRITORY",  # Default fallback
        }

        mission_id = (
            f"Mission-{mission_type.value}-{agent.agent_id}-{self.current_step}"
        )

        return utils_config.create_mission(
            mission_type=mission_type,
            parameters=mission_params,
            mission_id=mission_id,
            priority=priority,
        )

    def get_mission_target_area(self, mission_type, agent):
        """
        Determine the target area for a mission based on type and current game state.
        """
        if mission_type == utils_config.MissionType.DEFEND_POSITION:
            # Defend HQ area
            hq_pos = self.home_base["position"]
            return {
                "center": (
                    int(hq_pos[0] // utils_config.CELL_SIZE),
                    int(hq_pos[1] // utils_config.CELL_SIZE),
                ),
                "radius": 3,  # 3-cell radius around HQ
            }
        elif mission_type == utils_config.MissionType.GATHER_RESOURCES:
            # Target area with resources
            resources = self.global_state.get("resources", [])
            if resources:
                # Find resource cluster
                resource_positions = [r["location"] for r in resources]
                center_x = sum(pos[0] for pos in resource_positions) // len(
                    resource_positions
                )
                center_y = sum(pos[1] for pos in resource_positions) // len(
                    resource_positions
                )
                return {
                    "center": (center_x, center_y),
                    "radius": 5,  # 5-cell radius around resource cluster
                }
        elif mission_type == utils_config.MissionType.ELIMINATE_THREATS:
            # Target area with threats
            threats = [
                t
                for t in self.global_state.get("threats", [])
                if t["id"].faction_id != self.id
            ]
            if threats:
                threat_positions = [t["location"] for t in threats]
                center_x = sum(pos[0] for pos in threat_positions) // len(
                    threat_positions
                )
                center_y = sum(pos[1] for pos in threat_positions) // len(
                    threat_positions
                )
                return {
                    "center": (center_x, center_y),
                    "radius": 4,  # 4-cell radius around threat area
                }

        # Default: explore around agent's current position
        agent_grid_x = int(agent.x // utils_config.CELL_SIZE)
        agent_grid_y = int(agent.y // utils_config.CELL_SIZE)
        return {
            "center": (agent_grid_x, agent_grid_y),
            "radius": 8,  # 8-cell radius for exploration
        }

    def get_mission_success_criteria(self, mission_type, parameters):
        """
        Define success criteria for different mission types.
        """
        if mission_type == utils_config.MissionType.GATHER_RESOURCES:
            resource_type = parameters.get("priority_resource", "any")
            if resource_type == "gold":
                return "collect 3 gold"
            elif resource_type == "food":
                return "collect 5 food"
            else:
                return "collect 4 resources"
        elif mission_type == utils_config.MissionType.ELIMINATE_THREATS:
            return "eliminate all threats in area"
        elif mission_type == utils_config.MissionType.DEFEND_POSITION:
            return "defend position for 20 steps"
        elif mission_type == utils_config.MissionType.SECURE_AREA:
            return "patrol area for 15 steps"
        elif mission_type == utils_config.MissionType.EXPLORE_TERRITORY:
            return "explore 10 new cells"
        else:
            return "complete mission objectives"

    def compute_hq_reward(self, victory: bool = False) -> float:
        """
        Computes the HQ reward using the hierarchical reward system.
        This connects HQ strategy success to agent execution quality.
        """
        # Get agent performance data
        agent_performance = {}
        for agent in self.agents:
            if hasattr(agent, "ai") and hasattr(agent.ai, "memory"):
                rewards = agent.ai.memory.get("rewards", [])
                if rewards:
                    # Calculate average reward for this agent
                    avg_reward = sum(rewards) / len(rewards)
                    agent_performance[agent.agent_id] = avg_reward

        # Calculate mission progress
        mission_progress = self._calculate_mission_progress()

        # Use hierarchical reward manager to calculate HQ reward
        hq_reward = self.hierarchical_reward_manager.calculate_hq_reward(
            strategy=self.current_strategy or "UNKNOWN",
            parameters=getattr(self, "current_strategy_parameters", {}),
            execution_success=self._evaluate_strategy_execution_success(),
            agent_performance=agent_performance,
            mission_progress=mission_progress,
        )

        # Add victory bonus if applicable
        if victory:
            victory_bonus = utils_config.HIERARCHICAL_REWARD_CONFIG["hq_weights"][
                utils_config.RewardComponent.MISSION_SUCCESS
            ]
            hq_reward += victory_bonus

        # Store the hierarchical reward components for analysis
        self.hq_reward_components = (
            self.hierarchical_reward_manager.hq_reward_components
        )

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[HQ HIERARCHICAL REWARD] Faction {self.id}: {hq_reward:.3f} "
                f"(strategy: {self.current_strategy}, victory: {victory})",
                level=logging.INFO,
            )

        return hq_reward

    def _calculate_mission_progress(self) -> Dict[str, float]:
        """
        Calculate progress on various mission objectives.

        Returns:
            Dict[str, float]: Mission progress scores between 0.0 and 1.0
        """
        progress = {
            "resource_collection": 0.0,
            "threat_elimination": 0.0,
            "territory_control": 0.0,
            "agent_coordination": 0.0,
        }

        # Resource collection progress
        if hasattr(self, "gold_balance") and hasattr(self, "food_balance"):
            # Calculate based on resource balance (simplified)
            gold_progress = min(
                self.gold_balance / 100.0, 1.0
            )  # Normalize by expected max
            food_progress = min(self.food_balance / 100.0, 1.0)
            progress["resource_collection"] = (gold_progress + food_progress) / 2.0

        # Threat elimination progress
        if hasattr(self, "threats_eliminated"):
            # Calculate based on threats eliminated (simplified)
            threats_progress = min(
                self.threats_eliminated / 10.0, 1.0
            )  # Normalize by expected max
            progress["threat_elimination"] = threats_progress

        # Territory control progress
        if hasattr(self, "territory_count"):
            # Calculate based on territory controlled
            territory_progress = min(
                self.territory_count / 50.0, 1.0
            )  # Normalize by expected max
            progress["territory_control"] = territory_progress

        # Agent coordination progress
        if self.agents:
            # Calculate based on agent task distribution and success
            total_tasks = 0
            successful_tasks = 0
            for agent in self.agents:
                if hasattr(agent, "ai") and hasattr(agent.ai, "memory"):
                    rewards = agent.ai.memory.get("rewards", [])
                    if rewards:
                        total_tasks += len(rewards)
                        successful_tasks += sum(1 for r in rewards if r > 0)

            if total_tasks > 0:
                coordination_progress = successful_tasks / total_tasks
                progress["agent_coordination"] = coordination_progress

        return progress

    def process_learned_communication(self, current_step: int):
        """
        Process learned communication between agents.

        Args:
            current_step: Current simulation step
        """
        if not hasattr(self, "learned_communication"):
            return

        # Process communication for each agent
        for agent in self.agents:
            agent_id = agent.agent_id

            # Check if agent should communicate
            if self.learned_communication.should_communicate(agent, current_step):
                # Generate message
                message = self.learned_communication.generate_message(agent)
                if message:
                    # Send message to nearby agents
                    self._broadcast_learned_message(agent, message)

            # Process incoming messages
            self._process_incoming_messages(agent)

    def _broadcast_learned_message(self, sender: Any, message: Dict[str, Any]):
        """
        Broadcast a learned message to nearby agents.

        Args:
            sender: Sending agent
            message: Message dictionary
        """
        comm_range = utils_config.COMMUNICATION_CONFIG["communication_range"]
        max_recipients = utils_config.COMMUNICATION_CONFIG["message_types"][
            message["type"]
        ]["max_recipients"]

        # Find nearby agents
        nearby_agents = []
        for agent in self.agents:
            if agent != sender:
                distance = math.sqrt(
                    (sender.x - agent.x) ** 2 + (sender.y - agent.y) ** 2
                )
                if distance <= comm_range:
                    nearby_agents.append((agent, distance))

        # Sort by distance and select closest agents
        nearby_agents.sort(key=lambda x: x[1])
        recipients = [agent for agent, _ in nearby_agents[:max_recipients]]

        # Send message to recipients
        for recipient in recipients:
            if recipient.agent_id in self.learned_communication.message_queues:
                self.learned_communication.message_queues[recipient.agent_id].append(
                    message
                )

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[LEARNED COMMUNICATION] Agent {sender.agent_id} sent {message['type'].value} "
                f"to {len(recipients)} nearby agents",
                level=logging.DEBUG,
            )

    def _process_incoming_messages(self, agent: Any):
        """
        Process incoming messages for an agent.

        Args:
            agent: Agent to process messages for
        """
        agent_id = agent.agent_id

        if agent_id not in self.learned_communication.message_queues:
            return

        message_queue = self.learned_communication.message_queues[agent_id]
        processed_count = 0

        # Process messages (limit to avoid overwhelming)
        max_process_per_step = 3
        while message_queue and processed_count < max_process_per_step:
            message = message_queue.popleft()

            # Check if message has expired
            if hasattr(message, "expiry_steps") and message.get("expiry_steps", 0) <= 0:
                continue

            # Process message
            success = self.learned_communication.process_message(agent, message)
            if success:
                processed_count += 1

            # Decrement expiry steps
            if "expiry_steps" in message:
                message["expiry_steps"] -= 1

        if processed_count > 0 and utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[LEARNED COMMUNICATION] Agent {agent_id} processed {processed_count} messages",
                level=logging.DEBUG,
            )

    def get_coordination_reward(self, agent_id: str) -> float:
        """
        Get coordination reward for an agent from learned communication.

        Args:
            agent_id: Agent identifier

        Returns:
            Coordination reward value
        """
        if hasattr(self, "learned_communication"):
            return self.learned_communication.get_coordination_reward(agent_id)
        return 0.0

    def process_experience_sharing(self, current_step: int):
        """
        Process experience sharing between agents.

        Args:
            current_step: Current simulation step
        """
        if not hasattr(self, "experience_sharing"):
            return

        # Process experience sharing for each agent
        for agent in self.agents:
            agent_id = agent.agent_id

            # Check if agent should share experiences
            if (
                current_step
                % utils_config.EXPERIENCE_SHARING_CONFIG["sharing_frequency"]
                == 0
            ):
                self._process_agent_experience_sharing(agent)

            # Learn from shared experiences
            learning_reward = self.experience_sharing.learn_from_shared_experiences(
                agent_id
            )
            if learning_reward > 0:
                # Add learning reward to agent's reward
                if hasattr(agent, "ai") and hasattr(agent.ai, "memory"):
                    if "rewards" not in agent.ai.memory:
                        agent.ai.memory["rewards"] = []
                    agent.ai.memory["rewards"].append(learning_reward)

    def _process_agent_experience_sharing(self, agent: Any):
        """
        Process experience sharing for a specific agent.

        Args:
            agent: Agent to process experience sharing for
        """
        agent_id = agent.agent_id

        # Get agent's recent experiences from their memory
        if hasattr(agent, "ai") and hasattr(agent.ai, "memory"):
            recent_experiences = self._extract_recent_experiences(agent)

            # Process each experience
            for experience in recent_experiences:
                # Encode experience
                encoding, value, metadata = self.experience_sharing.encode_experience(
                    agent_id=agent_id,
                    state=experience.get("state"),
                    action=experience.get("action"),
                    reward=experience.get("reward", 0.0),
                    outcome=experience.get("outcome", "unknown"),
                    task_type=experience.get("task_type"),
                    coordination_data=experience.get("coordination_data", {}),
                )

                if encoding is not None and value is not None:
                    # Check if should share
                    if self.experience_sharing.should_share_experience(
                        agent_id, metadata
                    ):
                        # Share experience
                        success = self.experience_sharing.share_experience(
                            agent_id, metadata
                        )
                        if success:
                            # Add sharing reward
                            sharing_reward = utils_config.EXPERIENCE_SHARING_CONFIG[
                                "sharing_rewards"
                            ]["successful_sharing"]
                            if "rewards" not in agent.ai.memory:
                                agent.ai.memory["rewards"] = []
                            agent.ai.memory["rewards"].append(sharing_reward)

    def _extract_recent_experiences(self, agent: Any) -> List[Dict[str, Any]]:
        """
        Extract recent experiences from an agent's memory.

        Args:
            agent: Agent to extract experiences from

        Returns:
            List of recent experiences
        """
        experiences = []

        if hasattr(agent, "ai") and hasattr(agent.ai, "memory"):
            memory = agent.ai.memory

            # Extract experiences from memory (simplified)
            if "states" in memory and "actions" in memory and "rewards" in memory:
                states = memory.get("states", [])
                actions = memory.get("actions", [])
                rewards = memory.get("rewards", [])

                # Get recent experiences (last 5)
                recent_count = min(5, len(states))
                for i in range(recent_count):
                    if i < len(states) and i < len(actions) and i < len(rewards):
                        experience = {
                            "state": states[-(i + 1)] if i < len(states) else None,
                            "action": actions[-(i + 1)] if i < len(actions) else None,
                            "reward": rewards[-(i + 1)] if i < len(rewards) else 0.0,
                            "outcome": (
                                "success" if rewards[-(i + 1)] > 0 else "failure"
                            ),
                            "task_type": (
                                getattr(agent, "current_task", {}).get("type")
                                if hasattr(agent, "current_task")
                                else None
                            ),
                            "coordination_data": {},
                        }
                        experiences.append(experience)

        return experiences

    def get_collective_learning_reward(self, agent_id: str) -> float:
        """
        Get collective learning reward for an agent from experience sharing.

        Args:
            agent_id: Agent identifier

        Returns:
            Collective learning reward value
        """
        if hasattr(self, "experience_sharing"):
            return self.experience_sharing.get_collective_learning_reward(agent_id)
        return 0.0

    def process_learned_state_representation(self, current_step: int):
        """
        Process learned state representation for better HQ understanding.

        Args:
            current_step: Current simulation step
        """
        if not hasattr(self, "learned_state_representation"):
            return

        # Get enhanced global state
        enhanced_state = self.get_enhanced_global_state()

        # Encode state using learned representations
        representations = self.learned_state_representation.encode_state(enhanced_state)

        # Discover patterns
        patterns = self.learned_state_representation.discover_patterns()

        # Form concepts
        concepts = self.learned_state_representation.form_concepts()

        # Get learned state reward
        learned_state_reward = (
            self.learned_state_representation.get_learned_state_reward()
        )

        # Add learned state reward to HQ reward
        if learned_state_reward > 0:
            if not hasattr(self, "hq_step_rewards"):
                self.hq_step_rewards = []
            self.hq_step_rewards.append(learned_state_reward)

        if utils_config.ENABLE_LOGGING and (patterns or concepts):
            logger.log_msg(
                f"[LEARNED STATE REPRESENTATION] Faction {self.id} discovered {len(patterns)} patterns and formed {len(concepts)} concepts",
                level=logging.DEBUG,
            )

    def get_learned_state_reward(self) -> float:
        """
        Get learned state representation reward for the faction.

        Returns:
            Learned state representation reward value
        """
        if hasattr(self, "learned_state_representation"):
            return self.learned_state_representation.get_learned_state_reward()
        return 0.0

    def process_strategy_composition(self, current_step: int):
        """
        Process strategy composition for dynamic strategy sequencing.

        Args:
            current_step: Current simulation step
        """
        if not hasattr(self, "strategy_composition"):
            return

        # Get enhanced global state
        enhanced_state = self.get_enhanced_global_state()

        # Create goals based on current situation
        goals = self._create_situational_goals(enhanced_state)

        # Process active compositions
        for composition_id, composition in list(
            self.strategy_composition.active_compositions.items()
        ):
            # Execute next step in composition
            next_strategy = self.strategy_composition.execute_composition(
                composition, current_step
            )

            if next_strategy:
                # Execute the strategy
                self._execute_composed_strategy(next_strategy, composition)
            else:
                # Composition completed or failed
                success_score = self.strategy_composition.evaluate_composition_success(
                    composition, current_step
                )
                composition.success_rate = success_score

                # Move to completed compositions
                self.strategy_composition.completed_compositions.append(composition)
                del self.strategy_composition.active_compositions[composition_id]

        # Update goal progress
        for goal in self.strategy_composition.active_goals.values():
            self.strategy_composition.update_goal_progress(
                goal, enhanced_state, current_step
            )

        # Create new compositions for unaddressed goals
        for goal in goals:
            if goal not in self.strategy_composition.active_goals.values():
                composition = self.strategy_composition.compose_strategy(
                    enhanced_state, goal, current_step
                )
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[STRATEGY COMPOSITION] Faction {self.id} created composition "
                        f"{composition.composition_id} for goal {goal.goal_type.value}",
                        level=logging.DEBUG,
                    )

    def _create_situational_goals(self, state: Dict[str, Any]) -> List:
        """
        Create goals based on current situation.

        Args:
            state: Current game state

        Returns:
            List of situational goals
        """
        goals = []

        # Resource acquisition goal
        if state.get("gold_balance", 0) < 200 or state.get("food_balance", 0) < 200:
            goal = self.strategy_composition.create_goal(
                utils_config.StrategyGoalType.RESOURCE_ACQUISITION, priority=0.8
            )
            goals.append(goal)

        # Threat elimination goal
        if state.get("threat_count", 0) > 2:
            goal = self.strategy_composition.create_goal(
                utils_config.StrategyGoalType.THREAT_ELIMINATION, priority=0.9
            )
            goals.append(goal)

        # Agent management goal
        if state.get("friendly_agent_count", 0) < 3:
            goal = self.strategy_composition.create_goal(
                utils_config.StrategyGoalType.AGENT_MANAGEMENT, priority=0.7
            )
            goals.append(goal)

        # Defensive positioning goal
        if state.get("HQ_health", 100) < 50:
            goal = self.strategy_composition.create_goal(
                utils_config.StrategyGoalType.DEFENSIVE_POSITIONING, priority=0.9
            )
            goals.append(goal)

        return goals

    def _execute_composed_strategy(self, strategy: str, composition):
        """
        Execute a strategy from a composition.

        Args:
            strategy: Strategy to execute
            composition: Strategy composition context
        """
        # Execute the strategy using existing parametric system
        if hasattr(self, "current_strategy_parameters"):
            # Use existing parametric execution
            self.perform_HQ_Strategy_parametric(
                strategy, self.current_strategy_parameters
            )
        else:
            # Fallback to basic strategy execution
            if strategy in utils_config.HQ_STRATEGY_OPTIONS:
                self.perform_HQ_Strategy(strategy)

    def get_strategy_composition_reward(self) -> float:
        """
        Get strategy composition reward for the faction.

        Returns:
            Strategy composition reward value
        """
        if hasattr(self, "strategy_composition"):
            return self.strategy_composition.get_composition_reward()
        return 0.0

    def process_meta_learning(self, current_step: int):
        """
        Process meta-learning for strategy discovery.

        Args:
            current_step: Current simulation step
        """
        if not hasattr(self, "meta_learning"):
            return

        # Get enhanced global state
        enhanced_state = self.get_enhanced_global_state()

        # Discover new strategies
        discovered_strategies = self.meta_learning.discover_strategies(
            enhanced_state, current_step
        )

        # Process discovered strategies
        for strategy in discovered_strategies:
            self._integrate_discovered_strategy(strategy, current_step)

        # Update meta-learning progress
        self._update_meta_learning_progress(enhanced_state, current_step)

        if utils_config.ENABLE_LOGGING and discovered_strategies:
            logger.log_msg(
                f"[META-LEARNING] Faction {self.id} discovered {len(discovered_strategies)} new strategies",
                level=logging.INFO,
            )

    def _integrate_discovered_strategy(self, strategy, current_step: int):
        """
        Integrate a discovered strategy into the faction's strategy repertoire.

        Args:
            strategy: Discovered strategy to integrate
            current_step: Current simulation step
        """
        # Add to HQ strategy options if novel enough
        if strategy.novelty_score > 0.7 and strategy.success_rate > 0.6:
            # Create a new strategy option
            new_strategy_name = f"META_{strategy.strategy_name}"

            # Add to HQ strategy options (if not already present)
            if new_strategy_name not in utils_config.HQ_STRATEGY_OPTIONS:
                utils_config.HQ_STRATEGY_OPTIONS.append(new_strategy_name)

                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[META-LEARNING] Added new strategy {new_strategy_name} to HQ options",
                        level=logging.INFO,
                    )

            # Store strategy parameters for use
            if not hasattr(self, "discovered_strategy_parameters"):
                self.discovered_strategy_parameters = {}

            self.discovered_strategy_parameters[new_strategy_name] = strategy.parameters

    def _update_meta_learning_progress(self, state: Dict[str, Any], current_step: int):
        """
        Update meta-learning progress based on current performance.

        Args:
            state: Current game state
            current_step: Current simulation step
        """
        # Calculate current performance metrics
        performance_metrics = {
            "win_rate": state.get("win_rate", 0.5),
            "survival_rate": state.get("survival_rate", 0.5),
            "efficiency_score": state.get("efficiency_score", 0.5),
            "coordination_score": state.get("coordination_score", 0.5),
        }

        # Update meta-learning progress
        for metric, value in performance_metrics.items():
            if hasattr(self.meta_learning, "meta_learning_progress"):
                current_progress = self.meta_learning.meta_learning_progress.get(
                    metric, 0.0
                )
                alpha = 0.1  # Learning rate
                new_progress = (1 - alpha) * current_progress + alpha * value
                self.meta_learning.meta_learning_progress[metric] = new_progress

    def get_meta_learning_reward(self) -> float:
        """
        Get meta-learning reward for the faction.

        Returns:
            Meta-learning reward value
        """
        if hasattr(self, "meta_learning"):
            return self.meta_learning.get_meta_learning_reward()
        return 0.0

    def process_strategy_visualization(self, current_step: int):
        """
        Process strategy visualization and interpretability analysis.

        Args:
            current_step: Current simulation step
        """
        if not hasattr(self, "strategy_visualization"):
            return

        # Get enhanced global state
        enhanced_state = self.get_enhanced_global_state()

        # Update performance metrics for visualization
        self.strategy_visualization.update_performance_metrics(
            enhanced_state, current_step
        )

        # Generate visualizations if needed
        if current_step % self.strategy_visualization.update_frequency == 0:
            self._generate_strategy_visualizations(enhanced_state, current_step)
            # Also create scalar plots using existing system
            self.strategy_visualization.create_scalar_plots(current_step)

        # Generate interpretability analysis
        if current_step % 50 == 0:  # Every 50 steps
            self._generate_interpretability_analysis(enhanced_state, current_step)

        if utils_config.ENABLE_LOGGING and current_step % 100 == 0:
            logger.log_msg(
                f"[VISUALIZATION] Faction {self.id} processed visualization at step {current_step}",
                level=logging.DEBUG,
            )

    def _generate_strategy_visualizations(
        self, state: Dict[str, Any], current_step: int
    ):
        """
        Generate various strategy visualizations.

        Args:
            state: Current game state
            current_step: Current simulation step
        """
        try:
            # Strategy performance plot
            if (
                utils_config.VisualizationType.STRATEGY_PERFORMANCE
                in self.strategy_visualization.active_visualizations
            ):
                fig = self.strategy_visualization.create_strategy_performance_plot()
                self._save_visualization(
                    fig, f"strategy_performance_{current_step}.png"
                )

            # Parameter analysis plot
            if (
                utils_config.VisualizationType.PARAMETER_ANALYSIS
                in self.strategy_visualization.active_visualizations
            ):
                strategy_params = getattr(self, "current_strategy_parameters", {})
                fig = self.strategy_visualization.create_parameter_analysis_plot(
                    strategy_params
                )
                self._save_visualization(fig, f"parameter_analysis_{current_step}.png")

            # Strategy composition flow
            if (
                utils_config.VisualizationType.STRATEGY_COMPOSITION
                in self.strategy_visualization.active_visualizations
                and hasattr(self, "strategy_composition")
            ):
                composition_data = {
                    "active_compositions": getattr(
                        self.strategy_composition, "active_compositions", {}
                    ),
                }
                fig = self.strategy_visualization.create_strategy_composition_flow(
                    composition_data
                )
                self._save_visualization(
                    fig, f"strategy_composition_{current_step}.png"
                )

            # Communication network
            if (
                utils_config.VisualizationType.COMMUNICATION_NETWORKS
                in self.strategy_visualization.active_visualizations
                and hasattr(self, "learned_communication")
            ):
                communication_data = {
                    "agents": [agent.agent_id for agent in self.agents],
                    "communications": getattr(
                        self.learned_communication, "communication_history", {}
                    ),
                }
                fig = self.strategy_visualization.create_communication_network(
                    communication_data
                )
                self._save_visualization(
                    fig, f"communication_network_{current_step}.png"
                )

            # Experience sharing patterns
            if (
                utils_config.VisualizationType.EXPERIENCE_SHARING
                in self.strategy_visualization.active_visualizations
                and hasattr(self, "experience_sharing")
            ):
                sharing_data = {
                    "sharing_frequency": getattr(
                        self.experience_sharing, "sharing_success_rate", {}
                    ),
                    "experience_types": {
                        "successful": 10,
                        "failed": 5,
                        "adaptive": 8,
                    },  # Simulated
                    "learning_success_rate": getattr(
                        self.experience_sharing, "learning_success_rate", {}
                    ),
                    "collective_memory_size": {
                        current_step: len(
                            getattr(self.experience_sharing, "collective_memory", [])
                        )
                    },
                }
                fig = self.strategy_visualization.create_experience_sharing_plot(
                    sharing_data
                )
                self._save_visualization(fig, f"experience_sharing_{current_step}.png")

            # State representation
            if (
                utils_config.VisualizationType.STATE_REPRESENTATION
                in self.strategy_visualization.active_visualizations
                and hasattr(self, "learned_state_representation")
            ):
                state_data = {
                    "representation_components": {
                        "raw": 0.3,
                        "abstract": 0.5,
                        "temporal": 0.4,
                    },  # Simulated
                    "pattern_discovery": {current_step: 5},  # Simulated
                    "concept_formation": {
                        "resource": 0.8,
                        "threat": 0.6,
                        "coordination": 0.7,
                    },  # Simulated
                    "representation_quality": {current_step: 0.75},  # Simulated
                }
                fig = self.strategy_visualization.create_state_representation_plot(
                    state_data
                )
                self._save_visualization(
                    fig, f"state_representation_{current_step}.png"
                )

            # Reward components
            if (
                utils_config.VisualizationType.REWARD_COMPONENTS
                in self.strategy_visualization.active_visualizations
            ):
                reward_data = {
                    "reward_components": {
                        "task_completion": 0.3,
                        "efficiency": 0.25,
                        "coordination": 0.2,
                        "adaptation": 0.15,
                        "survival": 0.1,
                    },
                    "reward_evolution": {current_step: state.get("total_reward", 0.5)},
                    "component_contribution": {
                        "task_completion": {current_step: 0.3},
                        "efficiency": {current_step: 0.25},
                    },
                    "reward_efficiency": {current_step: 0.8},
                }
                fig = self.strategy_visualization.create_reward_components_plot(
                    reward_data
                )
                self._save_visualization(fig, f"reward_components_{current_step}.png")

            # Meta-learning progress
            if (
                utils_config.VisualizationType.META_LEARNING_PROGRESS
                in self.strategy_visualization.active_visualizations
                and hasattr(self, "meta_learning")
            ):
                meta_data = {
                    "discovered_strategies": {
                        current_step: len(
                            getattr(self.meta_learning, "discovered_strategies", {})
                        )
                    },
                    "strategy_quality": {
                        f"strategy_{i}": random.uniform(0.5, 1.0) for i in range(3)
                    },  # Simulated
                    "meta_success_rate": {current_step: 0.7},  # Simulated
                    "discovery_methods": {
                        "pattern_analysis": 0.8,
                        "genetic": 0.7,
                        "reinforcement": 0.6,
                    },  # Simulated
                }
                fig = self.strategy_visualization.create_meta_learning_progress_plot(
                    meta_data
                )
                self._save_visualization(
                    fig, f"meta_learning_progress_{current_step}.png"
                )

        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[VISUALIZATION] Error generating visualizations: {e}",
                    level=logging.WARNING,
                )

    def _generate_interpretability_analysis(
        self, state: Dict[str, Any], current_step: int
    ):
        """
        Generate interpretability analysis for the HQ network.

        Args:
            state: Current game state
            current_step: Current simulation step
        """
        try:
            if hasattr(self, "HQ_Network") and self.HQ_Network is not None:
                # Convert state to tensor
                state_vector = self._state_to_vector(state)
                state_tensor = torch.tensor(
                    state_vector, dtype=torch.float32
                ).unsqueeze(0)

                # Generate interpretability analysis
                interpretability_methods = [
                    utils_config.InterpretabilityMethod.GRADIENT_ATTRIBUTION,
                    utils_config.InterpretabilityMethod.FEATURE_IMPORTANCE,
                ]

                for method in interpretability_methods:
                    result = (
                        self.strategy_visualization.generate_interpretability_analysis(
                            self.HQ_Network, state_tensor, method
                        )
                    )

                    # Store result
                    if (
                        method.value
                        not in self.strategy_visualization.interpretability_results
                    ):
                        self.strategy_visualization.interpretability_results[
                            method.value
                        ] = []

                    self.strategy_visualization.interpretability_results[
                        method.value
                    ].append(result)

                    # Keep only recent results
                    if (
                        len(
                            self.strategy_visualization.interpretability_results[
                                method.value
                            ]
                        )
                        > 10
                    ):
                        self.strategy_visualization.interpretability_results[
                            method.value
                        ] = self.strategy_visualization.interpretability_results[
                            method.value
                        ][
                            -10:
                        ]

        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[VISUALIZATION] Error generating interpretability analysis: {e}",
                    level=logging.WARNING,
                )

    def _save_visualization(self, fig: plt.Figure, filename: str):
        """
        Save visualization figure using integrated matplotlib plotter.

        Args:
            fig: matplotlib Figure object
            filename: Filename to save
        """
        try:
            # Use integrated matplotlib plotter system
            if hasattr(self, 'strategy_visualization') and hasattr(self.strategy_visualization, 'matplotlib_plotter'):
                self.strategy_visualization.matplotlib_plotter.update_image_plot(
                    name=filename.replace('.png', ''),
                    fig=fig,
                    tensorboard_logger=getattr(self.strategy_visualization, 'tensorboard_logger', None),
                    step=self.current_step
                )
            else:
                # Fallback to direct save
                viz_dir = "VISUALS/STRATEGY_ANALYSIS"
                os.makedirs(viz_dir, exist_ok=True)
                filepath = os.path.join(viz_dir, filename)
                fig.savefig(filepath, dpi=100, bbox_inches="tight")
                plt.close(fig)

            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[VISUALIZATION] Saved visualization: {filename}",
                    level=logging.DEBUG,
                )

        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[VISUALIZATION] Error saving visualization: {e}",
                    level=logging.WARNING,
                )

    def _state_to_vector(self, state: Dict[str, Any]) -> List[float]:
        """Convert state dictionary to vector representation for interpretability."""
        # Extract key state components
        vector = [
            state.get("HQ_health", 100.0) / 100.0,
            state.get("gold_balance", 0.0) / 1000.0,
            state.get("food_balance", 0.0) / 1000.0,
            state.get("resource_count", 0.0) / 100.0,
            state.get("threat_count", 0.0) / 10.0,
            state.get("friendly_agent_count", 0.0) / 10.0,
            state.get("enemy_agent_count", 0.0) / 10.0,
            state.get("gatherer_count", 0.0) / 10.0,
            state.get("peacekeeper_count", 0.0) / 10.0,
            state.get("agent_density", 0.0) / 10.0,
        ]

        # Ensure vector has correct size
        while len(vector) < self.state_size:
            vector.append(0.0)

        return vector[: self.state_size]

    def get_strategy_visualization_reward(self) -> float:
        """
        Get strategy visualization reward for the faction.

        Returns:
            Strategy visualization reward value
        """
        if hasattr(self, "strategy_visualization"):
            return self.strategy_visualization.get_visualization_reward()
        return 0.0

    def _evaluate_strategy_execution_success(self) -> bool:
        """
        Evaluate whether the current strategy was successfully executed.

        Returns:
            bool: True if strategy execution was successful
        """
        if not self.current_strategy:
            return False

        # Check if strategy-specific success criteria were met
        if self.current_strategy == "COLLECT_GOLD":
            # Success if we have gold balance or agents are actively mining
            return self.gold_balance > 0 or any(
                agent.current_task == "gather"
                and agent.current_task_state == utils_config.TaskState.ONGOING
                for agent in self.agents
            )
        elif self.current_strategy == "ATTACK_THREATS":
            # Success if threats were eliminated or agents are actively attacking
            return (
                hasattr(self, "threats_eliminated") and self.threats_eliminated > 0
            ) or any(
                agent.current_task == "eliminate"
                and agent.current_task_state == utils_config.TaskState.ONGOING
                for agent in self.agents
            )
        elif self.current_strategy == "DEFEND_HQ":
            # Success if HQ is defended (no threats nearby or agents defending)
            return any(
                agent.current_task == "defend"
                and agent.current_task_state == utils_config.TaskState.ONGOING
                for agent in self.agents
            )
        elif self.current_strategy in ["RECRUIT_PEACEKEEPER", "RECRUIT_GATHERER"]:
            # Success if recruitment was successful
            return len(self.agents) > 0  # Simplified: success if we have agents
        else:
            # Default: consider successful if no major failures
            return True

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

    def take_hq_damage(self, damage: int):
        """
        Apply damage to HQ health.

        Args:
            damage: Amount of damage to apply

        Returns:
            bool: True if HQ is destroyed, False otherwise
        """
        if self.home_base["is_destroyed"]:
            return True  # Already destroyed

        self.home_base["health"] = max(0, self.home_base["health"] - damage)

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[HQ DAMAGE] Faction {self.id} HQ took {damage} damage. Health: {self.home_base['health']}/{self.home_base['max_health']}",
                level=logging.WARNING,
            )

        if self.home_base["health"] <= 0:
            self.home_base["is_destroyed"] = True
            self.home_base["health"] = 0

            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[HQ DESTROYED] Faction {self.id} HQ has been destroyed!",
                    level=logging.CRITICAL,
                )
            return True  # HQ is now destroyed

        # Update global state
        self.global_state["HQ_health"] = (
            self.home_base["health"] / self.home_base["max_health"]
        )
        return False  # HQ still alive

    def get_enemy_hqs(self):
        """
        Get all enemy HQs from the game manager.

        Returns:
            list: List of enemy HQ dictionaries
        """
        if not hasattr(self.game_manager, "faction_manager"):
            return []

        enemy_hqs = []
        for faction in self.game_manager.faction_manager.factions:
            if faction.id != self.id and not faction.home_base["is_destroyed"]:
                enemy_hqs.append(
                    {
                        "faction_id": faction.id,
                        "faction": faction.id,
                        "type": "Faction HQ",
                        "position": faction.home_base["position"],
                        "health": faction.home_base["health"],
                        "max_health": faction.home_base["max_health"],
                    }
                )

        return enemy_hqs

    def is_hq_destroyed(self) -> bool:
        """Check if this HQ is destroyed."""
        return self.home_base["is_destroyed"]

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

            # Add HQ health status to awareness
            if "health" in self.home_base and "max_health" in self.home_base:
                enhanced_state["hq_health_status"] = (
                    self.home_base["health"] / self.home_base["max_health"]
                    if self.home_base["max_health"] > 0
                    else 0.0
                )
                enhanced_state["hq_is_critical"] = (
                    1.0 if enhanced_state["hq_health_status"] < 0.25 else 0.0
                )  # Critical at < 25%
                enhanced_state["hq_is_damaged"] = (
                    1.0 if enhanced_state["hq_health_status"] < 0.75 else 0.0
                )  # Damaged at < 75%
            else:
                enhanced_state["hq_health_status"] = 1.0
                enhanced_state["hq_is_critical"] = 0.0
                enhanced_state["hq_is_damaged"] = 0.0

            # Check if HQ is under direct attack
            hq_under_attack = 0.0
            hq_x, hq_y = self.home_base["position"]
            for threat in self.global_state.get("threats", []):
                threat_pos = threat.get("location", (-1, -1))
                dist_sq = (threat_pos[0] - hq_x) ** 2 + (threat_pos[1] - hq_y) ** 2
                if dist_sq <= (50**2):  # Within 50 pixels of HQ
                    hq_under_attack = 1.0
                    break
            enhanced_state["hq_under_attack"] = hq_under_attack

            # Add threat reporting awareness - HQ knows when its troops are reporting enemies
            recent_reports_window = 20  # Steps to look back
            recent_threat_reports = [
                r
                for r in self.recent_threat_reports
                if self.current_step - r.get("step", 0) <= recent_reports_window
            ]
            enhanced_state["threat_reports_recent"] = min(
                len(recent_threat_reports) / 5.0, 1.0
            )  # 0-1 normalized
            enhanced_state["has_known_threats"] = (
                1.0 if self.global_state.get("threats", []) else 0.0
            )

            # Calculate threat freshness (how old is the threat intel?)
            if self.global_state.get("threats"):
                oldest_threat_age = 0
                for threat in self.global_state.get("threats", []):
                    last_seen = threat.get("last_seen", 0)
                    age = self.current_step - last_seen if last_seen else 0
                    oldest_threat_age = max(oldest_threat_age, age)
                # Normalize: 0 = very fresh (< 10 steps old), 1.0 = very stale (> 100 steps old)
                enhanced_state["threat_intel_freshness"] = max(
                    0.0, min(1.0, (oldest_threat_age - 10) / 90.0)
                )
            else:
                enhanced_state["threat_intel_freshness"] = (
                    1.0  # No threats = stale intel
                )

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

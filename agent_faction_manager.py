
# Manages factions and their agents.
from utils_config import (
     FACTON_COUNT, 
    INITAL_GATHERER_COUNT, 
    INITAL_PEACEKEEPER_COUNT, 
    CELL_SIZE, 
    TaskState, 
    AgentIDStruc, 
    create_task, 
    DEF_AGENT_STATE_SIZE,
    Agent_Interact_Range,
    STATE_FEATURES_MAP,
    NETWORK_TYPE_MAPPING,
    WORLD_HEIGHT,
    WORLD_WIDTH,
    LOGGING_ENABLED,
    HQ_STRATEGY_OPTIONS)

from agent_factions import Faction
from utils_helpers import generate_random_colour


import logging
from utils_logger import Logger






logger = Logger(log_file="agent_faction_manager.txt", log_level=logging.DEBUG)




#    _____ _    ____ _____ ___ ___  _   _   __  __    _    _   _    _    ____ _____ ____     ____ _        _    ____ ____  
#   |  ___/ \  / ___|_   _|_ _/ _ \| \ | | |  \/  |  / \  | \ | |  / \  / ___| ____|  _ \   / ___| |      / \  / ___/ ___| 
#   | |_ / _ \| |     | |  | | | | |  \| | | |\/| | / _ \ |  \| | / _ \| |  _|  _| | |_) | | |   | |     / _ \ \___ \___ \ 
#   |  _/ ___ \ |___  | |  | | |_| | |\  | | |  | |/ ___ \| |\  |/ ___ \ |_| | |___|  _ <  | |___| |___ / ___ \ ___) |__) |
#   |_|/_/   \_\____| |_| |___\___/|_| \_| |_|  |_/_/   \_\_| \_/_/   \_\____|_____|_| \_\  \____|_____/_/   \_\____/____/ 
#                                                                                                                          




class FactionManager:
    def __init__(self):
        self.factions = []
        self.faction_counter = 1  # Initialise a counter for unique IDs
        if LOGGING_ENABLED: logger.log_msg("FactionManager initialised.", level=logging.INFO)

    def update(self, resource_manager, agents):
        for faction in self.factions:
            if not isinstance(faction, Faction):
                logger.log_msg(f"[ERROR] Invalid faction: {faction}", level=logging.ERROR)
                continue
            if faction.network is None:
                logger.log_msg(f"[ERROR] Faction {faction.id} has no network.", level=logging.ERROR)
                continue

            faction.update(resource_manager, agents)






    def reset_factions(self, faction_count, resource_manager, agents, game_manager,
                   state_size=DEF_AGENT_STATE_SIZE, action_size=10, 
                   role_size=5, local_state_size=10, global_state_size=15, 
                   network_type="HQNetwork"):
        """
        Fully reset the list of factions and assign agents.
        """
        self.factions.clear()  #  Ensure previous factions are removed
        if LOGGING_ENABLED: logger.log_msg("[INFO] Resetting factions...", level=logging.INFO)

        for i in range(faction_count):
            name = f"Faction {i + 1}"
            colour = generate_random_colour()

            if LOGGING_ENABLED: logger.log_msg(f"[DEBUG] Creating {name} with network type: {network_type}", level=logging.DEBUG)

            try:
                new_faction = Faction(
                    name=name,
                    colour= colour,
                    id=i + 1,
                    resource_manager=resource_manager,
                    game_manager=game_manager,  #  Pass GameManager to Faction
                    agents=[],  #  Agents will be assigned after initialisation
                    state_size=state_size,
                    action_size=action_size,
                    role_size=role_size,
                    local_state_size=local_state_size,
                    global_state_size=global_state_size,
                    network_type=network_type
                )

                if new_faction.network is None:
                    if LOGGING_ENABLED: logger.log_msg(f"[ERROR] Faction {new_faction.id} failed to Initialise network.", level=logging.ERROR)
                    raise RuntimeError(f"[ERROR] Faction {new_faction.id} failed to Initialise network.")

                self.factions.append(new_faction)
                if LOGGING_ENABLED: logger.log_msg(f"[INFO] Successfully created {name} (ID: {new_faction.id})", level=logging.INFO)

            except Exception as e:
                if LOGGING_ENABLED: logger.log_msg(f"[ERROR] Failed to create {name}: {e}", level=logging.ERROR)
                import traceback
                if LOGGING_ENABLED: logger.log_msg(traceback.format_exc(), level=logging.ERROR)

        #  Assign agents to their factions after all factions are created
        for agent in agents:
            for faction in self.factions:
                if agent.faction.id == faction.id:
                    faction.agents.append(agent)

        #  Debug log to verify agents are assigned correctly
        for faction in self.factions:
            if LOGGING_ENABLED: logger.log_msg(f"[DEBUG] {faction.name} Initialised with {len(faction.agents)} agents.", level=logging.INFO)



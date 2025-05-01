

"""
THis file should control the episodic win rules and conditions

"""
"""Common Imports"""
from SHARED.core_imports import *
"""File Specific Imports"""
import UTILITIES.utils_config as utils_config



FACTION_GOAL = {
    'gold_collection': 1000,
    'food_collection': 1400
}

# Number of steps without meaningful activity to trigger a dynamic event
MAX_STALE_STEPS = 400


def check_victory(factions):
    """
    Check victory conditions for factions.
    A faction wins if:
    - It reaches the resource goal (gold or food).
    - It is the last faction standing.
    """
    active_factions = [f for f in factions if len(f.agents) > 0]

    # Check resource-based victory
    for faction in factions:
        if faction.gold_balance >= FACTION_GOAL['gold_collection'] or \
           faction.food_balance >= FACTION_GOAL['food_collection']:
            print("Resource victory!")
            return faction  # Return the winning faction

    # Check last faction standing
    if len(active_factions) == 1:
        print("Last faction standing!")
        return active_factions[0]

    # Check draw (no active factions)
    if len(active_factions) == 0:
        print("Draw!")
        return None  # Draw/stalemate

    return None  # No winner yet



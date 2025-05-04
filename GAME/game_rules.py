

"""
THis file should control the episodic win rules and conditions

"""
"""Common Imports"""
from SHARED.core_imports import *
"""File Specific Imports"""
import UTILITIES.utils_config as utils_config
from ENVIRONMENT.env_resources import AppleTree, GoldLump



FACTION_GOAL = {
    'gold_collection': 400,
    'food_collection': 400
}

def calculate_resource_victory_targets(resources, faction_count=1, target_ratio=utils_config.RESOURCE_VICTORY_TARGET_RATIO):
    """
    Dynamically calculates resource-based victory conditions based on spawned resources.

    Args:
        resources (list): The list of all resource objects (e.g., AppleTree, GoldLump).
        faction_count (int): Number of factions (for logging only).
        target_ratio (float): Fraction of total resources a faction must collect to win.

    Updates:
        FACTION_GOAL with new global target values.
    """
    total_gold = sum(res.quantity for res in resources if isinstance(res, GoldLump))
    total_food = sum(res.quantity for res in resources if isinstance(res, AppleTree))

    # Apply global percentage, no division by faction count
    gold_target = int(total_gold * target_ratio)
    food_target = int(total_food * target_ratio)

    FACTION_GOAL['gold_collection'] = gold_target
    FACTION_GOAL['food_collection'] = food_target

    print(f"""[Victory Targets]
    - Gold Collection Target: {gold_target} of {total_gold} total ({(gold_target / total_gold * 100):.1f}%)
    - Food Collection Target: {food_target} of {total_food} total ({(food_target / total_food * 100):.1f}%)
    \n""")




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
            print("\n\nResource victory!\n\n")
            return faction  # Return the winning faction

    # Check last faction standing
    if len(active_factions) == 1:
        print("\n\nLast faction standing!\n\n")
        return active_factions[0]

    # Check draw (no active factions)
    if len(active_factions) == 0:
        print("\n\nDraw!\n\n")
        return None  # Draw/stalemate

    return None  # No winner yet



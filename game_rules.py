


"""
THis file should control the episodic win rules and conditions

"""





import random
RESOURCE_GATHER_RATE = {
    'apple_tree': 5,
    'gold_lump': 2
}

FACTION_GOAL = {
    'gold_collection': 100,
    'food_collection': 500
}

MAX_STALE_STEPS = 400  # Number of steps without meaningful activity to trigger a dynamic event

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
            print ("Resource victory!")
            return faction  # Return the winning faction

    # Check last faction standing
    if len(active_factions) == 1:
        print ("Last faction standing!")
        return active_factions[0]

    # Check draw (no active factions)
    if len(active_factions) == 0:
        print ("Draw!")
        return None  # Draw/stalemate

    return None  # No winner yet




def resolve_conflict(agent1, agent2):
    """
    Resolve conflicts between two agents.
    Placeholder for custom conflict resolution logic.
    """
    if agent1.role == "peacekeeper" and agent2.role == "peacekeeper":
        # Both agents take damage
        agent1.Health -= 10
        agent2.Health -= 10
    else:
        # Simplified: One agent loses health randomly
        loser = random.choice([agent1, agent2])
        loser.Health -= 20

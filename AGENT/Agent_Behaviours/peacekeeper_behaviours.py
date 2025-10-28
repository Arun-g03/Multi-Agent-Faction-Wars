"""Peacekeeper Behaviours - Defense and combat focused agent behaviors.

This module contains actions specific to Peacekeeper agents:
- patrol(): Move toward nearest threats
- eliminate_threat(): Attack and eliminate enemy threats

"""

import UTILITIES.utils_config as utils_config
from SHARED.core_imports import *

logger = Logger(log_file="behavior_log.txt", log_level=logging.DEBUG)


class PeacekeeperBehavioursMixin:
    """Mixin class providing peacekeeper-specific behaviors."""

    def patrol(self):
        """
        Patrol towards the nearest threat.
        Returns ONGOING if moving toward the threat, or FAILURE if no threats are found.
        """
        threats = self.agent.faction.provide_state().get("threats", [])

        # Filter out friendly threats before calling find_closest_actor()
        valid_threats = [
            t for t in threats if t.get("faction") != self.agent.faction.id
        ]

        # Call find_closest_actor() correctly without an 'exclude' argument
        threat = find_closest_actor(
            valid_threats, entity_type="threat", requester=self.agent
        )

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
                        level=logging.INFO,
                    )
                return utils_config.TaskState.ONGOING
            else:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"{self.agent.role} could not execute movement action '{action}' while patrolling towards threat ID {threat_id}.",
                        level=logging.WARNING,
                    )
                return (
                    utils_config.TaskState.FAILURE
                )  # Penalise for missing movement action
        else:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} found no threats to patrol towards.",
                    level=logging.WARNING,
                )
            return utils_config.TaskState.FAILURE  # Failure when no threats are found

    def eliminate_threat(self, agents):
        """
        Attempt to eliminate the assigned threat (agent or HQ).
        If it's not in combat range, opportunistically attack any nearby enemy within combat range.
        Returns SUCCESS if threat is eliminated, ONGOING if attacking, or FAILURE if nothing in range.
        """
        task = self.agent.current_task
        if not task:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} has no valid task assigned for elimination.",
                    level=logging.WARNING,
                )
            return utils_config.TaskState.FAILURE

        threat = task.get("target")
        if not threat or "position" not in threat:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} could not find a valid threat in the task.",
                    level=logging.WARNING,
                )
            return utils_config.TaskState.FAILURE

        assigned_position = threat["position"]
        assigned_id = threat.get("id", None)
        threat_type = threat.get("type", "")

        # Step 1: Try to attack the assigned threat if within combat range
        if self.agent.is_near(
            assigned_position,
            threshold=utils_config.Agent_Interact_Range * utils_config.CELL_SIZE,
        ):
            # Check if target is an HQ
            if "HQ" in threat_type or "Faction HQ" in threat_type:
                target_hq = None
                if hasattr(self.agent.faction, "game_manager") and hasattr(
                    self.agent.faction.game_manager, "faction_manager"
                ):
                    for (
                        faction
                    ) in self.agent.faction.game_manager.faction_manager.factions:
                        if (
                            hasattr(assigned_id, "faction_id")
                            and faction.id == assigned_id.faction_id
                        ) or (
                            isinstance(assigned_id, int) and faction.id == assigned_id
                        ):
                            target_hq = faction
                            break

                if target_hq and not target_hq.home_base["is_destroyed"]:
                    # Attack the enemy HQ
                    self.event_manager.trigger_attack_animation(
                        position=target_hq.home_base["position"], duration=200
                    )

                    damage = 5  # HQ takes more damage than agents
                    is_destroyed = target_hq.take_hq_damage(damage)

                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"{self.agent.role} attacked enemy HQ (Faction {target_hq.id}) at {assigned_position}. HQ Health: {target_hq.home_base['health']}/{target_hq.home_base['max_health']}",
                            level=logging.INFO,
                        )
                    print(
                        f"{self.agent.role} attacked enemy HQ (Faction {target_hq.id}) at {assigned_position}. HQ Health: {target_hq.home_base['health']}/{target_hq.home_base['max_health']}"
                    )

                    if is_destroyed:
                        self.report_threat_eliminated(threat)
                        return utils_config.TaskState.SUCCESS
                    return utils_config.TaskState.ONGOING

            # Otherwise, treat as an agent
            target_agent = next((a for a in agents if a.agent_id == assigned_id), None)
            if target_agent and target_agent.faction.id != self.agent.faction.id:
                self.event_manager.trigger_attack_animation(
                    position=(target_agent.x, target_agent.y), duration=200
                )
                target_agent.Health -= 10
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"{self.agent.role} attacked assigned threat {target_agent.role} (ID: {assigned_id}) at {assigned_position}. Health is now {target_agent.Health}.",
                        level=logging.INFO,
                    )
                print(
                    f"{self.agent.role} attacked assigned threat {target_agent.role} (ID: {assigned_id}) at {assigned_position}. Health is now {target_agent.Health}."
                )
                if target_agent.Health <= 0:
                    self.report_threat_eliminated(threat)
                    return utils_config.TaskState.SUCCESS
                return utils_config.TaskState.ONGOING
        else:

            # Step 2: Attack any other enemy agent within combat range
            for enemy in agents:
                if enemy.faction.id != self.agent.faction.id and self.agent.is_near(
                    (enemy.x, enemy.y),
                    threshold=utils_config.Agent_Interact_Range
                    * utils_config.CELL_SIZE,
                ):
                    self.event_manager.trigger_attack_animation(
                        position=(enemy.x, enemy.y), duration=200
                    )
                    enemy.Health -= 10
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"{self.agent.role} attacked {enemy.role} (ID: {enemy.agent_id}) at ({enemy.x}, {enemy.y}). Health now {enemy.Health}.",
                            level=logging.INFO,
                        )
                    print(
                        f"{self.agent.role} attacked {enemy.role} (ID: {enemy.agent_id}) at ({enemy.x}, {enemy.y}). Health now {enemy.Health}."
                    )
                    if enemy.Health <= 0:
                        if utils_config.ENABLE_LOGGING:
                            logger.log_msg(
                                f"{self.agent.role} eliminated nearby  {enemy.role}.",
                                level=logging.INFO,
                            )
                        print(f"{self.agent.role} eliminated nearby  {enemy.role}.")
                    return utils_config.TaskState.ONGOING

            # Step 3: Attack any nearby enemy HQ
            if hasattr(self.agent.faction, "get_enemy_hqs"):
                enemy_hqs = self.agent.faction.get_enemy_hqs()
                for hq in enemy_hqs:
                    if self.agent.is_near(
                        hq["position"],
                        threshold=utils_config.Agent_Interact_Range
                        * utils_config.CELL_SIZE,
                    ):
                        self.event_manager.trigger_attack_animation(
                            position=hq["position"], duration=200
                        )

                        # Find the target faction and damage its HQ
                        for (
                            faction
                        ) in self.agent.faction.game_manager.faction_manager.factions:
                            if faction.id == hq["faction_id"]:
                                damage = 5
                                is_destroyed = faction.take_hq_damage(damage)

                                if utils_config.ENABLE_LOGGING:
                                    logger.log_msg(
                                        f"{self.agent.role} attacked enemy HQ (Faction {faction.id}). HQ Health: {faction.home_base['health']}/{faction.home_base['max_health']}",
                                        level=logging.INFO,
                                    )
                                print(
                                    f"{self.agent.role} attacked enemy HQ (Faction {faction.id}). HQ Health: {faction.home_base['health']}/{faction.home_base['max_health']}"
                                )

                                if is_destroyed:
                                    if utils_config.ENABLE_LOGGING:
                                        logger.log_msg(
                                            f"{self.agent.role} destroyed enemy HQ (Faction {faction.id})!",
                                            level=logging.CRITICAL,
                                        )
                                    print(
                                        f"{self.agent.role} destroyed enemy HQ (Faction {faction.id})!"
                                    )
                                return utils_config.TaskState.ONGOING

        # Nothing in range — fail the task this step
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"{self.agent.role} could not reach assigned threat and found no enemies in attack range.",
                level=logging.INFO,
            )
        return utils_config.TaskState.FAILURE

    def report_threat_eliminated(self, threat):
        """
        Mark a threat as resolved and remove it from the global state.

        Args:
            threat (dict): The threat dictionary to be marked as resolved.
        """
        if not isinstance(threat, dict):
            logger.log_msg(
                f"Invalid threat format passed to report_threat_eliminated: {threat}",
                level=logging.WARNING,
            )
            return

        # Mark the threat as inactive in the global state based on the unique ID
        for global_threat in self.agent.faction.global_state.get("threats", []):
            if global_threat.get("id") == threat.get("id"):  # Compare using unique ID
                global_threat["is_active"] = False
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"{self.agent.role} reported threat ID {threat['id']} at {threat.get('location')} as eliminated.",
                        level=logging.INFO,
                    )
                break  # Stop iteration once the matching threat is found
        else:
            # Log if the threat was not found in the global state
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"Threat ID {threat.get('id')} not found in global state during elimination report.",
                    level=logging.WARNING,
                )

        # Clean up resolved threats
        self.clean_resolved_threats()

    def clean_resolved_threats(self):
        """
        Remove resolved threats from the global state.
        """
        before_cleanup = len(self.agent.faction.global_state["threats"])
        self.agent.faction.global_state["threats"] = [
            threat
            for threat in self.agent.faction.global_state.get("threats", [])
            if threat.get("is_active", True)  # Keep only active threats
        ]
        after_cleanup = len(self.agent.faction.global_state["threats"])
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"{self.agent.role} cleaned resolved threats. Before: {before_cleanup}, After: {after_cleanup}.",
                level=logging.INFO,
            )

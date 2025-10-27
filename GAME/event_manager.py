"""
Event management system for handling game events such as animations and dynamic events.

"""

"""Common Imports"""
from SHARED.core_imports import *

"""File Specific Imports"""
import UTILITIES.utils_config as utils_config


class EventManager:
    def __init__(self, resource_manager, faction_manager, agents, renderer, camera):
        """
        Initialise the EventManager.

        :param resource_manager: Resource manager to handle resource operations.
        :param faction_manager: Faction manager to handle faction operations.
        :param agents: List of agents in the game.
        :param renderer: Renderer for handling visual elements.
        """
        self.events = []
        self.resource_manager = resource_manager
        self.faction_manager = faction_manager
        self.agents = agents
        self.renderer = renderer  # Renderer to draw animations
        self.camera = camera  # Camera to position the animation

    def add_event(self, event_type, data=None):
        """
        Add an event to the queue.

        :param event_type: The type of event (e.g., 'attack_animation', 'dynamic_event').
        :param data: Additional data for the event (e.g., position or event-specific information).
        """
        self.events.append({"type": event_type, "data": data})

    def get_events(self):
        """
        Retrieve and clear the event queue.

        :return: A list of all pending events.
        """
        events = self.events[:]
        self.events.clear()
        return events

    def trigger_attack_animation(self, position, duration=500):
        """
        Trigger an attack animation at a given grid position.

        :param grid_position: Tuple (grid_x, grid_y) for the animation's grid coordinates.
        :param duration: Duration of the animation in milliseconds.
        """
        # Convert grid position to world position
        world_x = position[0]
        world_y = position[1]
        world_position = (world_x, world_y)

        # Convert world position to screen position using the camera
        screen_x, screen_y = self.camera.apply(world_position)

        # Debug with terrain grid context

        # print(f"[Event Manager] Attack Triggered @ Grid Pos: {position}, World Pos: {world_position}, Screen Pos: ({screen_x}, {screen_y})")

        # Trigger animation at calculated screen position
        self.add_event(
            "attack_animation", {"position": world_position, "duration": duration}
        )

    def trigger_dynamic_event(self, max_trees=10, max_gold_lumps=5, health_penalty=10):
        """
        Trigger a dynamic event to redistribute resources and penalise health.
        """
        print(
            "Triggering Dynamic Event: Redistributing resources and applying health penalty!"
        )

        # Clean faction global states
        for faction in self.faction_manager.factions:
            faction.clean_global_state()

        # Generate new resources
        self.resource_manager.generate_resources(
            add_trees=max_trees, add_gold_lumps=max_gold_lumps
        )

        # Apply health penalty to all agents
        for agent in self.agents:
            agent.Health -= health_penalty
            if agent.Health <= 0:
                print(f"Agent {agent.agent_id} died due to dynamic event.")

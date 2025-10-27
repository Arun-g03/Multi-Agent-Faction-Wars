"""Peacekeeper Agent Type

This module defines the Peacekeeper agent, a defensive/combat-focused agent
that patrols, defends, and eliminates threats.

"""

from SHARED.core_imports import *
import UTILITIES.utils_config as utils_config
from AGENT.agent_base import BaseAgent
from RENDER.Game_Renderer import get_font

logger = Logger(log_file="behavior_log.txt", log_level=logging.DEBUG)


class Peacekeeper(BaseAgent):
    """Peacekeeper Agent - Defensive/Combat role.
    
    Peacekeepers are designed to:
    - Patrol and defend the faction's territory
    - Eliminate threats (enemy agents, enemy HQs)
    - Guard the home base
    - Respond to enemy incursions
    """
    
    def __init__(
            self,
            x,
            y,
            faction,
            base_sprite_path,
            terrain,
            agents,
            resource_manager,
            agent_id,
            role_actions,
            communication_system,
            state_size=utils_config.DEF_AGENT_STATE_SIZE,
            event_manager=None,
            mode="train",
            network_type="PPOModel"):
        """
        Initialize a Peacekeeper agent.
        
        Args:
            x: Initial x position
            y: Initial y position
            faction: Faction this agent belongs to
            base_sprite_path: Path to sprite image
            terrain: Terrain reference
            agents: List of all agents
            resource_manager: Resource manager reference
            agent_id: Unique agent identifier
            role_actions: Dictionary of role-specific actions
            communication_system: Communication system reference
            state_size: Size of the state vector
            event_manager: Event manager reference
            mode: Training mode ('train' or 'evaluate')
            network_type: Type of neural network to use
        """
        super().__init__(
            x=x,
            y=y,
            role="peacekeeper",
            faction=faction,
            terrain=terrain,
            resource_manager=resource_manager,
            role_actions=role_actions,
            agent_id=agent_id,
            communication_system=communication_system,
            event_manager=event_manager,
            mode=mode,
            network_type=network_type
        )
        
        # Load and configure sprite
        if not utils_config.HEADLESS_MODE:
            self.base_sprite = pygame.image.load(
                base_sprite_path).convert_alpha()
            sprite_size = int(utils_config.SCREEN_HEIGHT *
                              utils_config.AGENT_SCALE_FACTOR)
            self.base_sprite = pygame.transform.scale(
                self.base_sprite, (sprite_size, sprite_size))
            self.sprite = tint_sprite(
                self.base_sprite, faction.colour) if faction and hasattr(
                faction, 'colour') else self.base_sprite
        
        # Initialize font for rendering
        self.font = get_font(24)
        
        # Peacekeeper-specific attributes
        self.known_threats = []  # Track known enemy threats


def tint_sprite(sprite, tint_colour):
    """
    Tint a sprite with a given colour while preserving alpha values.
    
    Args:
        sprite: The sprite to tint
        tint_colour: The tint colour (RGB tuple)
        
    Returns:
        The tinted sprite
    """
    tinted_sprite = sprite.copy()
    tinted_sprite.fill(tint_colour, special_flags=pygame.BLEND_RGB_MULT)
    return tinted_sprite


# Project Architecture

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
   - [Agent System](#1-agent-system-agent)
   - [Neural Network System](#2-neural-network-system-neural_network)
   - [Environment System](#3-environment-system-environment)
   - [Game Management](#4-game-management-game)
   - [Rendering System](#5-rendering-system-render)
   - [Utilities](#6-utilities-utilities)
4. [Data Flow](#data-flow)
   - [Agent Action Cycle](#agent-action-cycle)
   - [HQ Decision Cycle](#hq-decision-cycle)
   - [Rendering Cycle](#rendering-cycle)
5. [Key Design Patterns](#key-design-patterns)
6. [Neural Network Architecture](#neural-network-architecture)
7. [Performance Characteristics](#performance-characteristics)
8. [Extension Points](#extension-points)

---

## Overview

Multi-Agent Faction Wars uses a modular architecture with clear separation of concerns. The project is organized into several key components that interact to create an emergent multi-agent system.

## Project Structure

```
Multi-Agent-Faction-Wars/
├── AGENT/                    # Agent logic and behaviors
│   ├── agent_base.py        # Base agent class with core mechanics
│   ├── agent_factions.py     # Faction management and HQ strategies
│   ├── agent_behaviours.py   # Reward assignment and task management
│   ├── Agent_Behaviours/     # Role-specific behaviors (gatherer, peacekeeper)
│   └── Agent_Types/          # Role-specific agent implementations
├── NEURAL_NETWORK/           # Neural network implementations
│   ├── PPO_Agent_Network.py # PPO reinforcement learning for agents
│   ├── HQ_Network.py         # HQ strategy selection network
│   ├── DQN_Model.py         # DQN implementation (future work)
│   └── AttentionLayer.py     # Custom attention mechanism
├── ENVIRONMENT/              # Game world and resources
│   ├── env_terrain.py       # Terrain generation and rendering
│   ├── env_resources.py     # Resource management
│   └── Resources/           # Individual resource types (AppleTree, GoldLump)
├── GAME/                     # Game logic and management
│   ├── game_manager.py      # Main game loop and orchestration
│   ├── game_rules.py        # Victory conditions and rules
│   ├── camera.py            # Camera system for viewport
│   └── event_manager.py     # Event system for animations
├── RENDER/                   # Rendering and UI
│   ├── Game_Renderer.py    # Main game rendering
│   ├── MainMenu_Renderer.py # Main menu UI
│   ├── Settings_Renderer.py # Settings menu UI
│   ├── PauseMenu_Renderer.py # Pause menu UI
│   └── Credits_Renderer.py  # Credits screen
├── UTILITIES/                # Utilities and configuration
│   ├── utils_config.py      # Global configuration constants
│   ├── utils_logger.py      # Custom logging system
│   ├── utils_helpers.py     # Helper functions and profiling
│   └── settings_manager.py  # Persistent settings management
└── SHARED/                   # Shared imports and common code
```

## Core Components {#core-components}

### 1. Agent System (`AGENT/`) {#agent-system}

The agent system is the heart of the simulation, implementing individual agents, their behaviors, and faction-level decision-making.

#### Base Agent (`agent_base.py`)
- Defines core agent mechanics: movement, health, energy
- Handles observation and state generation for neural networks
- Manages threat detection and task assignment
- Implements communication with HQ

#### Faction Management (`agent_factions.py`)
- Manages HQ (Headquarters) for each faction
- Implements HQ strategy selection using neural networks
- Handles resource collection and agent recruitment
- Provides faction-level state to HQ networks
- Manages threat intelligence and global state

#### Behaviors (`agent_behaviours.py` & `Agent_Behaviours/`)
- Assigns rewards to agents based on task outcomes
- Implements role-specific behaviors:
  - **Gatherers**: Collect food and gold, plant trees
  - **Peacekeepers**: Defend and eliminate threats
- Provides context-aware rewards (proximity bonuses, survival bonuses)

### 2. Neural Network System (`NEURAL_NETWORK/`) {#neural-network-system}

The neural network system handles learning for both individual agents and HQ strategic decisions.

#### PPO Agent Network (`PPO_Agent_Network.py`)
- Implements Proximal Policy Optimization (PPO) for agent learning
- Handles action selection and policy updates
- Manages experience replay and mini-batch training
- Includes memory overflow protection and gradient clipping

#### HQ Network (`HQ_Network.py`)
- Implements hierarchical decision-making for HQ
- Uses attention mechanisms to focus on important state information
- Selects strategies based on faction state (resources, threats, agents)
- Features dynamic network updates to handle varying input sizes

#### Attention Layer (`AttentionLayer.py`)
- Custom attention mechanism for processing complex state vectors
- Allows networks to focus on relevant information
- Handles dimension mismatches gracefully

### 3. Environment System (`ENVIRONMENT/`) {#environment-system}

The environment provides the game world and all resources agents interact with.

#### Terrain (`env_terrain.py`)
- Generates procedural terrain (land and water)
- Manages terrain ownership per faction
- Renders terrain with optimized caching
- Tracks traversability for agent pathfinding

#### Resources (`env_resources.py` & `Resources/`)
- Manages all game resources (trees, gold)
- Handles resource generation and placement
- Implements growth mechanics for planted trees
- Provides gathering/mining interfaces for agents

### 4. Game Management (`GAME/`) {#game-management}

The game management system orchestrates everything and handles the main game loop.

#### Game Manager (`game_manager.py`)
- Main game loop and step-by-step simulation
- Coordinates agent updates, training, and rendering
- Manages episode transitions and victory conditions
- Handles pause menu and episode restarts
- Integrates with TensorBoard for metrics logging

#### Game Rules (`game_rules.py`)
- Defines victory conditions (HQ destruction, resource collection, last standing)
- Calculates resource targets for victory
- Determines game state (ongoing, victory, defeat, draw)

#### Event Manager (`event_manager.py`)
- Handles game events (attacks, resource collection)
- Manages animation triggers
- Queues events for rendering

#### Camera (`camera.py`)
- Manages viewport and camera positioning
- Handles zoom and pan controls
- Converts world coordinates to screen coordinates

### 5. Rendering System (`RENDER/`) {#rendering-system}

The rendering system handles all visual output using Pygame.

#### Game Renderer (`Game_Renderer.py`)
- Renders terrain, agents, resources, HQs
- Displays HUD (health bars, resource counters, minimap)
- Handles animations (attack effects, tree growth)
- Manages tooltips and hover effects
- Implements sprite caching for performance

#### Menu Renderers
- **Main Menu**: Start screen with faction selection
- **Settings Menu**: Configuration of neural network, training, and game settings
- **Pause Menu**: In-game pause with restart/quit options
- **Credits**: Credits display

### 6. Utilities (`UTILITIES/`) {#utilities}

Utility systems provide configuration, logging, and support functions.

#### Configuration (`utils_config.py`)
- Global constants for the entire project
- Training parameters (learning rates, batch sizes)
- Game settings (world size, cell size, agent counts)
- Neural network architecture parameters

#### Logging (`utils_logger.py`)
- Custom logging system with buffered I/O
- Separate logs for errors and general information
- Force-flush support for episode ends
- Performance-optimized batch logging

#### Settings Manager (`settings_manager.py`)
- Persistent settings using JSON
- Tracks user preferences across runs
- Manages installer completion status
- Saves/loads configuration from settings menu

## Data Flow

### Agent Action Cycle
1. **Observation**: Agent observes environment (terrain, resources, threats)
2. **Action Selection**: PPO network selects action based on observation
3. **Behavior Execution**: Agent performs action (move, gather, attack, etc.)
4. **Reward Assignment**: Behavior system assigns reward based on outcome
5. **Memory Storage**: Experience stored for training
6. **Periodic Training**: Mini-batch training every 1000 steps

### HQ Decision Cycle
1. **State Collection**: HQ gathers global faction state (resources, threats, agents)
2. **Strategy Selection**: HQ network selects strategy based on state
3. **Strategy Execution**: HQ recruits agents, swaps roles, assigns tasks
4. **Reward Calculation**: Outcomes evaluated (was strategy logical? Did it succeed?)
5. **Memory Storage**: Strategy outcomes stored for training
6. **Episode Training**: Network trains at end of episode

### Rendering Cycle
1. **Update**: Game state updates (agent movement, actions, state changes)
2. **Rendering**: All game elements rendered to screen
3. **Event Processing**: User input processed (pause, zoom, etc.)
4. **Display**: Screen updated with new frame

## Key Design Patterns

### 1. Hierarchical Decision-Making
- **Agents**: Make tactical decisions (move, attack, gather)
- **HQ**: Makes strategic decisions (recruit, swap, prioritize)

### 2. Modular Behaviors
- Role-specific behaviors (gatherer vs peacekeeper)
- Mixin pattern for combining behaviors
- Task state machine (SUCCESS, FAILURE, ONGOING)

### 3. State-Aware Rewards
- Context-sensitive reward assignment
- Proximity bonuses for strategic positioning
- Survival bonuses for low-health agents
- Outcome-based rewards (was action successful?)

### 4. Performance Optimization
- Sprite caching to avoid repeated transformations
- Batch logging to reduce I/O
- Territory count caching per step
- Conditional tensor logging (debug vs production)

## Neural Network Architecture

### Agent Network (PPO)
- **Input**: Observation (terrain, resources, threats, health, energy)
- **Hidden**: Multi-layer fully connected network
- **Output**: Action probabilities and state value
- **Loss**: PPO clipped objective + value loss + entropy bonus

### HQ Network
- **Input**: Global state (resources, threats, agents, health)
- **Encoder**: Projects different state components (state, role, local, global)
- **Attention**: Focuses on important state information
- **Output**: Strategy probabilities and state value
- **Loss**: PPO loss with Huber loss for stability

### Attention Mechanism
- Queries, Keys, and Values derived from state
- Dot-product attention computes relevance
- Output combines state and attention-weighted features
- Handles dynamic input sizes gracefully

## Performance Characteristics

### Optimizations Implemented
1. **Batch Logging**: 200 messages buffered, 2s flush interval
2. **Sprite Caching**: Cached scaled sprites to avoid repeated transforms
3. **Terrain Caching**: Cached grass/water textures by cell size
4. **Territory Caching**: Territory count cached per step
5. **Tensor Optimization**: Conditional tensor logging, device-aware operations

### Performance Metrics
- **Runtime**: ~55s per episode (down from 88s, 37.6% improvement)
- **Function Calls**: 21.07M (down from 34.87M, 40% reduction)
- **Rendering**: 78% faster with sprite caching
- **Memory**: FIFO buffer prevents unbounded growth (max 20k samples)

## Extension Points

### Adding New Agent Roles
1. Create behavior file in `Agent_Behaviours/`
2. Add role to `ROLE_ACTIONS_MAP` in `utils_config.py`
3. Update reward assignment in `agent_behaviours.py`
4. Add role-specific observations in `agent_base.py`

### Adding New Resources
1. Create resource class in `ENVIRONMENT/Resources/`
2. Add resource type to resource manager
3. Implement gather/mine method
4. Add resource to global state tracking

### Adding New HQ Strategies
1. Add strategy to `HQ_STRATEGY_OPTIONS` in `utils_config.py`
2. Implement strategy logic in `agent_factions.py`
3. Add reward calculation for strategy outcomes
4. Update HQ state encoding if needed


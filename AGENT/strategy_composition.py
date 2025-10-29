"""
Strategy Composition System for HQ

This module implements a system for composing and sequencing strategies dynamically,
enabling the HQ to learn multi-step strategy sequences and break down high-level
goals into coordinated action plans.

The system provides:
1. Strategy composition (sequential, parallel, conditional, hierarchical)
2. Strategy sequencing (linear, branching, looping, convergent, divergent, recursive)
3. Goal-oriented strategy planning
4. Adaptive strategy execution
5. Emergent strategy discovery

Author: AI Assistant
Date: 2025-10-28
"""

"""Common Imports"""
from SHARED.core_imports import *
import UTILITIES.utils_config as utils_config
from UTILITIES.utils_helpers import profile_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union
import math
import time
from dataclasses import dataclass
from enum import Enum


logger = Logger(log_file="StrategyComposition_log.txt", log_level=logging.DEBUG)


@dataclass
class StrategyGoal:
    """Represents a high-level goal for strategy composition."""
    goal_type: utils_config.StrategyGoalType
    priority: float  # 0.0 to 1.0
    success_criteria: Dict[str, float]
    time_horizon: int  # Steps to achieve goal
    current_progress: float = 0.0
    created_step: int = 0
    achieved: bool = False


@dataclass
class StrategySequence:
    """Represents a sequence of strategies to execute."""
    sequence_id: str
    sequence_type: utils_config.StrategySequenceType
    strategies: List[str]  # Strategy names
    current_index: int = 0
    execution_state: str = "pending"  # pending, executing, completed, failed
    created_step: int = 0
    timeout_step: int = 0
    success_rate: float = 0.0


@dataclass
class StrategyComposition:
    """Represents a composed strategy with multiple components."""
    composition_id: str
    composition_type: utils_config.StrategyCompositionType
    primary_goal: StrategyGoal
    sequences: List[StrategySequence]
    parallel_strategies: List[str]
    conditional_strategies: Dict[str, str]  # condition -> strategy
    execution_state: str = "pending"
    created_step: int = 0
    timeout_step: int = 0
    success_rate: float = 0.0


class StrategyComposer(nn.Module):
    """
    Neural network for composing and sequencing strategies.
    Uses hierarchical RL with options framework.
    """
    
    def __init__(self, state_size: int = 32, hidden_size: int = 256):
        """
        Initialize the strategy composer.
        
        Args:
            state_size: Size of input state vector
            hidden_size: Size of hidden layers
        """
        super(StrategyComposer, self).__init__()
        
        self.state_size = state_size
        self.hidden_size = hidden_size
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
        )
        
        # Goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(len(utils_config.StrategyGoalType), hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
        )
        
        # Composition type predictor
        self.composition_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2 + hidden_size // 4, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, len(utils_config.StrategyCompositionType)),
            nn.Softmax(dim=-1),
        )
        
        # Sequence type predictor
        self.sequence_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2 + hidden_size // 4, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, len(utils_config.StrategySequenceType)),
            nn.Softmax(dim=-1),
        )
        
        # Strategy selector
        self.strategy_selector = nn.Sequential(
            nn.Linear(hidden_size // 2 + hidden_size // 4, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, len(utils_config.HQ_STRATEGY_OPTIONS)),
            nn.Softmax(dim=-1),
        )
        
        # Goal progress predictor
        self.goal_progress_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2 + hidden_size // 4, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )
        
        # Execution success predictor
        self.execution_success_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2 + hidden_size // 4, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, state: torch.Tensor, goal_type: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the strategy composer.
        
        Args:
            state: Current state tensor
            goal_type: Goal type tensor (one-hot encoded)
            
        Returns:
            Dictionary containing composition predictions
        """
        # Encode state and goal
        state_features = self.state_encoder(state)
        goal_features = self.goal_encoder(goal_type)
        
        # Combine features
        combined_features = torch.cat([state_features, goal_features], dim=-1)
        
        # Predict composition type
        composition_probs = self.composition_predictor(combined_features)
        
        # Predict sequence type
        sequence_probs = self.sequence_predictor(combined_features)
        
        # Select strategies
        strategy_probs = self.strategy_selector(combined_features)
        
        # Predict goal progress
        goal_progress = self.goal_progress_predictor(combined_features)
        
        # Predict execution success
        execution_success = self.execution_success_predictor(combined_features)
        
        return {
            "composition_probs": composition_probs,
            "sequence_probs": sequence_probs,
            "strategy_probs": strategy_probs,
            "goal_progress": goal_progress,
            "execution_success": execution_success,
        }


class StrategyCompositionSystem:
    """
    System for composing and sequencing strategies dynamically.
    Enables HQ to learn multi-step strategy sequences and break down
    high-level goals into coordinated action plans.
    """
    
    def __init__(self, faction_id: str, state_size: int = 32):
        """
        Initialize the strategy composition system.
        
        Args:
            faction_id: Unique identifier for the faction
            state_size: Size of input state vector
        """
        self.faction_id = faction_id
        self.state_size = state_size
        
        # Strategy composer neural network
        self.strategy_composer = StrategyComposer(state_size)
        
        # Active compositions and sequences
        self.active_compositions = {}
        self.active_sequences = {}
        self.completed_compositions = deque(maxlen=100)
        self.completed_sequences = deque(maxlen=100)
        
        # Goal management
        self.active_goals = {}
        self.achieved_goals = deque(maxlen=50)
        
        # Learning parameters
        self.composition_learning_rate = utils_config.STRATEGY_COMPOSITION_CONFIG["composition_learning_rate"]
        self.sequence_learning_rate = utils_config.STRATEGY_COMPOSITION_CONFIG["sequence_learning_rate"]
        
        # Performance tracking
        self.composition_success_rate = defaultdict(float)
        self.sequence_success_rate = defaultdict(float)
        self.goal_achievement_rate = defaultdict(float)
        
        # Strategy history for learning
        self.strategy_history = deque(maxlen=1000)
        self.composition_history = deque(maxlen=500)
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[STRATEGY COMPOSITION] Initialized for faction {faction_id}",
                level=logging.INFO,
            )
    
    def create_goal(self, goal_type: utils_config.StrategyGoalType, priority: float = 0.5) -> StrategyGoal:
        """
        Create a new strategy goal.
        
        Args:
            goal_type: Type of goal to create
            priority: Priority of the goal (0.0 to 1.0)
            
        Returns:
            Created strategy goal
        """
        goal_config = utils_config.STRATEGY_COMPOSITION_CONFIG["goal_types"][goal_type]
        
        goal = StrategyGoal(
            goal_type=goal_type,
            priority=priority,
            success_criteria=goal_config["success_criteria"].copy(),
            time_horizon=goal_config["time_horizon"],
            created_step=0,  # Will be set when used
        )
        
        goal_id = f"goal_{len(self.active_goals)}_{goal_type.value}"
        self.active_goals[goal_id] = goal
        
        return goal
    
    def compose_strategy(self, state: Dict[str, Any], goal: StrategyGoal, current_step: int) -> StrategyComposition:
        """
        Compose a strategy to achieve the given goal.
        
        Args:
            state: Current game state
            goal: Goal to achieve
            current_step: Current simulation step
            
        Returns:
            Composed strategy
        """
        # Convert state to tensor
        state_vector = self._state_to_vector(state)
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)
        
        # Convert goal type to one-hot tensor
        goal_type_tensor = self._goal_type_to_tensor(goal.goal_type)
        
        # Get composition predictions
        with torch.no_grad():
            predictions = self.strategy_composer(state_tensor, goal_type_tensor)
        
        # Select composition type
        composition_type_idx = torch.argmax(predictions["composition_probs"]).item()
        composition_type = list(utils_config.StrategyCompositionType)[composition_type_idx]
        
        # Select sequence type
        sequence_type_idx = torch.argmax(predictions["sequence_probs"]).item()
        sequence_type = list(utils_config.StrategySequenceType)[sequence_type_idx]
        
        # Select strategies
        strategy_probs = predictions["strategy_probs"].squeeze(0)
        top_strategies = torch.topk(strategy_probs, k=5).indices.tolist()
        selected_strategies = [utils_config.HQ_STRATEGY_OPTIONS[i] for i in top_strategies]
        
        # Create strategy sequence
        sequence = StrategySequence(
            sequence_id=f"seq_{len(self.active_sequences)}_{sequence_type.value}",
            sequence_type=sequence_type,
            strategies=selected_strategies,
            created_step=current_step,
            timeout_step=current_step + utils_config.STRATEGY_COMPOSITION_CONFIG["strategy_timeout"],
        )
        
        # Create strategy composition
        composition = StrategyComposition(
            composition_id=f"comp_{len(self.active_compositions)}_{composition_type.value}",
            composition_type=composition_type,
            primary_goal=goal,
            sequences=[sequence],
            parallel_strategies=[],
            conditional_strategies={},
            created_step=current_step,
            timeout_step=current_step + utils_config.STRATEGY_COMPOSITION_CONFIG["strategy_timeout"],
        )
        
        # Store active composition
        self.active_compositions[composition.composition_id] = composition
        self.active_sequences[sequence.sequence_id] = sequence
        
        # Update goal
        goal.created_step = current_step
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[STRATEGY COMPOSITION] Created composition {composition.composition_id} "
                f"for goal {goal.goal_type.value}",
                level=logging.INFO,
            )
        
        return composition
    
    def execute_composition(self, composition: StrategyComposition, current_step: int) -> str:
        """
        Execute the next step in a strategy composition.
        
        Args:
            composition: Strategy composition to execute
            current_step: Current simulation step
            
        Returns:
            Next strategy to execute
        """
        if composition.execution_state == "completed":
            return None
        
        # Check for timeout
        if current_step > composition.timeout_step:
            composition.execution_state = "failed"
            return None
        
        # Execute based on composition type
        if composition.composition_type == utils_config.StrategyCompositionType.SEQUENTIAL:
            return self._execute_sequential(composition, current_step)
        elif composition.composition_type == utils_config.StrategyCompositionType.PARALLEL:
            return self._execute_parallel(composition, current_step)
        elif composition.composition_type == utils_config.StrategyCompositionType.CONDITIONAL:
            return self._execute_conditional(composition, current_step)
        elif composition.composition_type == utils_config.StrategyCompositionType.HIERARCHICAL:
            return self._execute_hierarchical(composition, current_step)
        elif composition.composition_type == utils_config.StrategyCompositionType.ADAPTIVE:
            return self._execute_adaptive(composition, current_step)
        else:
            return self._execute_sequential(composition, current_step)
    
    def _execute_sequential(self, composition: StrategyComposition, current_step: int) -> str:
        """Execute sequential strategy composition."""
        for sequence in composition.sequences:
            if sequence.execution_state == "completed":
                continue
            
            if sequence.current_index < len(sequence.strategies):
                strategy = sequence.strategies[sequence.current_index]
                sequence.current_index += 1
                
                if sequence.current_index >= len(sequence.strategies):
                    sequence.execution_state = "completed"
                
                return strategy
        
        # All sequences completed
        composition.execution_state = "completed"
        return None
    
    def _execute_parallel(self, composition: StrategyComposition, current_step: int) -> str:
        """Execute parallel strategy composition."""
        # For parallel execution, we need to return multiple strategies
        # For now, return the first available strategy
        for sequence in composition.sequences:
            if sequence.execution_state == "completed":
                continue
            
            if sequence.current_index < len(sequence.strategies):
                strategy = sequence.strategies[sequence.current_index]
                sequence.current_index += 1
                
                if sequence.current_index >= len(sequence.strategies):
                    sequence.execution_state = "completed"
                
                return strategy
        
        # All sequences completed
        composition.execution_state = "completed"
        return None
    
    def _execute_conditional(self, composition: StrategyComposition, current_step: int) -> str:
        """Execute conditional strategy composition."""
        # For conditional execution, we need to evaluate conditions
        # For now, execute sequentially
        return self._execute_sequential(composition, current_step)
    
    def _execute_hierarchical(self, composition: StrategyComposition, current_step: int) -> str:
        """Execute hierarchical strategy composition."""
        # For hierarchical execution, we need to manage nested compositions
        # For now, execute sequentially
        return self._execute_sequential(composition, current_step)
    
    def _execute_adaptive(self, composition: StrategyComposition, current_step: int) -> str:
        """Execute adaptive strategy composition."""
        # For adaptive execution, we need to adapt based on results
        # For now, execute sequentially
        return self._execute_sequential(composition, current_step)
    
    def update_goal_progress(self, goal: StrategyGoal, state: Dict[str, Any], current_step: int):
        """
        Update progress toward a goal.
        
        Args:
            goal: Goal to update
            state: Current game state
            current_step: Current simulation step
        """
        # Calculate progress based on success criteria
        progress = 0.0
        total_weight = 0.0
        
        for criterion, target_value in goal.success_criteria.items():
            if criterion in state:
                current_value = state[criterion]
                if target_value > 0:
                    criterion_progress = min(current_value / target_value, 1.0)
                    progress += criterion_progress
                    total_weight += 1.0
        
        if total_weight > 0:
            goal.current_progress = progress / total_weight
        
        # Check if goal is achieved
        if goal.current_progress >= 0.8:  # 80% progress threshold
            goal.achieved = True
            goal_id = None
            for gid, g in self.active_goals.items():
                if g == goal:
                    goal_id = gid
                    break
            
            if goal_id:
                self.achieved_goals.append(goal)
                del self.active_goals[goal_id]
                
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[STRATEGY COMPOSITION] Goal {goal.goal_type.value} achieved "
                        f"with progress {goal.current_progress:.3f}",
                        level=logging.INFO,
                    )
    
    def evaluate_composition_success(self, composition: StrategyComposition, current_step: int) -> float:
        """
        Evaluate the success of a strategy composition.
        
        Args:
            composition: Strategy composition to evaluate
            current_step: Current simulation step
            
        Returns:
            Success score (0.0 to 1.0)
        """
        # Base success on goal achievement
        goal_progress = composition.primary_goal.current_progress
        
        # Factor in execution efficiency
        expected_duration = composition.primary_goal.time_horizon
        actual_duration = current_step - composition.created_step
        efficiency = min(expected_duration / max(actual_duration, 1), 1.0)
        
        # Factor in sequence completion
        completed_sequences = sum(1 for seq in composition.sequences if seq.execution_state == "completed")
        sequence_completion = completed_sequences / len(composition.sequences) if composition.sequences else 0.0
        
        # Calculate overall success
        success_score = (goal_progress * 0.5 + efficiency * 0.3 + sequence_completion * 0.2)
        
        return success_score
    
    def get_composition_reward(self) -> float:
        """
        Calculate reward for strategy composition quality.
        
        Returns:
            Strategy composition reward
        """
        # Base reward from composition success rates
        avg_composition_success = np.mean(list(self.composition_success_rate.values())) if self.composition_success_rate else 0.0
        base_reward = avg_composition_success * 0.3
        
        # Bonus for goal achievement
        goal_achievement_bonus = len(self.achieved_goals) * 0.1
        
        # Bonus for efficient execution
        efficiency_bonus = 0.0
        if self.composition_history:
            recent_compositions = list(self.composition_history)[-10:]
            avg_efficiency = np.mean([comp.success_rate for comp in recent_compositions])
            efficiency_bonus = avg_efficiency * 0.2
        
        # Bonus for adaptive behavior
        adaptive_compositions = sum(1 for comp in self.active_compositions.values() 
                                  if comp.composition_type == utils_config.StrategyCompositionType.ADAPTIVE)
        adaptive_bonus = adaptive_compositions * 0.05
        
        return base_reward + goal_achievement_bonus + efficiency_bonus + adaptive_bonus
    
    def _state_to_vector(self, state: Dict[str, Any]) -> List[float]:
        """
        Convert state dictionary to vector representation.
        
        Args:
            state: State dictionary
            
        Returns:
            State vector
        """
        # Extract key state components for strategy composition
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
        
        # Add strategy composition context
        vector.extend([
            len(self.active_compositions) / 10.0,
            len(self.active_goals) / 10.0,
            len(self.achieved_goals) / 10.0,
            len(self.completed_compositions) / 100.0,
        ])
        
        # Add temporal information
        vector.extend([
            state.get("step", 0.0) / 1000.0,
            state.get("episode", 0.0) / 100.0,
            state.get("strategy_duration", 0.0) / 100.0,
            state.get("last_strategy_change", 0.0) / 50.0,
        ])
        
        # Add performance metrics
        vector.extend([
            state.get("win_rate", 0.5),
            state.get("survival_rate", 0.5),
            state.get("efficiency_score", 0.5),
            state.get("coordination_score", 0.5),
        ])
        
        # Add mission and communication context
        vector.extend([
            state.get("mission_progress", 0.0),
            state.get("communication_success_rate", 0.5),
            state.get("experience_sharing_rate", 0.5),
            state.get("learned_state_quality", 0.5),
        ])
        
        # Ensure vector has correct size
        while len(vector) < self.state_size:
            vector.append(0.0)
        
        return vector[:self.state_size]
    
    def _goal_type_to_tensor(self, goal_type: utils_config.StrategyGoalType) -> torch.Tensor:
        """
        Convert goal type to one-hot tensor.
        
        Args:
            goal_type: Goal type to convert
            
        Returns:
            One-hot encoded tensor
        """
        goal_types = list(utils_config.StrategyGoalType)
        goal_index = goal_types.index(goal_type)
        
        tensor = torch.zeros(len(goal_types))
        tensor[goal_index] = 1.0
        
        return tensor.unsqueeze(0)
    
    def get_composition_summary(self) -> Dict[str, Any]:
        """
        Get summary of strategy composition system performance.
        
        Returns:
            Dictionary containing performance metrics
        """
        return {
            "faction_id": self.faction_id,
            "active_compositions": len(self.active_compositions),
            "active_sequences": len(self.active_sequences),
            "active_goals": len(self.active_goals),
            "achieved_goals": len(self.achieved_goals),
            "completed_compositions": len(self.completed_compositions),
            "composition_success_rate": dict(self.composition_success_rate),
            "sequence_success_rate": dict(self.sequence_success_rate),
            "goal_achievement_rate": dict(self.goal_achievement_rate),
            "avg_composition_success": np.mean(list(self.composition_success_rate.values())) if self.composition_success_rate else 0.0,
            "avg_sequence_success": np.mean(list(self.sequence_success_rate.values())) if self.sequence_success_rate else 0.0,
            "avg_goal_achievement": np.mean(list(self.goal_achievement_rate.values())) if self.goal_achievement_rate else 0.0,
        }
    
    def reset_episode(self):
        """Reset the strategy composition system for a new episode."""
        self.active_compositions.clear()
        self.active_sequences.clear()
        self.active_goals.clear()
        self.completed_compositions.clear()
        self.completed_sequences.clear()
        self.achieved_goals.clear()
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[STRATEGY COMPOSITION] Reset episode for faction {self.faction_id}",
                level=logging.INFO,
            )

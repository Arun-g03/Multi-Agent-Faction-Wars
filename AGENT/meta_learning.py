"""
Meta-Learning for Strategy Discovery System

This module implements a meta-learning system for discovering and refining new strategies,
enabling the HQ to learn novel approaches beyond the predefined strategies.

The system provides:
1. Multiple meta-learning approaches (gradient-based, evolutionary, reinforcement, memory-based)
2. Strategy discovery methods (pattern analysis, genetic algorithm, neural architecture search)
3. Strategy evaluation metrics (success rate, efficiency, adaptability, novelty, robustness)
4. Knowledge transfer between contexts
5. Emergent strategy combination

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
import random
from dataclasses import dataclass
from enum import Enum


logger = Logger(log_file="MetaLearning_log.txt", log_level=logging.DEBUG)


@dataclass
class DiscoveredStrategy:
    """Represents a discovered strategy."""
    strategy_id: str
    strategy_name: str
    strategy_type: str
    parameters: Dict[str, Any]
    discovery_method: utils_config.StrategyDiscoveryMethod
    evaluation_metrics: Dict[str, float]
    success_rate: float
    novelty_score: float
    created_step: int
    last_used_step: int
    usage_count: int = 0
    is_active: bool = True


@dataclass
class MetaLearningEpisode:
    """Represents a meta-learning episode."""
    episode_id: str
    context: Dict[str, Any]
    strategies_tested: List[str]
    results: Dict[str, float]
    meta_learning_type: utils_config.MetaLearningType
    created_step: int
    success: bool = False


class MetaLearner(nn.Module):
    """
    Neural network for meta-learning strategy discovery.
    Uses Model-Agnostic Meta-Learning (MAML) approach.
    """
    
    def __init__(self, state_size: int = 32, strategy_size: int = 10, hidden_size: int = 256):
        """
        Initialize the meta-learner.
        
        Args:
            state_size: Size of input state vector
            strategy_size: Size of strategy representation
            hidden_size: Size of hidden layers
        """
        super(MetaLearner, self).__init__()
        
        self.state_size = state_size
        self.strategy_size = strategy_size
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
        
        # Strategy encoder
        self.strategy_encoder = nn.Sequential(
            nn.Linear(strategy_size, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
        )
        
        # Meta-learning head
        self.meta_head = nn.Sequential(
            nn.Linear(hidden_size // 2 + hidden_size // 4, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, strategy_size),
        )
        
        # Strategy evaluator
        self.strategy_evaluator = nn.Sequential(
            nn.Linear(hidden_size // 2 + hidden_size // 4, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, len(utils_config.StrategyEvaluationMetric)),
        )
        
        # Novelty detector
        self.novelty_detector = nn.Sequential(
            nn.Linear(hidden_size // 2 + hidden_size // 4, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, state: torch.Tensor, strategy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the meta-learner.
        
        Args:
            state: Current state tensor
            strategy: Strategy tensor
            
        Returns:
            Dictionary containing meta-learning predictions
        """
        # Encode state and strategy
        state_features = self.state_encoder(state)
        strategy_features = self.strategy_encoder(strategy)
        
        # Combine features
        combined_features = torch.cat([state_features, strategy_features], dim=-1)
        
        # Meta-learning prediction
        meta_prediction = self.meta_head(combined_features)
        
        # Strategy evaluation
        evaluation_scores = self.strategy_evaluator(combined_features)
        
        # Novelty detection
        novelty_score = self.novelty_detector(combined_features)
        
        return {
            "meta_prediction": meta_prediction,
            "evaluation_scores": evaluation_scores,
            "novelty_score": novelty_score,
        }
    
    def meta_update(self, support_set: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
                    query_set: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
        """
        Perform meta-learning update using MAML.
        
        Args:
            support_set: Support set for inner loop
            query_set: Query set for outer loop
            
        Returns:
            Meta-learning loss
        """
        # Inner loop: adapt to support set
        adapted_params = self._inner_loop_update(support_set)
        
        # Outer loop: evaluate on query set
        meta_loss = self._outer_loop_update(query_set, adapted_params)
        
        return meta_loss
    
    def _inner_loop_update(self, support_set: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Perform inner loop update for MAML."""
        # Store original parameters
        original_params = {name: param.clone() for name, param in self.named_parameters()}
        
        # Inner loop updates
        inner_lr = utils_config.META_LEARNING_CONFIG["inner_learning_rate"]
        inner_steps = utils_config.META_LEARNING_CONFIG["inner_steps"]
        
        for step in range(inner_steps):
            inner_loss = 0.0
            for state, strategy, target in support_set:
                output = self.forward(state, strategy)
                loss = F.mse_loss(output["meta_prediction"], target)
                inner_loss += loss
            
            # Gradient update
            inner_loss.backward()
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        param.data -= inner_lr * param.grad
                        param.grad.zero_()
        
        # Return adapted parameters
        adapted_params = {name: param.clone() for name, param in self.named_parameters()}
        
        # Restore original parameters
        for name, param in self.named_parameters():
            param.data = original_params[name]
        
        return adapted_params
    
    def _outer_loop_update(self, query_set: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
                          adapted_params: Dict[str, torch.Tensor]) -> float:
        """Perform outer loop update for MAML."""
        # Temporarily set adapted parameters
        original_params = {name: param.clone() for name, param in self.named_parameters()}
        
        for name, param in self.named_parameters():
            param.data = adapted_params[name]
        
        # Evaluate on query set
        meta_loss = 0.0
        for state, strategy, target in query_set:
            output = self.forward(state, strategy)
            loss = F.mse_loss(output["meta_prediction"], target)
            meta_loss += loss
        
        # Restore original parameters
        for name, param in self.named_parameters():
            param.data = original_params[name]
        
        return meta_loss


class MetaLearningSystem:
    """
    System for meta-learning strategy discovery.
    Enables HQ to discover and refine new strategies beyond predefined ones.
    """
    
    def __init__(self, faction_id: str, state_size: int = 32):
        """
        Initialize the meta-learning system.
        
        Args:
            faction_id: Unique identifier for the faction
            state_size: Size of input state vector
        """
        self.faction_id = faction_id
        self.state_size = state_size
        
        # Meta-learner neural network
        self.meta_learner = MetaLearner(state_size)
        
        # Discovered strategies
        self.discovered_strategies = {}
        self.strategy_population = []
        self.strategy_performance_history = defaultdict(list)
        
        # Meta-learning episodes
        self.meta_episodes = deque(maxlen=1000)
        self.active_episodes = {}
        
        # Learning parameters
        self.meta_learning_rate = utils_config.META_LEARNING_CONFIG["meta_learning_rate"]
        self.discovery_frequency = utils_config.META_LEARNING_CONFIG["discovery_frequency"]
        self.evaluation_episodes = utils_config.META_LEARNING_CONFIG["evaluation_episodes"]
        
        # Performance tracking
        self.discovery_success_rate = defaultdict(float)
        self.strategy_quality_scores = defaultdict(float)
        self.meta_learning_progress = defaultdict(float)
        
        # Strategy discovery methods
        self.discovery_methods = list(utils_config.StrategyDiscoveryMethod)
        self.current_discovery_method = utils_config.StrategyDiscoveryMethod.PATTERN_ANALYSIS
        
        # Meta-learning types
        self.meta_learning_types = list(utils_config.MetaLearningType)
        self.current_meta_learning_type = utils_config.MetaLearningType.GRADIENT_BASED
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[META-LEARNING] Initialized for faction {faction_id}",
                level=logging.INFO,
            )
    
    def discover_strategies(self, state: Dict[str, Any], current_step: int) -> List[DiscoveredStrategy]:
        """
        Discover new strategies using various methods.
        
        Args:
            state: Current game state
            current_step: Current simulation step
            
        Returns:
            List of discovered strategies
        """
        if current_step % self.discovery_frequency != 0:
            return []
        
        discovered_strategies = []
        
        # Try different discovery methods
        for method in self.discovery_methods:
            try:
                new_strategies = self._discover_with_method(method, state, current_step)
                discovered_strategies.extend(new_strategies)
            except Exception as e:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[META-LEARNING] Discovery method {method.value} failed: {e}",
                        level=logging.WARNING,
                    )
        
        # Evaluate discovered strategies
        evaluated_strategies = []
        for strategy in discovered_strategies:
            evaluation_metrics = self._evaluate_strategy(strategy, state)
            strategy.evaluation_metrics = evaluation_metrics
            
            # Check if strategy meets thresholds
            if (strategy.evaluation_metrics.get("novelty", 0) >= utils_config.META_LEARNING_CONFIG["novelty_threshold"] and
                strategy.evaluation_metrics.get("success_rate", 0) >= utils_config.META_LEARNING_CONFIG["success_threshold"]):
                evaluated_strategies.append(strategy)
        
        # Add to discovered strategies
        for strategy in evaluated_strategies:
            self.discovered_strategies[strategy.strategy_id] = strategy
            self.strategy_population.append(strategy)
            
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[META-LEARNING] Discovered strategy {strategy.strategy_name} "
                    f"using {strategy.discovery_method.value}",
                    level=logging.INFO,
                )
        
        return evaluated_strategies
    
    def _discover_with_method(self, method: utils_config.StrategyDiscoveryMethod, 
                             state: Dict[str, Any], current_step: int) -> List[DiscoveredStrategy]:
        """Discover strategies using a specific method."""
        if method == utils_config.StrategyDiscoveryMethod.PATTERN_ANALYSIS:
            return self._discover_with_pattern_analysis(state, current_step)
        elif method == utils_config.StrategyDiscoveryMethod.GENETIC_ALGORITHM:
            return self._discover_with_genetic_algorithm(state, current_step)
        elif method == utils_config.StrategyDiscoveryMethod.REINFORCEMENT_SEARCH:
            return self._discover_with_reinforcement_search(state, current_step)
        elif method == utils_config.StrategyDiscoveryMethod.TRANSFER_FROM_SIMILAR:
            return self._discover_with_transfer_learning(state, current_step)
        elif method == utils_config.StrategyDiscoveryMethod.EMERGENT_COMBINATION:
            return self._discover_with_emergent_combination(state, current_step)
        else:
            return []
    
    def _discover_with_pattern_analysis(self, state: Dict[str, Any], current_step: int) -> List[DiscoveredStrategy]:
        """Discover strategies by analyzing successful patterns."""
        strategies = []
        
        # Analyze successful episodes
        successful_episodes = [ep for ep in self.meta_episodes if ep.success]
        
        if len(successful_episodes) >= 3:
            # Find common patterns in successful episodes
            common_patterns = self._find_common_patterns(successful_episodes)
            
            for pattern in common_patterns:
                strategy = DiscoveredStrategy(
                    strategy_id=f"pattern_{len(self.discovered_strategies)}_{current_step}",
                    strategy_name=f"Pattern Strategy {len(self.discovered_strategies)}",
                    strategy_type="pattern_based",
                    parameters=pattern,
                    discovery_method=utils_config.StrategyDiscoveryMethod.PATTERN_ANALYSIS,
                    evaluation_metrics={},
                    success_rate=0.0,
                    novelty_score=0.0,
                    created_step=current_step,
                    last_used_step=current_step,
                )
                strategies.append(strategy)
        
        return strategies
    
    def _discover_with_genetic_algorithm(self, state: Dict[str, Any], current_step: int) -> List[DiscoveredStrategy]:
        """Discover strategies using genetic algorithm."""
        strategies = []
        
        # Initialize population if empty
        if not self.strategy_population:
            self._initialize_strategy_population()
        
        # Genetic operations
        population_size = utils_config.META_LEARNING_CONFIG["strategy_population_size"]
        mutation_rate = utils_config.META_LEARNING_CONFIG["mutation_rate"]
        crossover_rate = utils_config.META_LEARNING_CONFIG["crossover_rate"]
        
        # Selection
        selected_strategies = self._genetic_selection(self.strategy_population, population_size // 2)
        
        # Crossover
        offspring = self._genetic_crossover(selected_strategies, crossover_rate)
        
        # Mutation
        mutated_offspring = self._genetic_mutation(offspring, mutation_rate)
        
        # Create new strategies
        for i, params in enumerate(mutated_offspring):
            strategy = DiscoveredStrategy(
                strategy_id=f"genetic_{len(self.discovered_strategies)}_{current_step}_{i}",
                strategy_name=f"Genetic Strategy {len(self.discovered_strategies)}",
                strategy_type="genetic",
                parameters=params,
                discovery_method=utils_config.StrategyDiscoveryMethod.GENETIC_ALGORITHM,
                evaluation_metrics={},
                success_rate=0.0,
                novelty_score=0.0,
                created_step=current_step,
                last_used_step=current_step,
            )
            strategies.append(strategy)
        
        return strategies
    
    def _discover_with_reinforcement_search(self, state: Dict[str, Any], current_step: int) -> List[DiscoveredStrategy]:
        """Discover strategies using reinforcement learning search."""
        strategies = []
        
        # Convert state to tensor
        state_vector = self._state_to_vector(state)
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)
        
        # Generate strategy candidates
        num_candidates = 5
        for i in range(num_candidates):
            # Generate random strategy parameters
            strategy_params = self._generate_random_strategy_parameters()
            strategy_params_list = list(strategy_params.values())
            # Ensure we have exactly 10 parameters
            while len(strategy_params_list) < 10:
                strategy_params_list.append(0.0)
            strategy_params_list = strategy_params_list[:10]  # Take first 10 parameters
            strategy_tensor = torch.tensor(strategy_params_list, dtype=torch.float32).unsqueeze(0)
            
            # Use meta-learner to predict strategy quality
            with torch.no_grad():
                output = self.meta_learner(state_tensor, strategy_tensor)
                novelty_score = output["novelty_score"].item()
                evaluation_scores = output["evaluation_scores"].squeeze(0).tolist()
            
            # Create strategy if novel enough
            if novelty_score >= utils_config.META_LEARNING_CONFIG["novelty_threshold"]:
                strategy = DiscoveredStrategy(
                    strategy_id=f"rl_{len(self.discovered_strategies)}_{current_step}_{i}",
                    strategy_name=f"RL Strategy {len(self.discovered_strategies)}",
                    strategy_type="reinforcement_learning",
                    parameters=strategy_params,
                    discovery_method=utils_config.StrategyDiscoveryMethod.REINFORCEMENT_SEARCH,
                    evaluation_metrics={},
                    success_rate=0.0,
                    novelty_score=novelty_score,
                    created_step=current_step,
                    last_used_step=current_step,
                )
                strategies.append(strategy)
        
        return strategies
    
    def _discover_with_transfer_learning(self, state: Dict[str, Any], current_step: int) -> List[DiscoveredStrategy]:
        """Discover strategies using transfer learning from similar contexts."""
        strategies = []
        
        # Find similar contexts
        similar_episodes = self._find_similar_contexts(state)
        
        if similar_episodes:
            # Transfer successful strategies from similar contexts
            for episode in similar_episodes[:3]:  # Top 3 similar episodes
                for strategy_name in episode.strategies_tested:
                    if episode.results.get(strategy_name, 0) > 0.7:  # High success rate
                        # Create transferred strategy
                        strategy = DiscoveredStrategy(
                            strategy_id=f"transfer_{len(self.discovered_strategies)}_{current_step}",
                            strategy_name=f"Transferred Strategy {len(self.discovered_strategies)}",
                            strategy_type="transferred",
                            parameters={"transferred_from": episode.episode_id, "original_strategy": strategy_name},
                            discovery_method=utils_config.StrategyDiscoveryMethod.TRANSFER_FROM_SIMILAR,
                            evaluation_metrics={},
                            success_rate=episode.results[strategy_name],
                            novelty_score=0.3,  # Lower novelty for transferred strategies
                            created_step=current_step,
                            last_used_step=current_step,
                        )
                        strategies.append(strategy)
        
        return strategies
    
    def _discover_with_emergent_combination(self, state: Dict[str, Any], current_step: int) -> List[DiscoveredStrategy]:
        """Discover strategies by combining existing strategies."""
        strategies = []
        
        # Get existing strategies
        existing_strategies = list(self.discovered_strategies.values())
        
        if len(existing_strategies) >= 2:
            # Combine strategies
            for i in range(min(3, len(existing_strategies) // 2)):
                strategy1 = existing_strategies[i * 2]
                strategy2 = existing_strategies[i * 2 + 1]
                
                # Combine parameters
                combined_params = self._combine_strategy_parameters(strategy1.parameters, strategy2.parameters)
                
                strategy = DiscoveredStrategy(
                    strategy_id=f"emergent_{len(self.discovered_strategies)}_{current_step}_{i}",
                    strategy_name=f"Emergent Strategy {len(self.discovered_strategies)}",
                    strategy_type="emergent_combination",
                    parameters=combined_params,
                    discovery_method=utils_config.StrategyDiscoveryMethod.EMERGENT_COMBINATION,
                    evaluation_metrics={},
                    success_rate=0.0,
                    novelty_score=0.8,  # High novelty for emergent combinations
                    created_step=current_step,
                    last_used_step=current_step,
                )
                strategies.append(strategy)
        
        return strategies
    
    def _evaluate_strategy(self, strategy: DiscoveredStrategy, state: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a discovered strategy."""
        metrics = {}
        
        # Convert state to tensor
        state_vector = self._state_to_vector(state)
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)
        
        # Convert strategy to tensor
        strategy_params = list(strategy.parameters.values())
        # Ensure we have exactly 10 parameters
        while len(strategy_params) < 10:
            strategy_params.append(0.0)
        strategy_params = strategy_params[:10]  # Take first 10 parameters
        strategy_tensor = torch.tensor(strategy_params, dtype=torch.float32).unsqueeze(0)
        
        # Use meta-learner to evaluate
        with torch.no_grad():
            output = self.meta_learner(state_tensor, strategy_tensor)
            evaluation_scores = output["evaluation_scores"].squeeze(0).tolist()
            novelty_score = output["novelty_score"].item()
        
        # Map evaluation scores to metrics
        evaluation_metrics = list(utils_config.StrategyEvaluationMetric)
        for i, metric in enumerate(evaluation_metrics):
            metrics[metric.value] = evaluation_scores[i] if i < len(evaluation_scores) else 0.0
        
        # Add novelty score
        metrics["novelty"] = novelty_score
        
        return metrics
    
    def _find_common_patterns(self, episodes: List[MetaLearningEpisode]) -> List[Dict[str, Any]]:
        """Find common patterns in successful episodes."""
        patterns = []
        
        # Analyze strategy combinations
        strategy_combinations = defaultdict(int)
        for episode in episodes:
            for strategy in episode.strategies_tested:
                strategy_combinations[strategy] += 1
        
        # Find most common strategies
        common_strategies = sorted(strategy_combinations.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for strategy_name, count in common_strategies:
            if count >= 2:  # Appears in at least 2 successful episodes
                patterns.append({"strategy": strategy_name, "frequency": count / len(episodes)})
        
        return patterns
    
    def _initialize_strategy_population(self):
        """Initialize strategy population for genetic algorithm."""
        population_size = utils_config.META_LEARNING_CONFIG["strategy_population_size"]
        
        for i in range(population_size):
            params = self._generate_random_strategy_parameters()
            strategy = DiscoveredStrategy(
                strategy_id=f"init_{i}",
                strategy_name=f"Initial Strategy {i}",
                strategy_type="initial",
                parameters=params,
                discovery_method=utils_config.StrategyDiscoveryMethod.GENETIC_ALGORITHM,
                evaluation_metrics={},
                success_rate=0.0,
                novelty_score=0.0,
                created_step=0,
                last_used_step=0,
            )
            self.strategy_population.append(strategy)
    
    def _genetic_selection(self, population: List[DiscoveredStrategy], num_selected: int) -> List[DiscoveredStrategy]:
        """Select strategies using tournament selection."""
        selected = []
        
        for _ in range(num_selected):
            # Tournament selection
            tournament_size = 3
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = max(tournament, key=lambda s: s.success_rate)
            selected.append(winner)
        
        return selected
    
    def _genetic_crossover(self, parents: List[DiscoveredStrategy], crossover_rate: float) -> List[Dict[str, Any]]:
        """Perform crossover between parent strategies."""
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            if random.random() < crossover_rate:
                parent1 = parents[i]
                parent2 = parents[i + 1]
                
                # Combine parameters
                combined_params = self._combine_strategy_parameters(parent1.parameters, parent2.parameters)
                offspring.append(combined_params)
            else:
                # No crossover, keep parent parameters
                offspring.append(parents[i].parameters)
        
        return offspring
    
    def _genetic_mutation(self, offspring: List[Dict[str, Any]], mutation_rate: float) -> List[Dict[str, Any]]:
        """Mutate offspring strategies."""
        mutated = []
        
        for params in offspring:
            if random.random() < mutation_rate:
                # Mutate parameters
                mutated_params = self._mutate_strategy_parameters(params)
                mutated.append(mutated_params)
            else:
                mutated.append(params)
        
        return mutated
    
    def _generate_random_strategy_parameters(self) -> Dict[str, Any]:
        """Generate random strategy parameters."""
        return {
            "aggression_level": random.uniform(0.0, 1.0),
            "resource_threshold": random.uniform(0.0, 1.0),
            "urgency": random.uniform(0.0, 1.0),
            "mission_autonomy": random.uniform(0.0, 1.0),
            "coordination_preference": random.uniform(0.0, 1.0),
            "agent_adaptability": random.uniform(0.0, 1.0),
            "failure_tolerance": random.uniform(0.0, 1.0),
            "agent_count_target": random.randint(1, 10),
            "mission_complexity": random.randint(1, 4),
        }
    
    def _combine_strategy_parameters(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> Dict[str, Any]:
        """Combine two strategy parameter sets."""
        combined = {}
        
        for key in params1:
            if key in params2:
                if isinstance(params1[key], (int, float)):
                    # Average numeric parameters
                    combined[key] = (params1[key] + params2[key]) / 2
                else:
                    # Randomly choose for non-numeric parameters
                    combined[key] = random.choice([params1[key], params2[key]])
            else:
                combined[key] = params1[key]
        
        return combined
    
    def _mutate_strategy_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate strategy parameters."""
        mutated = params.copy()
        
        for key, value in mutated.items():
            if isinstance(value, (int, float)):
                # Add small random change
                mutation_strength = 0.1
                if isinstance(value, int):
                    mutated[key] = max(1, int(value + random.uniform(-mutation_strength, mutation_strength) * value))
                else:
                    mutated[key] = max(0.0, min(1.0, value + random.uniform(-mutation_strength, mutation_strength)))
        
        return mutated
    
    def _find_similar_contexts(self, state: Dict[str, Any]) -> List[MetaLearningEpisode]:
        """Find episodes with similar contexts."""
        similar_episodes = []
        
        for episode in self.meta_episodes:
            similarity = self._calculate_context_similarity(state, episode.context)
            if similarity > 0.7:  # High similarity threshold
                similar_episodes.append(episode)
        
        # Sort by similarity
        similar_episodes.sort(key=lambda ep: self._calculate_context_similarity(state, ep.context), reverse=True)
        
        return similar_episodes[:5]  # Top 5 similar episodes
    
    def _calculate_context_similarity(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
        """Calculate similarity between two states."""
        # Simple similarity based on key metrics
        key_metrics = ["HQ_health", "gold_balance", "food_balance", "resource_count", "threat_count"]
        
        similarities = []
        for metric in key_metrics:
            if metric in state1 and metric in state2:
                val1 = state1[metric]
                val2 = state2[metric]
                if val1 > 0 or val2 > 0:
                    similarity = 1.0 - abs(val1 - val2) / max(val1, val2)
                    similarities.append(max(0.0, similarity))
        
        return np.mean(similarities) if similarities else 0.0
    
    def _state_to_vector(self, state: Dict[str, Any]) -> List[float]:
        """Convert state dictionary to vector representation."""
        # Extract key state components for meta-learning
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
        
        # Add meta-learning context
        vector.extend([
            len(self.discovered_strategies) / 20.0,
            len(self.meta_episodes) / 1000.0,
            len(self.strategy_population) / 50.0,
            np.mean(list(self.discovery_success_rate.values())) if self.discovery_success_rate else 0.0,
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
        
        # Add strategy composition context
        vector.extend([
            state.get("active_compositions", 0) / 10.0,
            state.get("active_goals", 0) / 10.0,
            state.get("achieved_goals", 0) / 50.0,
            state.get("composition_success_rate", 0.5),
        ])
        
        # Ensure vector has correct size
        while len(vector) < self.state_size:
            vector.append(0.0)
        
        return vector[:self.state_size]
    
    def get_meta_learning_reward(self) -> float:
        """Calculate reward for meta-learning progress."""
        # Base reward from discovery success rate
        avg_discovery_success = np.mean(list(self.discovery_success_rate.values())) if self.discovery_success_rate else 0.0
        base_reward = avg_discovery_success * 0.3
        
        # Bonus for novel strategies
        novel_strategies = sum(1 for s in self.discovered_strategies.values() if s.novelty_score > 0.7)
        novelty_bonus = novel_strategies * 0.1
        
        # Bonus for high-quality strategies
        high_quality_strategies = sum(1 for s in self.discovered_strategies.values() if s.success_rate > 0.8)
        quality_bonus = high_quality_strategies * 0.15
        
        # Bonus for meta-learning progress
        meta_progress = np.mean(list(self.meta_learning_progress.values())) if self.meta_learning_progress else 0.0
        progress_bonus = meta_progress * 0.2
        
        return base_reward + novelty_bonus + quality_bonus + progress_bonus
    
    def get_meta_learning_summary(self) -> Dict[str, Any]:
        """Get summary of meta-learning system performance."""
        return {
            "faction_id": self.faction_id,
            "discovered_strategies": len(self.discovered_strategies),
            "strategy_population_size": len(self.strategy_population),
            "meta_episodes": len(self.meta_episodes),
            "discovery_success_rate": dict(self.discovery_success_rate),
            "strategy_quality_scores": dict(self.strategy_quality_scores),
            "meta_learning_progress": dict(self.meta_learning_progress),
            "avg_discovery_success": np.mean(list(self.discovery_success_rate.values())) if self.discovery_success_rate else 0.0,
            "avg_strategy_quality": np.mean(list(self.strategy_quality_scores.values())) if self.strategy_quality_scores else 0.0,
            "avg_meta_progress": np.mean(list(self.meta_learning_progress.values())) if self.meta_learning_progress else 0.0,
        }
    
    def reset_episode(self):
        """Reset the meta-learning system for a new episode."""
        self.active_episodes.clear()
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[META-LEARNING] Reset episode for faction {self.faction_id}",
                level=logging.INFO,
            )

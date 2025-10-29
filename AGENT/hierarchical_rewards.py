"""
Hierarchical Reward System for Multi-Agent Faction Wars

This module implements a two-level reward system that connects HQ strategy success
to agent execution quality, enabling true hierarchical reinforcement learning.

The system provides:
1. Agent-level tactical rewards for individual task execution
2. HQ-level strategic rewards for strategy selection and execution
3. Cross-level feedback between agents and HQ
4. Coordination and adaptation bonuses
5. Mission success evaluation

Author: AI Assistant
Date: 2025-10-28
"""

"""Common Imports"""
from SHARED.core_imports import *
import UTILITIES.utils_config as utils_config
from UTILITIES.utils_helpers import profile_function
from collections import defaultdict, deque
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


logger = Logger(log_file="HierarchicalRewards_log.txt", log_level=logging.DEBUG)


class HierarchicalRewardManager:
    """
    Manages hierarchical rewards for both agents and HQ, providing cross-level feedback
    and enabling true hierarchical reinforcement learning.
    """
    
    def __init__(self, faction_id: str):
        """
        Initialize the hierarchical reward manager for a faction.
        
        Args:
            faction_id (str): Unique identifier for the faction
        """
        self.faction_id = faction_id
        
        # Experience tracking
        self.agent_experiences = defaultdict(list)  # agent_id -> list of experiences
        self.hq_experiences = []  # HQ-level experiences
        self.coordination_history = deque(maxlen=utils_config.EXPERIENCE_REPORTING_CONFIG["coordination_window"])
        self.adaptation_history = deque(maxlen=utils_config.EXPERIENCE_REPORTING_CONFIG["adaptation_window"])
        
        # Performance metrics
        self.mission_progress = {
            "resource_collection": 0.0,
            "threat_elimination": 0.0,
            "territory_control": 0.0,
            "agent_coordination": 0.0,
        }
        
        # Reward components tracking
        self.agent_reward_components = defaultdict(lambda: defaultdict(float))
        self.hq_reward_components = defaultdict(float)
        
        # Episode tracking
        self.episode_step = 0
        self.episode_start_step = 0
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[HIERARCHICAL REWARDS] Initialized for faction {faction_id}",
                level=logging.INFO,
            )
    
    def report_agent_experience(
        self,
        agent_id: str,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        task_type: str = None,
        task_state: str = None,
        coordination_data: Dict = None,
        adaptation_data: Dict = None,
    ):
        """
        Report an agent's experience for hierarchical reward calculation.
        
        Args:
            agent_id (str): Unique identifier for the agent
            state: Current state
            action: Action taken
            reward: Immediate reward received
            next_state: Next state
            done: Whether episode is done
            task_type: Type of task being performed
            task_state: Current state of the task
            coordination_data: Data about coordination with other agents
            adaptation_data: Data about adaptive behavior
        """
        experience = {
            "step": self.episode_step,
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "task_type": task_type,
            "task_state": task_state,
            "coordination_data": coordination_data or {},
            "adaptation_data": adaptation_data or {},
        }
        
        self.agent_experiences[agent_id].append(experience)
        
        # Track coordination and adaptation
        if coordination_data:
            self.coordination_history.append({
                "agent_id": agent_id,
                "step": self.episode_step,
                "data": coordination_data,
            })
        
        if adaptation_data:
            self.adaptation_history.append({
                "agent_id": agent_id,
                "step": self.episode_step,
                "data": adaptation_data,
            })
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[AGENT EXPERIENCE] Agent {agent_id} reported experience: task={task_type}, state={task_state}",
                level=logging.DEBUG,
            )
    
    def report_hq_experience(
        self,
        strategy: str,
        parameters: Dict,
        execution_success: bool,
        agent_feedback: Dict,
        mission_progress: Dict,
    ):
        """
        Report HQ's strategic experience for hierarchical reward calculation.
        
        Args:
            strategy: Strategy selected by HQ
            parameters: Strategy parameters
            execution_success: Whether strategy was successfully executed
            agent_feedback: Feedback from agents about strategy execution
            mission_progress: Progress on mission objectives
        """
        experience = {
            "step": self.episode_step,
            "strategy": strategy,
            "parameters": parameters,
            "execution_success": execution_success,
            "agent_feedback": agent_feedback,
            "mission_progress": mission_progress,
        }
        
        self.hq_experiences.append(experience)
        
        # Update mission progress
        self.mission_progress.update(mission_progress)
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[HQ EXPERIENCE] HQ reported experience: strategy={strategy}, success={execution_success}",
                level=logging.DEBUG,
            )
    
    def calculate_agent_reward(
        self,
        agent_id: str,
        base_reward: float,
        task_type: str,
        task_state: str,
        efficiency_score: float = 0.0,
        coordination_score: float = 0.0,
        adaptation_score: float = 0.0,
        survival_score: float = 0.0,
    ) -> float:
        """
        Calculate hierarchical reward for an agent based on multiple components.
        
        Args:
            agent_id: Unique identifier for the agent
            base_reward: Base reward from task execution
            task_type: Type of task performed
            task_state: State of task completion
            efficiency_score: How efficiently the task was completed
            coordination_score: How well the agent coordinated with others
            adaptation_score: How well the agent adapted to failures
            survival_score: Agent's survival and health maintenance
            
        Returns:
            float: Calculated hierarchical reward
        """
        config = utils_config.HIERARCHICAL_REWARD_CONFIG
        weights = config["agent_weights"]
        
        # Calculate component rewards
        task_completion_reward = self._calculate_task_completion_reward(
            base_reward, task_type, task_state
        )
        efficiency_reward = efficiency_score * weights[utils_config.RewardComponent.EFFICIENCY]
        coordination_reward = coordination_score * weights[utils_config.RewardComponent.COORDINATION]
        adaptation_reward = adaptation_score * weights[utils_config.RewardComponent.ADAPTATION]
        survival_reward = survival_score * weights[utils_config.RewardComponent.SURVIVAL]
        
        # Calculate HQ guidance bonus
        hq_guidance_bonus = self._calculate_hq_guidance_bonus(agent_id, task_type)
        
        # Combine rewards
        total_reward = (
            task_completion_reward +
            efficiency_reward +
            coordination_reward +
            adaptation_reward +
            survival_reward +
            hq_guidance_bonus
        )
        
        # Normalize reward
        max_reward = config["normalization"]["max_agent_reward"]
        total_reward = np.clip(total_reward, -max_reward, max_reward)
        
        # Store component breakdown
        self.agent_reward_components[agent_id] = {
            "task_completion": task_completion_reward,
            "efficiency": efficiency_reward,
            "coordination": coordination_reward,
            "adaptation": adaptation_reward,
            "survival": survival_reward,
            "hq_guidance": hq_guidance_bonus,
            "total": total_reward,
        }
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[AGENT REWARD] Agent {agent_id}: total={total_reward:.3f}, "
                f"task={task_completion_reward:.3f}, coord={coordination_reward:.3f}",
                level=logging.DEBUG,
            )
        
        return total_reward
    
    def calculate_hq_reward(
        self,
        strategy: str,
        parameters: Dict,
        execution_success: bool,
        agent_performance: Dict,
        mission_progress: Dict,
    ) -> float:
        """
        Calculate hierarchical reward for HQ based on strategic performance.
        
        Args:
            strategy: Strategy selected by HQ
            parameters: Strategy parameters
            execution_success: Whether strategy was successfully executed
            agent_performance: Performance metrics from agents
            mission_progress: Progress on mission objectives
            
        Returns:
            float: Calculated hierarchical reward
        """
        config = utils_config.HIERARCHICAL_REWARD_CONFIG
        weights = config["hq_weights"]
        
        # Calculate component rewards
        strategy_selection_reward = self._calculate_strategy_selection_reward(
            strategy, parameters, mission_progress
        )
        strategy_execution_reward = self._calculate_strategy_execution_reward(
            execution_success, agent_performance
        )
        resource_management_reward = self._calculate_resource_management_reward(
            parameters, agent_performance
        )
        agent_management_reward = self._calculate_agent_management_reward(
            parameters, agent_performance
        )
        threat_response_reward = self._calculate_threat_response_reward(
            strategy, agent_performance
        )
        mission_success_reward = self._calculate_mission_success_reward(
            mission_progress
        )
        
        # Calculate agent feedback bonus
        agent_feedback_bonus = self._calculate_agent_feedback_bonus(agent_performance)
        
        # Combine rewards
        total_reward = (
            strategy_selection_reward +
            strategy_execution_reward +
            resource_management_reward +
            agent_management_reward +
            threat_response_reward +
            mission_success_reward +
            agent_feedback_bonus
        )
        
        # Normalize reward
        max_reward = config["normalization"]["max_hq_reward"]
        total_reward = np.clip(total_reward, -max_reward, max_reward)
        
        # Store component breakdown
        self.hq_reward_components = {
            "strategy_selection": strategy_selection_reward,
            "strategy_execution": strategy_execution_reward,
            "resource_management": resource_management_reward,
            "agent_management": agent_management_reward,
            "threat_response": threat_response_reward,
            "mission_success": mission_success_reward,
            "agent_feedback": agent_feedback_bonus,
            "total": total_reward,
        }
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[HQ REWARD] Strategy {strategy}: total={total_reward:.3f}, "
                f"selection={strategy_selection_reward:.3f}, execution={strategy_execution_reward:.3f}",
                level=logging.INFO,
            )
        
        return total_reward
    
    def _calculate_task_completion_reward(
        self, base_reward: float, task_type: str, task_state: str
    ) -> float:
        """Calculate reward for task completion."""
        if task_state == utils_config.TaskState.SUCCESS:
            # Task-specific multipliers
            multipliers = {
                "gather": 1.0,
                "eliminate": 1.2,  # Higher reward for threat elimination
                "plant": 1.1,      # Moderate reward for resource planting
                "defend": 1.3,     # High reward for defensive actions
                "explore": 0.8,    # Lower reward for exploration
            }
            multiplier = multipliers.get(task_type, 1.0)
            return base_reward * multiplier
        elif task_state == utils_config.TaskState.FAILURE:
            return base_reward * 0.5  # Reduced penalty for failures
        else:
            return base_reward * 0.1  # Small reward for ongoing tasks
    
    def _calculate_hq_guidance_bonus(self, agent_id: str, task_type: str) -> float:
        """Calculate bonus for following HQ guidance effectively."""
        if not self.hq_experiences:
            return 0.0
        
        latest_hq = self.hq_experiences[-1]
        strategy = latest_hq["strategy"]
        parameters = latest_hq["parameters"]
        
        # Check if agent's task aligns with HQ strategy
        alignment_bonus = 0.0
        if strategy == "COLLECT_GOLD" and task_type == "gather":
            alignment_bonus = 0.2
        elif strategy == "ATTACK_THREATS" and task_type == "eliminate":
            alignment_bonus = 0.3
        elif strategy == "DEFEND_HQ" and task_type == "defend":
            alignment_bonus = 0.25
        
        # Scale by HQ parameters
        scaling = utils_config.HIERARCHICAL_REWARD_CONFIG["scaling"]["hq_to_agent_guidance"]
        return alignment_bonus * scaling
    
    def _calculate_strategy_selection_reward(
        self, strategy: str, parameters: Dict, mission_progress: Dict
    ) -> float:
        """Calculate reward for HQ strategy selection quality."""
        config = utils_config.HIERARCHICAL_REWARD_CONFIG
        weights = config["hq_weights"]
        
        # Base reward for strategy selection
        base_reward = weights[utils_config.RewardComponent.STRATEGY_SELECTION]
        
        # Bonus for selecting appropriate strategy based on mission progress
        progress_bonus = 0.0
        if strategy == "COLLECT_GOLD" and mission_progress.get("resource_collection", 0) < 0.5:
            progress_bonus = 0.3
        elif strategy == "ATTACK_THREATS" and mission_progress.get("threat_elimination", 0) < 0.8:
            progress_bonus = 0.4
        elif strategy == "DEFEND_HQ" and mission_progress.get("territory_control", 0) < 0.6:
            progress_bonus = 0.2
        
        return base_reward + progress_bonus
    
    def _calculate_strategy_execution_reward(
        self, execution_success: bool, agent_performance: Dict
    ) -> float:
        """Calculate reward for HQ strategy execution quality."""
        config = utils_config.HIERARCHICAL_REWARD_CONFIG
        weights = config["hq_weights"]
        
        if execution_success:
            base_reward = weights[utils_config.RewardComponent.STRATEGY_EXECUTION]
            
            # Bonus for high agent performance
            avg_agent_performance = np.mean(list(agent_performance.values())) if agent_performance else 0.0
            performance_bonus = avg_agent_performance * 0.5
            
            return base_reward + performance_bonus
        else:
            return -weights[utils_config.RewardComponent.STRATEGY_EXECUTION] * 0.5
    
    def _calculate_resource_management_reward(
        self, parameters: Dict, agent_performance: Dict
    ) -> float:
        """Calculate reward for HQ resource management."""
        config = utils_config.HIERARCHICAL_REWARD_CONFIG
        weights = config["hq_weights"]
        
        # Base reward for resource management
        base_reward = weights[utils_config.RewardComponent.RESOURCE_MANAGEMENT]
        
        # Bonus for efficient resource allocation
        efficiency_bonus = 0.0
        if "resource_threshold" in parameters:
            threshold = parameters["resource_threshold"]
            # Lower threshold = more aggressive resource collection = better management
            efficiency_bonus = (1.0 - threshold) * 0.2
        
        return base_reward + efficiency_bonus
    
    def _calculate_agent_management_reward(
        self, parameters: Dict, agent_performance: Dict
    ) -> float:
        """Calculate reward for HQ agent management."""
        config = utils_config.HIERARCHICAL_REWARD_CONFIG
        weights = config["hq_weights"]
        
        # Base reward for agent management
        base_reward = weights[utils_config.RewardComponent.AGENT_MANAGEMENT]
        
        # Bonus for good agent role distribution
        role_bonus = 0.0
        if "target_role" in parameters and agent_performance:
            target_role = parameters["target_role"]
            # Check if agents are performing well in their assigned roles
            role_performance = np.mean(list(agent_performance.values()))
            role_bonus = role_performance * 0.3
        
        return base_reward + role_bonus
    
    def _calculate_threat_response_reward(
        self, strategy: str, agent_performance: Dict
    ) -> float:
        """Calculate reward for HQ threat response."""
        config = utils_config.HIERARCHICAL_REWARD_CONFIG
        weights = config["hq_weights"]
        
        # Base reward for threat response
        base_reward = weights[utils_config.RewardComponent.THREAT_RESPONSE]
        
        # Bonus for appropriate threat response
        response_bonus = 0.0
        if strategy in ["ATTACK_THREATS", "DEFEND_HQ"]:
            # Check if agents are effectively responding to threats
            if agent_performance:
                avg_performance = np.mean(list(agent_performance.values()))
                response_bonus = avg_performance * 0.4
        
        return base_reward + response_bonus
    
    def _calculate_mission_success_reward(self, mission_progress: Dict) -> float:
        """Calculate reward for overall mission success."""
        config = utils_config.HIERARCHICAL_REWARD_CONFIG
        weights = config["hq_weights"]
        
        # Base reward for mission success
        base_reward = weights[utils_config.RewardComponent.MISSION_SUCCESS]
        
        # Calculate overall mission progress
        total_progress = np.mean(list(mission_progress.values())) if mission_progress else 0.0
        
        # Scale reward by mission progress
        progress_reward = total_progress * base_reward
        
        return progress_reward
    
    def _calculate_agent_feedback_bonus(self, agent_performance: Dict) -> float:
        """Calculate bonus based on agent performance feedback."""
        if not agent_performance:
            return 0.0
        
        avg_performance = np.mean(list(agent_performance.values()))
        scaling = utils_config.HIERARCHICAL_REWARD_CONFIG["scaling"]["agent_to_hq_feedback"]
        
        return avg_performance * scaling
    
    def get_coordination_score(self, agent_id: str, window_size: int = 10) -> float:
        """
        Calculate coordination score for an agent based on recent interactions.
        
        Args:
            agent_id: Unique identifier for the agent
            window_size: Number of recent steps to consider
            
        Returns:
            float: Coordination score between 0.0 and 1.0
        """
        if not self.coordination_history:
            return 0.0
        
        # Get recent coordination data for this agent
        recent_coordinations = [
            entry for entry in self.coordination_history
            if entry["agent_id"] == agent_id
        ][-window_size:]
        
        if not recent_coordinations:
            return 0.0
        
        # Calculate coordination score based on interaction frequency and quality
        interaction_count = len(recent_coordinations)
        max_interactions = window_size
        
        # Base score from interaction frequency
        frequency_score = min(interaction_count / max_interactions, 1.0)
        
        # Quality score from coordination data
        quality_scores = []
        for coord in recent_coordinations:
            data = coord["data"]
            if "successful_coordination" in data:
                quality_scores.append(1.0 if data["successful_coordination"] else 0.0)
            else:
                quality_scores.append(0.5)  # Neutral score if no quality data
        
        quality_score = np.mean(quality_scores) if quality_scores else 0.0
        
        # Combine frequency and quality
        coordination_score = (frequency_score * 0.6 + quality_score * 0.4)
        
        return coordination_score
    
    def get_adaptation_score(self, agent_id: str, window_size: int = 5) -> float:
        """
        Calculate adaptation score for an agent based on recent adaptive behavior.
        
        Args:
            agent_id: Unique identifier for the agent
            window_size: Number of recent steps to consider
            
        Returns:
            float: Adaptation score between 0.0 and 1.0
        """
        if not self.adaptation_history:
            return 0.0
        
        # Get recent adaptation data for this agent
        recent_adaptations = [
            entry for entry in self.adaptation_history
            if entry["agent_id"] == agent_id
        ][-window_size:]
        
        if not recent_adaptations:
            return 0.0
        
        # Calculate adaptation score based on adaptive behavior success
        adaptation_scores = []
        for adapt in recent_adaptations:
            data = adapt["data"]
            if "adaptive_strategy_success" in data:
                adaptation_scores.append(1.0 if data["adaptive_strategy_success"] else 0.0)
            elif "failure_recovery" in data:
                adaptation_scores.append(1.0 if data["failure_recovery"] else 0.0)
            else:
                adaptation_scores.append(0.0)  # No adaptation data
        
        adaptation_score = np.mean(adaptation_scores) if adaptation_scores else 0.0
        
        return adaptation_score
    
    def get_efficiency_score(
        self, agent_id: str, task_type: str, distance: float, time_taken: int
    ) -> float:
        """
        Calculate efficiency score for an agent's task execution.
        
        Args:
            agent_id: Unique identifier for the agent
            task_type: Type of task performed
            distance: Distance to target
            time_taken: Time taken to complete task
            
        Returns:
            float: Efficiency score between 0.0 and 1.0
        """
        # Base efficiency from distance (closer is better)
        distance_efficiency = max(0.0, 1.0 - (distance / 100.0))  # Normalize by max expected distance
        
        # Time efficiency (faster is better)
        expected_time = {
            "gather": 20,
            "eliminate": 30,
            "plant": 15,
            "defend": 10,
            "explore": 25,
        }
        expected = expected_time.get(task_type, 20)
        time_efficiency = max(0.0, 1.0 - (time_taken / (expected * 2)))  # Allow up to 2x expected time
        
        # Combine distance and time efficiency
        efficiency_score = (distance_efficiency * 0.6 + time_efficiency * 0.4)
        
        return efficiency_score
    
    def get_survival_score(self, agent_id: str, health: float, max_health: float = 100.0) -> float:
        """
        Calculate survival score for an agent based on health maintenance.
        
        Args:
            agent_id: Unique identifier for the agent
            health: Current health value
            max_health: Maximum health value
            
        Returns:
            float: Survival score between 0.0 and 1.0
        """
        health_ratio = health / max_health
        
        # Higher health = better survival score
        survival_score = health_ratio
        
        # Bonus for maintaining high health over time
        if health_ratio > 0.8:
            survival_score += 0.2
        elif health_ratio > 0.6:
            survival_score += 0.1
        
        return min(survival_score, 1.0)
    
    def update_episode_step(self, step: int):
        """Update the current episode step."""
        self.episode_step = step
    
    def reset_episode(self):
        """Reset the reward manager for a new episode."""
        self.agent_experiences.clear()
        self.hq_experiences.clear()
        self.coordination_history.clear()
        self.adaptation_history.clear()
        self.agent_reward_components.clear()
        self.hq_reward_components.clear()
        self.mission_progress = {
            "resource_collection": 0.0,
            "threat_elimination": 0.0,
            "territory_control": 0.0,
            "agent_coordination": 0.0,
        }
        self.episode_step = 0
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[HIERARCHICAL REWARDS] Reset episode for faction {self.faction_id}",
                level=logging.INFO,
            )
    
    def get_reward_summary(self) -> Dict:
        """
        Get a summary of all rewards calculated in the current episode.
        
        Returns:
            Dict: Summary of agent and HQ rewards
        """
        summary = {
            "faction_id": self.faction_id,
            "episode_step": self.episode_step,
            "agent_rewards": dict(self.agent_reward_components),
            "hq_rewards": dict(self.hq_reward_components),
            "mission_progress": dict(self.mission_progress),
            "total_agent_experiences": sum(len(exps) for exps in self.agent_experiences.values()),
            "total_hq_experiences": len(self.hq_experiences),
        }
        
        return summary

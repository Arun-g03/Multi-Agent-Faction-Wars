"""
Experience Sharing System for Multi-Agent Learning

This module implements a system for sharing experiences between agents on the same
faction, enabling collective learning and knowledge transfer.

The system provides:
1. Experience encoding and decoding
2. Similarity-based experience matching
3. Value-based experience selection
4. Collective learning from shared experiences
5. Knowledge transfer between agents

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
from typing import Dict, List, Tuple, Optional, Any
import math
import time


logger = Logger(log_file="ExperienceSharing_log.txt", log_level=logging.DEBUG)


class ExperienceEncoder(nn.Module):
    """
    Neural network for encoding experiences into compact representations
    that can be shared between agents.
    """
    
    def __init__(self, state_size: int, action_size: int, encoding_size: int = 32):
        """
        Initialize the experience encoder.
        
        Args:
            state_size: Size of agent state vector
            action_size: Size of action space
            encoding_size: Size of encoded experience representation
        """
        super(ExperienceEncoder, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.encoding_size = encoding_size
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, utils_config.EXPERIENCE_SHARING_CONFIG["state_encoding_size"]),
        )
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_size, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, utils_config.EXPERIENCE_SHARING_CONFIG["action_encoding_size"]),
        )
        
        # Context encoder (reward, outcome, etc.)
        self.context_encoder = nn.Sequential(
            nn.Linear(4, 16),  # reward, outcome, task_type, coordination_data
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, utils_config.EXPERIENCE_SHARING_CONFIG["context_encoding_size"]),
        )
        
        # Combined encoder
        total_input_size = (
            utils_config.EXPERIENCE_SHARING_CONFIG["state_encoding_size"] +
            utils_config.EXPERIENCE_SHARING_CONFIG["action_encoding_size"] +
            utils_config.EXPERIENCE_SHARING_CONFIG["context_encoding_size"]
        )
        
        self.combined_encoder = nn.Sequential(
            nn.Linear(total_input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, encoding_size),
            nn.Tanh(),  # Output in [-1, 1]
        )
        
        # Experience value estimator
        self.value_estimator = nn.Sequential(
            nn.Linear(encoding_size, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode an experience into a compact representation.
        
        Args:
            state: Agent state tensor
            action: Action tensor (one-hot or index)
            context: Context tensor [reward, outcome, task_type, coordination_data]
            
        Returns:
            Tuple of (encoded_experience, experience_value)
        """
        # Encode components
        state_encoded = self.state_encoder(state)
        action_encoded = self.action_encoder(action)
        context_encoded = self.context_encoder(context)
        
        # Combine encodings
        combined = torch.cat([state_encoded, action_encoded, context_encoded], dim=-1)
        
        # Generate final encoding
        experience_encoding = self.combined_encoder(combined)
        
        # Estimate experience value
        experience_value = self.value_estimator(experience_encoding)
        
        return experience_encoding, experience_value
    
    def decode_experience(self, encoding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode an experience encoding back to interpretable components.
        Note: This is a simplified decoder for analysis purposes.
        
        Args:
            encoding: Encoded experience tensor
            
        Returns:
            Dictionary containing decoded components
        """
        # This would require a proper decoder network in a full implementation
        # For now, we'll return the encoding as-is for analysis
        return {
            "encoding": encoding,
            "value": self.value_estimator(encoding),
        }


class ExperienceSharingSystem:
    """
    System for sharing experiences between agents on the same faction,
    enabling collective learning and knowledge transfer.
    """
    
    def __init__(self, faction_id: str, agents: List[Any]):
        """
        Initialize the experience sharing system.
        
        Args:
            faction_id: Unique identifier for the faction
            agents: List of agents in the faction
        """
        self.faction_id = faction_id
        self.agents = agents
        
        # Experience encoders for each agent
        self.experience_encoders = {}
        self.shared_experiences = defaultdict(list)  # agent_id -> list of shared experiences
        self.collective_memory = deque(maxlen=utils_config.COLLECTIVE_LEARNING_CONFIG["collective_memory_size"])
        
        # Learning parameters
        self.shared_learning_rate = utils_config.EXPERIENCE_SHARING_CONFIG["shared_learning_rate"]
        self.experience_weight_decay = utils_config.EXPERIENCE_SHARING_CONFIG["experience_weight_decay"]
        
        # Performance tracking
        self.sharing_success_rate = defaultdict(float)
        self.learning_success_rate = defaultdict(float)
        self.collective_performance = defaultdict(list)
        
        # Initialize experience encoders for each agent
        for agent in agents:
            agent_id = agent.agent_id
            
            # Get agent's state and action sizes
            state_size = getattr(agent, 'state_size', utils_config.DEF_AGENT_STATE_SIZE)
            action_size = len(getattr(agent, 'role_actions', []))
            
            self.experience_encoders[agent_id] = ExperienceEncoder(
                state_size=state_size,
                action_size=action_size,
                encoding_size=utils_config.EXPERIENCE_SHARING_CONFIG["experience_encoding_size"]
            )
            
            # Initialize performance tracking
            self.sharing_success_rate[agent_id] = 0.5
            self.learning_success_rate[agent_id] = 0.5
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[EXPERIENCE SHARING] Initialized for faction {faction_id} with {len(agents)} agents",
                level=logging.INFO,
            )
    
    def encode_experience(
        self,
        agent_id: str,
        state: Any,
        action: Any,
        reward: float,
        outcome: str,
        task_type: str = None,
        coordination_data: Dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Encode an experience for sharing.
        
        Args:
            agent_id: Agent identifier
            state: Agent state
            action: Action taken
            reward: Reward received
            outcome: Outcome of the action
            task_type: Type of task
            coordination_data: Coordination information
            
        Returns:
            Tuple of (encoded_experience, experience_value, experience_metadata)
        """
        if agent_id not in self.experience_encoders:
            return None, None, None
        
        encoder = self.experience_encoders[agent_id]
        
        # Convert inputs to tensors
        state_tensor = torch.tensor(state, dtype=torch.float32) if not isinstance(state, torch.Tensor) else state
        action_tensor = torch.tensor(action, dtype=torch.float32) if not isinstance(action, torch.Tensor) else action
        
        # Create context tensor
        context_values = [
            reward,  # Normalized reward
            float(outcome == "success"),  # Success indicator
            float(task_type == "gather") if task_type else 0.0,  # Task type indicator
            coordination_data.get("coordination_score", 0.0) if coordination_data else 0.0,  # Coordination score
        ]
        context_tensor = torch.tensor(context_values, dtype=torch.float32)
        
        # Encode experience
        with torch.no_grad():
            encoding, value = encoder(state_tensor.unsqueeze(0), action_tensor.unsqueeze(0), context_tensor.unsqueeze(0))
        
        # Create experience metadata
        metadata = {
            "agent_id": agent_id,
            "timestamp": time.time(),
            "reward": reward,
            "outcome": outcome,
            "task_type": task_type,
            "coordination_data": coordination_data or {},
            "encoding": encoding.squeeze(0),
            "value": value.squeeze(0).item(),
        }
        
        return encoding.squeeze(0), value.squeeze(0).item(), metadata
    
    def should_share_experience(self, agent_id: str, experience_metadata: Dict[str, Any]) -> bool:
        """
        Determine if an experience should be shared.
        
        Args:
            agent_id: Agent identifier
            experience_metadata: Experience metadata
            
        Returns:
            True if experience should be shared
        """
        # Check experience value threshold
        value_threshold = utils_config.EXPERIENCE_SHARING_CONFIG["value_threshold"]
        if experience_metadata["value"] < value_threshold:
            return False
        
        # Check sharing success rate (agents with low success rate share less)
        success_rate = self.sharing_success_rate[agent_id]
        if success_rate < 0.3:
            return np.random.random() < 0.3
        elif success_rate > 0.7:
            return np.random.random() < 0.8
        else:
            return np.random.random() < 0.5
    
    def find_similar_experiences(self, agent_id: str, experience_encoding: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Find similar experiences from other agents.
        
        Args:
            agent_id: Agent identifier
            experience_encoding: Encoded experience to match
            
        Returns:
            List of similar experiences
        """
        similar_experiences = []
        similarity_threshold = utils_config.EXPERIENCE_SHARING_CONFIG["similarity_threshold"]
        
        # Search in collective memory
        for experience in self.collective_memory:
            if experience["agent_id"] != agent_id:  # Don't match with self
                similarity = self._calculate_experience_similarity(experience_encoding, experience["encoding"])
                if similarity >= similarity_threshold:
                    similar_experiences.append({
                        **experience,
                        "similarity": similarity,
                    })
        
        # Sort by similarity and value
        similar_experiences.sort(key=lambda x: x["similarity"] * x["value"], reverse=True)
        
        return similar_experiences[:5]  # Return top 5 similar experiences
    
    def _calculate_experience_similarity(self, encoding1: torch.Tensor, encoding2: torch.Tensor) -> float:
        """
        Calculate similarity between two experience encodings.
        
        Args:
            encoding1: First experience encoding
            encoding2: Second experience encoding
            
        Returns:
            Similarity score between 0 and 1
        """
        # Use cosine similarity
        similarity = F.cosine_similarity(encoding1.unsqueeze(0), encoding2.unsqueeze(0), dim=1)
        return similarity.item()
    
    def share_experience(self, agent_id: str, experience_metadata: Dict[str, Any]) -> bool:
        """
        Share an experience with other agents.
        
        Args:
            agent_id: Agent identifier
            experience_metadata: Experience metadata
            
        Returns:
            True if sharing was successful
        """
        try:
            # Add to collective memory
            self.collective_memory.append(experience_metadata)
            
            # Find nearby agents to share with
            sharing_range = utils_config.EXPERIENCE_SHARING_CONFIG["sharing_range"]
            nearby_agents = self._find_nearby_agents(agent_id, sharing_range)
            
            # Share with nearby agents
            shared_count = 0
            for nearby_agent in nearby_agents:
                if nearby_agent.agent_id != agent_id:
                    self.shared_experiences[nearby_agent.agent_id].append(experience_metadata)
                    shared_count += 1
            
            # Update sharing success rate
            self._update_sharing_success_rate(agent_id, True)
            
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[EXPERIENCE SHARING] Agent {agent_id} shared experience with {shared_count} nearby agents",
                    level=logging.DEBUG,
                )
            
            return True
            
        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[EXPERIENCE SHARING ERROR] Failed to share experience from {agent_id}: {e}",
                    level=logging.ERROR,
                )
            
            self._update_sharing_success_rate(agent_id, False)
            return False
    
    def learn_from_shared_experiences(self, agent_id: str) -> float:
        """
        Learn from shared experiences.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Learning reward
        """
        if agent_id not in self.shared_experiences:
            return 0.0
        
        shared_experiences = self.shared_experiences[agent_id]
        if not shared_experiences:
            return 0.0
        
        learning_reward = 0.0
        successful_learnings = 0
        
        # Process shared experiences
        for experience in shared_experiences[:10]:  # Limit to recent experiences
            try:
                # Calculate learning value based on experience similarity and value
                learning_value = experience["value"] * 0.1  # Scale down for learning reward
                learning_reward += learning_value
                successful_learnings += 1
                
            except Exception as e:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[EXPERIENCE LEARNING ERROR] Failed to learn from experience: {e}",
                        level=logging.ERROR,
                    )
        
        # Update learning success rate
        if successful_learnings > 0:
            self._update_learning_success_rate(agent_id, True)
            learning_reward += utils_config.EXPERIENCE_SHARING_CONFIG["sharing_rewards"]["learning_from_others"]
        else:
            self._update_learning_success_rate(agent_id, False)
        
        # Clear processed experiences
        self.shared_experiences[agent_id] = []
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[EXPERIENCE LEARNING] Agent {agent_id} learned from {successful_learnings} shared experiences",
                level=logging.DEBUG,
            )
        
        return learning_reward
    
    def _find_nearby_agents(self, agent_id: str, range_distance: float) -> List[Any]:
        """
        Find agents within sharing range.
        
        Args:
            agent_id: Agent identifier
            range_distance: Maximum distance for sharing
            
        Returns:
            List of nearby agents
        """
        # Find the source agent
        source_agent = None
        for agent in self.agents:
            if agent.agent_id == agent_id:
                source_agent = agent
                break
        
        if not source_agent:
            return []
        
        # Find nearby agents
        nearby_agents = []
        for agent in self.agents:
            if agent != source_agent:
                distance = math.sqrt((source_agent.x - agent.x)**2 + (source_agent.y - agent.y)**2)
                if distance <= range_distance:
                    nearby_agents.append(agent)
        
        return nearby_agents
    
    def _update_sharing_success_rate(self, agent_id: str, success: bool):
        """Update sharing success rate for an agent."""
        current_rate = self.sharing_success_rate[agent_id]
        alpha = 0.1
        new_rate = (1 - alpha) * current_rate + alpha * (1.0 if success else 0.0)
        self.sharing_success_rate[agent_id] = new_rate
    
    def _update_learning_success_rate(self, agent_id: str, success: bool):
        """Update learning success rate for an agent."""
        current_rate = self.learning_success_rate[agent_id]
        alpha = 0.1
        new_rate = (1 - alpha) * current_rate + alpha * (1.0 if success else 0.0)
        self.learning_success_rate[agent_id] = new_rate
    
    def get_collective_learning_reward(self, agent_id: str) -> float:
        """
        Calculate collective learning reward for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Collective learning reward
        """
        sharing_rate = self.sharing_success_rate[agent_id]
        learning_rate = self.learning_success_rate[agent_id]
        
        # Base reward from sharing and learning success
        base_reward = (sharing_rate + learning_rate) / 2.0
        
        # Bonus for high collective performance
        if sharing_rate > 0.8 and learning_rate > 0.8:
            base_reward += utils_config.EXPERIENCE_SHARING_CONFIG["sharing_rewards"]["collective_improvement"]
        
        return base_reward
    
    def get_experience_sharing_summary(self) -> Dict[str, Any]:
        """
        Get summary of experience sharing system performance.
        
        Returns:
            Dictionary containing performance metrics
        """
        return {
            "faction_id": self.faction_id,
            "total_agents": len(self.agents),
            "collective_memory_size": len(self.collective_memory),
            "sharing_success_rates": dict(self.sharing_success_rate),
            "learning_success_rates": dict(self.learning_success_rate),
            "avg_sharing_success": np.mean(list(self.sharing_success_rate.values())),
            "avg_learning_success": np.mean(list(self.learning_success_rate.values())),
            "total_shared_experiences": sum(len(exps) for exps in self.shared_experiences.values()),
        }
    
    def reset_episode(self):
        """Reset the experience sharing system for a new episode."""
        self.shared_experiences.clear()
        self.collective_memory.clear()
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[EXPERIENCE SHARING] Reset episode for faction {self.faction_id}",
                level=logging.INFO,
            )

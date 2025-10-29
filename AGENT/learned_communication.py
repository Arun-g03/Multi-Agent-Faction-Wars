"""
Learned Communication Network for Multi-Agent Coordination

This module implements a neural network-based communication system that enables
agents to learn effective communication patterns and coordination strategies.

The system provides:
1. Learned message encoding and decoding
2. Adaptive communication frequency
3. Coordination strategy learning
4. Communication success tracking
5. Emergent coordination patterns

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


logger = Logger(log_file="LearnedCommunication_log.txt", log_level=logging.DEBUG)


class CommunicationNetwork(nn.Module):
    """
    Neural network for learned communication between agents.
    Handles message encoding, decoding, and coordination strategy learning.
    """

    def __init__(self, state_size: int, message_size: int, hidden_size: int = 128):
        """
        Initialize the communication network.

        Args:
            state_size: Size of agent state vector
            message_size: Size of message encoding
            hidden_size: Size of hidden layers
        """
        super(CommunicationNetwork, self).__init__()

        self.state_size = state_size
        self.message_size = message_size
        self.hidden_size = hidden_size

        # Message encoder: converts agent state to message
        self.message_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, message_size),
            nn.Tanh(),  # Output in [-1, 1]
        )

        # Message decoder: converts message back to state information
        self.message_decoder = nn.Sequential(
            nn.Linear(message_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, state_size),
        )

        # Communication decision network: decides when and what to communicate
        self.communication_decision = nn.Sequential(
            nn.Linear(state_size + message_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(
                hidden_size // 2, len(utils_config.CommunicationType) + 1
            ),  # +1 for "no communication"
            nn.Softmax(dim=-1),
        )

        # Coordination strategy network: learns coordination patterns
        self.coordination_strategy = nn.Sequential(
            nn.Linear(state_size + message_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, len(utils_config.CoordinationStrategy)),
            nn.Softmax(dim=-1),
        )

        # Message importance scorer: scores message importance
        self.message_importance = nn.Sequential(
            nn.Linear(message_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[COMMUNICATION NETWORK] Initialized with state_size={state_size}, "
                f"message_size={message_size}, hidden_size={hidden_size}",
                level=logging.INFO,
            )

    def encode_message(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode agent state into a message.

        Args:
            state: Agent state tensor

        Returns:
            Encoded message tensor
        """
        return self.message_encoder(state)

    def decode_message(self, message: torch.Tensor) -> torch.Tensor:
        """
        Decode message back to state information.

        Args:
            message: Encoded message tensor

        Returns:
            Decoded state information tensor
        """
        return self.message_decoder(message)

    def decide_communication(
        self, state: torch.Tensor, message: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decide whether and what type of communication to send.

        Args:
            state: Current agent state
            message: Encoded message

        Returns:
            Tuple of (communication_type_probs, coordination_strategy_probs)
        """
        combined_input = torch.cat([state, message], dim=-1)
        comm_probs = self.communication_decision(combined_input)
        coord_probs = self.coordination_strategy(combined_input)

        return comm_probs, coord_probs

    def score_message_importance(self, message: torch.Tensor) -> torch.Tensor:
        """
        Score the importance of a message.

        Args:
            message: Encoded message tensor

        Returns:
            Importance score tensor
        """
        return self.message_importance(message)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the communication network.

        Args:
            state: Agent state tensor

        Returns:
            Dictionary containing all network outputs
        """
        message = self.encode_message(state)
        comm_probs, coord_probs = self.decide_communication(state, message)
        importance = self.score_message_importance(message)

        return {
            "message": message,
            "communication_probs": comm_probs,
            "coordination_probs": coord_probs,
            "importance": importance,
        }


class LearnedCommunicationSystem:
    """
    Learned communication system that manages communication between agents
    and learns effective coordination strategies.
    """

    def __init__(self, faction_id: str, agents: List[Any]):
        """
        Initialize the learned communication system.

        Args:
            faction_id: Unique identifier for the faction
            agents: List of agents in the faction
        """
        self.faction_id = faction_id
        self.agents = agents

        # Communication networks for each agent
        self.communication_networks = {}
        self.message_queues = defaultdict(deque)
        self.communication_history = defaultdict(list)
        self.coordination_history = defaultdict(list)

        # Learning parameters
        self.learning_rate = utils_config.COMMUNICATION_CONFIG["learning_rate"]
        self.memory_size = utils_config.COMMUNICATION_CONFIG["memory_size"]
        self.communication_memory = deque(maxlen=self.memory_size)

        # Performance tracking
        self.communication_success_rate = defaultdict(float)
        self.coordination_success_rate = defaultdict(float)
        self.message_importance_scores = defaultdict(list)

        # Initialize communication networks for each agent
        for agent in agents:
            agent_id = agent.agent_id
            state_size = utils_config.COMMUNICATION_STATE_SIZE
            message_size = utils_config.COMMUNICATION_CONFIG["message_encoding_size"]

            self.communication_networks[agent_id] = CommunicationNetwork(
                state_size=state_size,
                message_size=message_size,
            )

            # Initialize performance tracking
            self.communication_success_rate[agent_id] = (
                0.5  # Start with neutral success rate
            )
            self.coordination_success_rate[agent_id] = 0.5

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[LEARNED COMMUNICATION] Initialized for faction {faction_id} with {len(agents)} agents",
                level=logging.INFO,
            )

    def get_communication_state(self, agent: Any) -> torch.Tensor:
        """
        Get communication state for an agent.

        Args:
            agent: Agent instance

        Returns:
            Communication state tensor
        """
        # Agent position (normalized)
        pos_x = agent.x / utils_config.WORLD_WIDTH
        pos_y = agent.y / utils_config.WORLD_HEIGHT

        # Agent health (normalized)
        health = agent.Health / 100.0

        # Agent role (one-hot)
        role_gatherer = 1.0 if agent.role == "gatherer" else 0.0
        role_peacekeeper = 1.0 if agent.role == "peacekeeper" else 0.0

        # Current task (one-hot)
        task_none = 1.0 if not agent.current_task else 0.0
        task_gather = (
            1.0
            if agent.current_task and agent.current_task.get("type") == "gather"
            else 0.0
        )
        task_eliminate = (
            1.0
            if agent.current_task and agent.current_task.get("type") == "eliminate"
            else 0.0
        )
        task_defend = (
            1.0
            if agent.current_task and agent.current_task.get("type") == "defend"
            else 0.0
        )

        # Task progress
        task_progress = 0.0
        if agent.current_task and hasattr(agent, "current_task_state"):
            if agent.current_task_state == utils_config.TaskState.SUCCESS:
                task_progress = 1.0
            elif agent.current_task_state == utils_config.TaskState.ONGOING:
                task_progress = 0.5
            elif agent.current_task_state == utils_config.TaskState.FAILURE:
                task_progress = 0.0

        # Nearby agents count (normalized)
        nearby_agents = 0.0
        if hasattr(agent, "faction") and agent.faction:
            other_agents = [a for a in agent.faction.agents if a != agent]
            nearby_count = 0
            for other_agent in other_agents:
                distance = math.sqrt(
                    (agent.x - other_agent.x) ** 2 + (agent.y - other_agent.y) ** 2
                )
                if distance <= utils_config.COMMUNICATION_CONFIG["communication_range"]:
                    nearby_count += 1
            nearby_agents = min(nearby_count / 5.0, 1.0)  # Normalize by max expected

        # Communication history (recent success rate)
        comm_history = self.communication_success_rate.get(agent.agent_id, 0.5)

        # Coordination score
        coord_score = self.coordination_success_rate.get(agent.agent_id, 0.5)

        # Emergency level (based on health and threats)
        emergency_level = 0.0
        if health < 0.3:  # Low health
            emergency_level += 0.5
        if hasattr(agent, "nearest_threat") and agent.nearest_threat:
            threat_distance = math.sqrt(
                (agent.x - agent.nearest_threat["location"][0]) ** 2
                + (agent.y - agent.nearest_threat["location"][1]) ** 2
            )
            if threat_distance < 50:  # Very close threat
                emergency_level += 0.5
        emergency_level = min(emergency_level, 1.0)

        # Resource availability in area
        resource_availability = 0.0
        if hasattr(agent, "nearest_resource") and agent.nearest_resource:
            resource_distance = math.sqrt(
                (agent.x - agent.nearest_resource.x) ** 2
                + (agent.y - agent.nearest_resource.y) ** 2
            )
            resource_availability = max(0.0, 1.0 - resource_distance / 200.0)

        # Construct state vector
        state = torch.tensor(
            [
                pos_x,
                pos_y,  # Agent position
                health,  # Agent health
                role_gatherer,
                role_peacekeeper,  # Agent role
                task_none,
                task_gather,
                task_eliminate,
                task_defend,  # Current task
                task_progress,  # Task progress
                nearby_agents,  # Nearby agents count
                comm_history,  # Communication history
                coord_score,  # Coordination score
                emergency_level,  # Emergency level
                resource_availability,  # Resource availability
            ],
            dtype=torch.float32,
        )

        return state

    def should_communicate(self, agent: Any, current_step: int) -> bool:
        """
        Determine if an agent should attempt communication.

        Args:
            agent: Agent instance
            current_step: Current simulation step

        Returns:
            True if agent should communicate
        """
        agent_id = agent.agent_id

        # Check communication frequency
        comm_freq = utils_config.COMMUNICATION_CONFIG["communication_frequency"]
        if current_step % comm_freq != 0:
            return False

        # Check if agent has pending messages
        if (
            len(self.message_queues[agent_id])
            >= utils_config.COMMUNICATION_CONFIG["message_queue_size"]
        ):
            return False

        # Check communication success rate (agents with low success rate communicate less)
        success_rate = self.communication_success_rate[agent_id]
        if success_rate < 0.3:  # Low success rate
            return np.random.random() < 0.3  # 30% chance
        elif success_rate > 0.7:  # High success rate
            return np.random.random() < 0.8  # 80% chance
        else:
            return np.random.random() < 0.5  # 50% chance

    def generate_message(self, agent: Any) -> Optional[Dict[str, Any]]:
        """
        Generate a message for an agent to send.

        Args:
            agent: Agent instance

        Returns:
            Message dictionary or None if no communication
        """
        agent_id = agent.agent_id

        if agent_id not in self.communication_networks:
            return None

        # Get communication state
        state = self.get_communication_state(agent)

        # Get network outputs
        network = self.communication_networks[agent_id]
        with torch.no_grad():
            outputs = network(state.unsqueeze(0))

        message = outputs["message"].squeeze(0)
        comm_probs = outputs["communication_probs"].squeeze(0)
        coord_probs = outputs["coordination_probs"].squeeze(0)
        importance = outputs["importance"].squeeze(0).item()

        # Decide communication type
        comm_type_idx = torch.argmax(comm_probs).item()
        if comm_type_idx >= len(utils_config.CommunicationType):
            return None  # "No communication" selected

        comm_type = list(utils_config.CommunicationType)[comm_type_idx]

        # Decide coordination strategy
        coord_strategy_idx = torch.argmax(coord_probs).item()
        coord_strategy = list(utils_config.CoordinationStrategy)[coord_strategy_idx]

        # Create message
        message_dict = {
            "sender_id": agent_id,
            "type": comm_type,
            "content": message.tolist(),
            "importance": importance,
            "coordination_strategy": coord_strategy,
            "timestamp": time.time(),
            "expiry_steps": utils_config.COMMUNICATION_CONFIG["message_types"][
                comm_type
            ]["expiry_steps"],
        }

        return message_dict

    def process_message(self, receiver: Any, message: Dict[str, Any]) -> bool:
        """
        Process a received message.

        Args:
            receiver: Receiving agent
            message: Message dictionary

        Returns:
            True if message was successfully processed
        """
        receiver_id = receiver.agent_id
        sender_id = message["sender_id"]
        comm_type = message["type"]

        try:
            # Decode message content
            if receiver_id in self.communication_networks:
                network = self.communication_networks[receiver_id]
                message_tensor = torch.tensor(message["content"], dtype=torch.float32)

                with torch.no_grad():
                    decoded_info = network.decode_message(message_tensor.unsqueeze(0))

                # Process based on communication type
                if comm_type == utils_config.CommunicationType.RESOURCE_SHARING:
                    self._process_resource_sharing(receiver, decoded_info)
                elif comm_type == utils_config.CommunicationType.THREAT_WARNING:
                    self._process_threat_warning(receiver, decoded_info)
                elif comm_type == utils_config.CommunicationType.TASK_COORDINATION:
                    self._process_task_coordination(receiver, decoded_info)
                elif comm_type == utils_config.CommunicationType.HELP_REQUEST:
                    self._process_help_request(receiver, decoded_info)
                elif comm_type == utils_config.CommunicationType.STATUS_UPDATE:
                    self._process_status_update(receiver, decoded_info)
                elif comm_type == utils_config.CommunicationType.STRATEGY_SYNC:
                    self._process_strategy_sync(receiver, decoded_info)
                elif comm_type == utils_config.CommunicationType.FORMATION_REQUEST:
                    self._process_formation_request(receiver, decoded_info)
                elif comm_type == utils_config.CommunicationType.EMERGENCY_ALERT:
                    self._process_emergency_alert(receiver, decoded_info)

                # Update communication success rate
                self._update_communication_success_rate(sender_id, True)
                self._update_communication_success_rate(receiver_id, True)

                # Store in communication history
                self.communication_history[receiver_id].append(
                    {
                        "sender": sender_id,
                        "type": comm_type,
                        "success": True,
                        "timestamp": message["timestamp"],
                    }
                )

                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[COMMUNICATION] Agent {receiver_id} successfully processed {comm_type.value} from {sender_id}",
                        level=logging.DEBUG,
                    )

                return True

        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[COMMUNICATION ERROR] Failed to process message from {sender_id} to {receiver_id}: {e}",
                    level=logging.ERROR,
                )

            # Update communication success rate
            self._update_communication_success_rate(sender_id, False)
            self._update_communication_success_rate(receiver_id, False)

            return False

    def _process_resource_sharing(self, receiver: Any, decoded_info: torch.Tensor):
        """Process resource sharing message."""
        # Extract resource information from decoded message
        # This would update receiver's known resources
        pass

    def _process_threat_warning(self, receiver: Any, decoded_info: torch.Tensor):
        """Process threat warning message."""
        # Extract threat information from decoded message
        # This would update receiver's known threats
        pass

    def _process_task_coordination(self, receiver: Any, decoded_info: torch.Tensor):
        """Process task coordination message."""
        # Extract task coordination information
        # This would influence receiver's task selection
        pass

    def _process_help_request(self, receiver: Any, decoded_info: torch.Tensor):
        """Process help request message."""
        # Extract help request information
        # This would add to receiver's help requests
        pass

    def _process_status_update(self, receiver: Any, decoded_info: torch.Tensor):
        """Process status update message."""
        # Extract status information
        # This would update receiver's knowledge of other agents
        pass

    def _process_strategy_sync(self, receiver: Any, decoded_info: torch.Tensor):
        """Process strategy synchronization message."""
        # Extract strategy information
        # This would influence receiver's strategy selection
        pass

    def _process_formation_request(self, receiver: Any, decoded_info: torch.Tensor):
        """Process formation request message."""
        # Extract formation information
        # This would influence receiver's positioning
        pass

    def _process_emergency_alert(self, receiver: Any, decoded_info: torch.Tensor):
        """Process emergency alert message."""
        # Extract emergency information
        # This would trigger emergency response behavior
        pass

    def _update_communication_success_rate(self, agent_id: str, success: bool):
        """Update communication success rate for an agent."""
        current_rate = self.communication_success_rate[agent_id]
        alpha = 0.1  # Learning rate for success rate update
        new_rate = (1 - alpha) * current_rate + alpha * (1.0 if success else 0.0)
        self.communication_success_rate[agent_id] = new_rate

    def update_coordination_success(self, agent_id: str, success: bool):
        """Update coordination success rate for an agent."""
        current_rate = self.coordination_success_rate[agent_id]
        alpha = 0.1
        new_rate = (1 - alpha) * current_rate + alpha * (1.0 if success else 0.0)
        self.coordination_success_rate[agent_id] = new_rate

    def get_coordination_reward(self, agent_id: str) -> float:
        """
        Calculate coordination reward for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Coordination reward value
        """
        comm_success = self.communication_success_rate[agent_id]
        coord_success = self.coordination_success_rate[agent_id]

        # Base reward from communication and coordination success
        base_reward = (comm_success + coord_success) / 2.0

        # Bonus for high coordination
        if coord_success > 0.8:
            base_reward += 0.2

        # Bonus for efficient communication
        if comm_success > 0.7:
            base_reward += 0.1

        return base_reward

    def get_communication_summary(self) -> Dict[str, Any]:
        """
        Get summary of communication system performance.

        Returns:
            Dictionary containing performance metrics
        """
        return {
            "faction_id": self.faction_id,
            "total_agents": len(self.agents),
            "communication_success_rates": dict(self.communication_success_rate),
            "coordination_success_rates": dict(self.coordination_success_rate),
            "avg_communication_success": np.mean(
                list(self.communication_success_rate.values())
            ),
            "avg_coordination_success": np.mean(
                list(self.coordination_success_rate.values())
            ),
            "total_messages_sent": sum(
                len(history) for history in self.communication_history.values()
            ),
        }

    def reset_episode(self):
        """Reset the communication system for a new episode."""
        self.message_queues.clear()
        self.communication_history.clear()
        self.coordination_history.clear()
        self.communication_memory.clear()

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[LEARNED COMMUNICATION] Reset episode for faction {self.faction_id}",
                level=logging.INFO,
            )

"""
Learned State Representation System for HQ

This module implements a system for learning sophisticated state representations
that enable the HQ to develop better understanding of the game state and make
more informed strategic decisions.

The system provides:
1. Multi-modal state encoders (convolutional, recurrent, transformer, attention)
2. Abstract concept learning
3. Temporal pattern recognition
4. Spatial relationship learning
5. Causal relationship modeling
6. Hierarchical feature learning

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


logger = Logger(log_file="LearnedStateRepresentation_log.txt", log_level=logging.DEBUG)


class StateEncoder(nn.Module):
    """
    Base class for state encoders that learn different aspects of the game state.
    """

    def __init__(
        self, input_size: int, output_size: int, encoder_type: str = "attention"
    ):
        """
        Initialize the state encoder.

        Args:
            input_size: Size of input state vector
            output_size: Size of output representation
            encoder_type: Type of encoder to use
        """
        super(StateEncoder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.encoder_type = encoder_type

        if encoder_type == "attention":
            self._build_attention_encoder()
        elif encoder_type == "convolutional":
            self._build_convolutional_encoder()
        elif encoder_type == "recurrent":
            self._build_recurrent_encoder()
        elif encoder_type == "transformer":
            self._build_transformer_encoder()
        elif encoder_type == "autoencoder":
            self._build_autoencoder()
        else:
            self._build_default_encoder()

    def _build_attention_encoder(self):
        """Build attention-based encoder."""
        # Ensure embed_dim is divisible by num_heads
        num_heads = utils_config.LEARNED_STATE_CONFIG["num_attention_heads"]
        embed_dim = ((self.input_size + num_heads - 1) // num_heads) * num_heads

        # Project input to embed_dim if needed
        if embed_dim != self.input_size:
            self.input_projection = nn.Linear(self.input_size, embed_dim)
        else:
            self.input_projection = nn.Identity()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=utils_config.LEARNED_STATE_CONFIG["attention_dropout"],
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, self.output_size),
            nn.LayerNorm(self.output_size),
            nn.ReLU(),
        )

    def _build_convolutional_encoder(self):
        """Build convolutional encoder."""
        # Reshape input to 2D for convolution
        self.input_projection = nn.Linear(self.input_size, 64)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.output_projection = nn.Sequential(
            nn.Linear(64, self.output_size),
            nn.LayerNorm(self.output_size),
            nn.ReLU(),
        )

    def _build_recurrent_encoder(self):
        """Build recurrent encoder (LSTM)."""
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
        )

        self.output_projection = nn.Sequential(
            nn.Linear(128, self.output_size),
            nn.LayerNorm(self.output_size),
            nn.ReLU(),
        )

    def _build_transformer_encoder(self):
        """Build transformer encoder."""
        # Ensure embed_dim is divisible by num_heads
        num_heads = utils_config.LEARNED_STATE_CONFIG["num_attention_heads"]
        embed_dim = ((self.input_size + num_heads - 1) // num_heads) * num_heads

        # Project input to embed_dim if needed
        if embed_dim != self.input_size:
            self.input_projection = nn.Linear(self.input_size, embed_dim)
        else:
            self.input_projection = nn.Identity()

        self.positional_encoding = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, self.output_size),
            nn.LayerNorm(self.output_size),
            nn.ReLU(),
        )

    def _build_autoencoder(self):
        """Build autoencoder."""
        encoder_layers = utils_config.LEARNED_STATE_CONFIG["encoder_types"][
            utils_config.StateEncoderType.AUTOENCODER
        ]["encoder_layers"]
        decoder_layers = utils_config.LEARNED_STATE_CONFIG["encoder_types"][
            utils_config.StateEncoderType.AUTOENCODER
        ]["decoder_layers"]

        # Encoder
        encoder_modules = []
        prev_size = self.input_size
        for layer_size in encoder_layers:
            encoder_modules.extend(
                [
                    nn.Linear(prev_size, layer_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ]
            )
            prev_size = layer_size

        self.encoder = nn.Sequential(*encoder_modules)

        # Decoder
        decoder_modules = []
        prev_size = encoder_layers[-1]
        for layer_size in decoder_layers:
            decoder_modules.extend(
                [
                    nn.Linear(prev_size, layer_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ]
            )
            prev_size = layer_size

        self.decoder = nn.Sequential(*decoder_modules)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(encoder_layers[-1], self.output_size),
            nn.LayerNorm(self.output_size),
            nn.ReLU(),
        )

    def _build_default_encoder(self):
        """Build default feedforward encoder."""
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.output_size),
            nn.LayerNorm(self.output_size),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x: Input state tensor

        Returns:
            Encoded state representation
        """
        if self.encoder_type == "attention":
            return self._forward_attention(x)
        elif self.encoder_type == "convolutional":
            return self._forward_convolutional(x)
        elif self.encoder_type == "recurrent":
            return self._forward_recurrent(x)
        elif self.encoder_type == "transformer":
            return self._forward_transformer(x)
        elif self.encoder_type == "autoencoder":
            return self._forward_autoencoder(x)
        else:
            return self._forward_default(x)

    def _forward_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for attention encoder."""
        # Project input to embed_dim
        x = self.input_projection(x)

        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        # Output projection
        return self.output_projection(x)

    def _forward_convolutional(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for convolutional encoder."""
        # Project to 2D
        x = self.input_projection(x)

        # Reshape for convolution
        x = x.unsqueeze(1)  # Add channel dimension

        # Convolutional layers
        x = self.conv_layers(x)

        # Flatten and project
        x = x.squeeze(-1)
        return self.output_projection(x)

    def _forward_recurrent(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for recurrent encoder."""
        # LSTM
        lstm_output, (hidden, cell) = self.lstm(x)

        # Use last hidden state
        last_hidden = hidden[-1]

        # Ensure output has batch dimension
        if last_hidden.dim() == 1:
            last_hidden = last_hidden.unsqueeze(0)

        return self.output_projection(last_hidden)

    def _forward_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for transformer encoder."""
        # Project input to embed_dim
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding

        # Transformer
        x = self.transformer(x)

        # Use mean pooling
        x = x.mean(dim=1)

        return self.output_projection(x)

    def _forward_autoencoder(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for autoencoder."""
        # Encode
        encoded = self.encoder(x)

        # Output projection
        return self.output_projection(encoded)

    def _forward_default(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for default encoder."""
        return self.encoder(x)


class LearnedStateRepresentationSystem:
    """
    System for learning sophisticated state representations that enable
    the HQ to develop better understanding of the game state.
    """

    def __init__(self, faction_id: str, input_state_size: int = 29):
        """
        Initialize the learned state representation system.

        Args:
            faction_id: Unique identifier for the faction
            input_state_size: Size of input state vector
        """
        self.faction_id = faction_id
        self.input_state_size = input_state_size

        # State encoders for different representation types
        self.state_encoders = {}
        self.state_representations = {}
        self.temporal_memory = deque(
            maxlen=utils_config.LEARNED_STATE_CONFIG["temporal_window"]
        )
        self.spatial_memory = deque(maxlen=100)

        # Learning parameters
        self.learning_rate = utils_config.LEARNED_STATE_CONFIG["learning_rate"]
        self.representation_learning_rate = utils_config.LEARNED_STATE_CONFIG[
            "representation_learning_rate"
        ]

        # Performance tracking
        self.encoding_quality = defaultdict(float)
        self.pattern_discovery_rate = defaultdict(float)
        self.concept_formation_rate = defaultdict(float)

        # Initialize state encoders
        self._initialize_state_encoders()

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[LEARNED STATE REPRESENTATION] Initialized for faction {faction_id}",
                level=logging.INFO,
            )

    def _initialize_state_encoders(self):
        """Initialize state encoders for different representation types."""
        config = utils_config.LEARNED_STATE_CONFIG

        for repr_type in utils_config.StateRepresentationType:
            component_size = utils_config.STATE_REPRESENTATION_COMPONENTS[
                repr_type.value
            ]

            # Choose encoder type based on representation type
            if repr_type == utils_config.StateRepresentationType.TEMPORAL_PATTERNS:
                encoder_type = "recurrent"
            elif repr_type == utils_config.StateRepresentationType.SPATIAL_RELATIONS:
                encoder_type = "convolutional"
            elif repr_type == utils_config.StateRepresentationType.ABSTRACT_CONCEPTS:
                encoder_type = "attention"
            elif repr_type == utils_config.StateRepresentationType.CAUSAL_MODELS:
                encoder_type = "transformer"
            elif (
                repr_type == utils_config.StateRepresentationType.HIERARCHICAL_FEATURES
            ):
                encoder_type = "autoencoder"
            else:
                encoder_type = "default"

            self.state_encoders[repr_type] = StateEncoder(
                input_size=self.input_state_size,
                output_size=component_size,
                encoder_type=encoder_type,
            )

            # Initialize performance tracking
            self.encoding_quality[repr_type] = 0.5
            self.pattern_discovery_rate[repr_type] = 0.0
            self.concept_formation_rate[repr_type] = 0.0

    def encode_state(self, state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Encode the current state using all representation types.

        Args:
            state: Current game state dictionary

        Returns:
            Dictionary containing encoded representations for each type
        """
        # Convert state to tensor
        state_vector = self._state_to_vector(state)
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)

        representations = {}

        # Encode using each representation type
        for repr_type, encoder in self.state_encoders.items():
            with torch.no_grad():
                representation = encoder(state_tensor)
                representations[repr_type] = representation.squeeze(0)

                # Update encoding quality
                self._update_encoding_quality(repr_type, representation)

        # Store representations
        self.state_representations = representations

        # Update temporal memory
        self.temporal_memory.append(
            {
                "state": state_vector,
                "representations": representations,
                "timestamp": time.time(),
            }
        )

        return representations

    def _state_to_vector(self, state: Dict[str, Any]) -> List[float]:
        """
        Convert state dictionary to vector representation.

        Args:
            state: State dictionary

        Returns:
            State vector
        """
        # Extract key state components
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

        # Add spatial information
        nearest_resource = state.get("nearest_resource", {})
        nearest_threat = state.get("nearest_threat", {})

        vector.extend(
            [
                nearest_resource.get("distance", float("inf")) / 200.0,
                nearest_threat.get("distance", float("inf")) / 150.0,
                (
                    float(nearest_resource.get("type") == "gold")
                    if nearest_resource
                    else 0.0
                ),
                float(nearest_threat.get("type") == "enemy") if nearest_threat else 0.0,
            ]
        )

        # Add temporal information
        vector.extend(
            [
                state.get("step", 0.0) / 1000.0,
                state.get("episode", 0.0) / 100.0,
                state.get("strategy_duration", 0.0) / 100.0,
                state.get("last_strategy_change", 0.0) / 50.0,
            ]
        )

        # Add coordination information
        vector.extend(
            [
                state.get("communication_success_rate", 0.5),
                state.get("coordination_success_rate", 0.5),
                state.get("experience_sharing_rate", 0.5),
                state.get("collective_learning_rate", 0.5),
            ]
        )

        # Add mission information
        vector.extend(
            [
                state.get("mission_progress", 0.0),
                state.get("mission_success_rate", 0.5),
                state.get("adaptive_behavior_rate", 0.5),
                state.get("strategy_effectiveness", 0.5),
            ]
        )

        # Add resource management information
        vector.extend(
            [
                state.get("resource_efficiency", 0.5),
                state.get("threat_response_time", 0.0) / 100.0,
                state.get("territory_control", 0.0),
                state.get("expansion_rate", 0.0),
            ]
        )

        # Add performance metrics
        vector.extend(
            [
                state.get("win_rate", 0.5),
                state.get("survival_rate", 0.5),
                state.get("efficiency_score", 0.5),
                state.get("coordination_score", 0.5),
            ]
        )

        # Ensure vector has correct size
        while len(vector) < self.input_state_size:
            vector.append(0.0)

        return vector[: self.input_state_size]

    def _update_encoding_quality(
        self,
        repr_type: utils_config.StateRepresentationType,
        representation: torch.Tensor,
    ):
        """Update encoding quality for a representation type."""
        # Calculate encoding quality based on representation properties
        quality = 0.0

        # Check for non-zero values (activity)
        non_zero_ratio = (representation != 0).float().mean().item()
        quality += non_zero_ratio * 0.3

        # Check for diversity (entropy)
        if representation.numel() > 1:
            probs = F.softmax(representation, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
            max_entropy = math.log(representation.numel())
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            quality += normalized_entropy * 0.4

        # Check for magnitude (strength)
        magnitude = torch.norm(representation).item()
        normalized_magnitude = min(magnitude / math.sqrt(representation.numel()), 1.0)
        quality += normalized_magnitude * 0.3

        # Update quality with exponential moving average
        current_quality = self.encoding_quality[repr_type]
        alpha = 0.1
        new_quality = (1 - alpha) * current_quality + alpha * quality
        self.encoding_quality[repr_type] = new_quality

    def discover_patterns(self) -> Dict[str, Any]:
        """
        Discover patterns in the state representations.

        Returns:
            Dictionary containing discovered patterns
        """
        patterns = {}

        if len(self.temporal_memory) < 5:
            return patterns

        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns()
        if temporal_patterns:
            patterns["temporal"] = temporal_patterns

        # Analyze spatial patterns
        spatial_patterns = self._analyze_spatial_patterns()
        if spatial_patterns:
            patterns["spatial"] = spatial_patterns

        # Analyze causal patterns
        causal_patterns = self._analyze_causal_patterns()
        if causal_patterns:
            patterns["causal"] = causal_patterns

        # Update pattern discovery rate
        if patterns:
            self._update_pattern_discovery_rate(len(patterns))

        return patterns

    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in the state representations."""
        if len(self.temporal_memory) < 3:
            return {}

        patterns = {}

        # Analyze trends
        recent_states = [entry["state"] for entry in list(self.temporal_memory)[-5:]]
        if len(recent_states) >= 3:
            # Calculate trends for key metrics
            trends = {}
            for i in range(min(5, len(recent_states[0]))):
                values = [state[i] for state in recent_states]
                if len(values) >= 3:
                    # Simple trend calculation
                    trend = (values[-1] - values[0]) / len(values)
                    trends[f"metric_{i}"] = trend

            if trends:
                patterns["trends"] = trends

        return patterns

    def _analyze_spatial_patterns(self) -> Dict[str, Any]:
        """Analyze spatial patterns in the state representations."""
        patterns = {}

        # Analyze resource-threat relationships
        if len(self.temporal_memory) >= 2:
            recent_entry = list(self.temporal_memory)[-1]
            state = recent_entry["state"]

            # Extract spatial information
            resource_distance = state[10] if len(state) > 10 else 0.0
            threat_distance = state[11] if len(state) > 11 else 0.0

            # Calculate spatial relationships
            if resource_distance > 0 and threat_distance > 0:
                spatial_ratio = resource_distance / threat_distance
                patterns["resource_threat_ratio"] = spatial_ratio

        return patterns

    def _analyze_causal_patterns(self) -> Dict[str, Any]:
        """Analyze causal patterns in the state representations."""
        patterns = {}

        if len(self.temporal_memory) < 3:
            return patterns

        # Analyze cause-effect relationships
        recent_entries = list(self.temporal_memory)[-3:]

        # Look for correlations between different state components
        correlations = {}
        for i in range(min(5, len(recent_entries[0]["state"]))):
            for j in range(i + 1, min(5, len(recent_entries[0]["state"]))):
                values_i = [entry["state"][i] for entry in recent_entries]
                values_j = [entry["state"][j] for entry in recent_entries]

                if len(values_i) >= 2 and len(values_j) >= 2:
                    # Simple correlation calculation
                    correlation = np.corrcoef(values_i, values_j)[0, 1]
                    if not np.isnan(correlation):
                        correlations[f"correlation_{i}_{j}"] = correlation

        if correlations:
            patterns["correlations"] = correlations

        return patterns

    def _update_pattern_discovery_rate(self, num_patterns: int):
        """Update pattern discovery rate."""
        for repr_type in utils_config.StateRepresentationType:
            current_rate = self.pattern_discovery_rate[repr_type]
            alpha = 0.1
            new_rate = (1 - alpha) * current_rate + alpha * (
                1.0 if num_patterns > 0 else 0.0
            )
            self.pattern_discovery_rate[repr_type] = new_rate

    def form_concepts(self) -> Dict[str, Any]:
        """
        Form abstract concepts from the state representations.

        Returns:
            Dictionary containing formed concepts
        """
        concepts = {}

        if not self.state_representations:
            return concepts

        # Form concepts based on representation types
        for repr_type, representation in self.state_representations.items():
            concept = self._form_concept_for_type(repr_type, representation)
            if concept:
                concepts[repr_type.value] = concept

        # Update concept formation rate
        if concepts:
            self._update_concept_formation_rate(len(concepts))

        return concepts

    def _form_concept_for_type(
        self,
        repr_type: utils_config.StateRepresentationType,
        representation: torch.Tensor,
    ) -> Dict[str, Any]:
        """Form a concept for a specific representation type."""
        concept = {}

        if repr_type == utils_config.StateRepresentationType.ABSTRACT_CONCEPTS:
            # Form abstract concepts based on representation values
            concept["dominant_features"] = torch.topk(
                representation, k=3
            ).indices.tolist()
            concept["feature_strength"] = torch.topk(
                representation, k=3
            ).values.tolist()

        elif repr_type == utils_config.StateRepresentationType.TEMPORAL_PATTERNS:
            # Form temporal concepts
            concept["temporal_stability"] = torch.std(representation).item()
            concept["temporal_trend"] = torch.mean(representation).item()

        elif repr_type == utils_config.StateRepresentationType.SPATIAL_RELATIONS:
            # Form spatial concepts
            concept["spatial_clustering"] = torch.mean(representation).item()
            concept["spatial_diversity"] = torch.std(representation).item()

        elif repr_type == utils_config.StateRepresentationType.CAUSAL_MODELS:
            # Form causal concepts
            concept["causal_strength"] = torch.norm(representation).item()
            concept["causal_complexity"] = torch.sum(torch.abs(representation)).item()

        elif repr_type == utils_config.StateRepresentationType.HIERARCHICAL_FEATURES:
            # Form hierarchical concepts
            concept["hierarchy_level"] = torch.max(representation).item()
            concept["hierarchy_depth"] = torch.min(representation).item()

        return concept

    def _update_concept_formation_rate(self, num_concepts: int):
        """Update concept formation rate."""
        for repr_type in utils_config.StateRepresentationType:
            current_rate = self.concept_formation_rate[repr_type]
            alpha = 0.1
            new_rate = (1 - alpha) * current_rate + alpha * (
                1.0 if num_concepts > 0 else 0.0
            )
            self.concept_formation_rate[repr_type] = new_rate

    def get_learned_state_reward(self) -> float:
        """
        Calculate reward for learned state representation quality.

        Returns:
            Learned state representation reward
        """
        # Base reward from encoding quality
        avg_encoding_quality = np.mean(list(self.encoding_quality.values()))
        base_reward = avg_encoding_quality * 0.3

        # Bonus for pattern discovery
        avg_pattern_discovery = np.mean(list(self.pattern_discovery_rate.values()))
        pattern_bonus = avg_pattern_discovery * 0.2

        # Bonus for concept formation
        avg_concept_formation = np.mean(list(self.concept_formation_rate.values()))
        concept_bonus = avg_concept_formation * 0.25

        # Bonus for high-quality representations
        if avg_encoding_quality > 0.8:
            base_reward += utils_config.LEARNED_STATE_CONFIG["representation_rewards"][
                "representation_quality"
            ]

        # Bonus for pattern discovery
        if avg_pattern_discovery > 0.5:
            base_reward += utils_config.LEARNED_STATE_CONFIG["representation_rewards"][
                "pattern_discovery"
            ]

        # Bonus for concept formation
        if avg_concept_formation > 0.5:
            base_reward += utils_config.LEARNED_STATE_CONFIG["representation_rewards"][
                "concept_formation"
            ]

        return base_reward + pattern_bonus + concept_bonus

    def get_state_representation_summary(self) -> Dict[str, Any]:
        """
        Get summary of learned state representation system performance.

        Returns:
            Dictionary containing performance metrics
        """
        return {
            "faction_id": self.faction_id,
            "encoding_quality": dict(self.encoding_quality),
            "pattern_discovery_rate": dict(self.pattern_discovery_rate),
            "concept_formation_rate": dict(self.concept_formation_rate),
            "avg_encoding_quality": np.mean(list(self.encoding_quality.values())),
            "avg_pattern_discovery": np.mean(
                list(self.pattern_discovery_rate.values())
            ),
            "avg_concept_formation": np.mean(
                list(self.concept_formation_rate.values())
            ),
            "temporal_memory_size": len(self.temporal_memory),
            "spatial_memory_size": len(self.spatial_memory),
        }

    def reset_episode(self):
        """Reset the learned state representation system for a new episode."""
        self.temporal_memory.clear()
        self.spatial_memory.clear()
        self.state_representations.clear()

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[LEARNED STATE REPRESENTATION] Reset episode for faction {self.faction_id}",
                level=logging.INFO,
            )

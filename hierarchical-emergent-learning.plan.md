<!-- cd33bb72-07cf-4c32-86dc-b89b2b41d969 677e33de-6b03-473d-af32-6a0c781b1cf6 -->
# Hierarchical Emergent Learning System

## Overview

Transform the current rule-based HQ and limited agent autonomy into a true emergent hierarchical learning system where:

- **HQ Level**: Learns to understand game state deeply and compose/adapt strategies dynamically
- **Agent Level**: Learns to follow high-level objectives while adapting tactics independently and coordinating with teammates
- **Military-style coordination**: "Mission-oriented" where HQ sets objectives, agents execute with autonomy

---

## Phase 1: HQ Strategic Intelligence Enhancement

### 1.1 Replace Hardcoded Strategies with Parametric Actions ✅ COMPLETED

**Current Problem**: HQ chooses from 11 fixed strategies with hardcoded if-else logic in `perform_HQ_Strategy()`.

**Solution**: Create parametric strategy system where HQ outputs continuous action parameters.

**Changes to `AGENT/agent_factions.py`**:

```python
# Current: HQ outputs discrete strategy index
# New: HQ outputs strategy type + parameters

class Faction:
    def perform_HQ_Strategy_parametric(self, strategy_type, parameters):
        """
        Execute strategy with learned parameters.
        
        strategy_type: categorical (recruit/swap/prioritize/defend)
        parameters: {
            'target_role': 'gatherer' or 'peacekeeper',
            'priority_resource': 'gold' or 'food',
            'aggression_level': 0.0-1.0 (defensive to offensive),
            'resource_threshold': 0.0-1.0 (when to execute),
            'agent_count_target': int (desired agents)
        }
        """
```

**Changes to `NEURAL_NETWORK/HQ_Network.py`**:

- Modify output head to produce both strategy type (categorical) and parameters (continuous)
- Add separate heads for different parameter types
- Use multi-task learning objective

### 1.2 Dynamic Strategy Composition

**Problem**: Can't combine or sequence strategies (e.g., "recruit gatherer THEN send to gold vein").

**Solution**: Add strategy sequencing and composition.

**New file: `AGENT/agent_strategy_composer.py`**:

```python
class StrategyComposer:
    """
    Allows HQ to learn multi-step strategy sequences.
    Uses hierarchical RL with options framework.
    """
    def compose_strategy(self, current_state, goal_state):
        # Learn to break down high-level goals into sequences
        # Example: "Win via gold" -> [recruit_gatherers, secure_resources, defend_HQ]
```

### 1.3 Improve HQ State Understanding ✅ COMPLETED

**Problem**: HQ state encoding is manually engineered with fixed normalization.

**Solution**: Add learned state representation.

**Changes to `NEURAL_NETWORK/HQ_Network.py`**:

- Add state encoder network that learns important features
- Use contrastive learning to improve state discrimination
- Add recurrent component (LSTM/GRU) to track temporal patterns
```python
class HQ_Network(nn.Module):
    def __init__(self, ...):
        # Add learned state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(raw_state_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Add temporal processing
        self.temporal_processor = nn.LSTM(256, 128, num_layers=2)
```


### 1.4 Enhanced HQ Reward Shaping ✅ COMPLETED

**Problem**: HQ rewards are episode-based and don't capture strategy effectiveness during execution.

**Solution**: Implement temporal difference rewards and strategy effectiveness metrics.

**Changes to `AGENT/agent_factions.py` - `compute_hq_reward()`**:

- Track strategy effectiveness: did it achieve intended outcome?
- Measure agent coordination: are agents working together?
- Resource efficiency: resources spent vs gained
- Adaptive goals: reward achieving what situation demands

---

## Phase 2: Agent Tactical Autonomy & Coordination

### 2.1 Mission-Oriented Task System ✅ COMPLETED

**Problem**: Tasks are too specific (exact position), limiting agent autonomy.

**Solution**: High-level mission objectives with agent autonomy in execution.

**New file: `AGENT/agent_missions.py`**:

```python
class Mission:
    """
    High-level objective from HQ with flexibility in execution.
    
    Instead of: "Go mine gold at position (45, 67)"
    Use: "Secure gold resources in western sector"
    
    Agents decide HOW to complete mission using their own learning.
    """
    def __init__(self, mission_type, objective_area, success_criteria, priority):
        self.mission_type = mission_type  # 'secure_resources', 'eliminate_threats', 'defend_area'
        self.objective_area = objective_area  # Region, not exact position
        self.success_criteria = success_criteria  # What constitutes success
        self.priority = priority  # Urgency (0-1)
        self.assigned_agents = []
```

**Changes to `AGENT/agent_factions.py` - `assign_high_level_tasks()`**:

- Replace specific tasks with mission assignments
- Agents interpret missions based on local context
- HQ monitors mission progress, not micro-actions

### 2.2 Multi-Agent Coordination Learning ✅ COMPLETED

**Problem**: Agents don't learn to coordinate; they act independently or follow orders.

**Solution**: Implement multi-agent reinforcement learning (MARL) with communication.

**New file: `NEURAL_NETWORK/MARL_Coordination.py`**:

```python
class CoordinationModule:
    """
    Enables agents to learn coordinated behaviors:
    - Form teams dynamically
    - Share sub-goals
    - Execute coordinated attacks/gathering
    - Cover each other
    """
    
    def __init__(self):
        # Attention mechanism over nearby agents
        self.agent_attention = MultiHeadAttention(...)
        
        # Communication protocol learning
        self.comm_encoder = nn.Linear(state_size, comm_size)
        self.comm_decoder = nn.Linear(comm_size, action_influence)
```

**Changes to `AGENT/agent_communication.py`**:

- Add learned communication: agents share learned message vectors
- Agents learn WHAT to communicate, not just broadcast predefined messages
- Communication influences action selection

### 2.3 Adaptive Action Selection ✅ COMPLETED

**Problem**: Agents follow tasks rigidly; if task becomes invalid, they fail rather than adapt.

**Solution**: Dynamic task adaptation and fallback behaviors.

**Changes to `AGENT/agent_behaviours.py` - `perform_task()`**:

```python
def perform_task_adaptive(self, state, resource_manager, agents):
    """
    Enhanced task execution with adaptation:
    1. Attempt assigned task
    2. If blocked/invalid: assess alternatives
    3. Take best available action toward mission goal
    4. Report status to HQ
    """
    
    # Check if task is still valid
    if not self.is_task_still_valid(self.agent.current_task):
        # Learn to find alternative approach to mission
        alternative_action = self.find_alternative_to_mission()
        if alternative_action:
            return self.execute_alternative(alternative_action)
    
    # Normal execution
    return self.execute_task()
```

### 2.4 Context-Aware Independent Action ✅ COMPLETED

**Problem**: When no task, agents act randomly; don't consider broader context or coordination.

**Solution**: Enhance independent action with mission awareness and teammate coordination.

**Changes to `AGENT/agent_behaviours.py` - line 142-188**:

```python
def act_independently_coordinated(self, state, resource_manager, agents):
    """
    When no explicit task:
    1. Assess current faction mission/strategy
    2. Check what teammates are doing
    3. Find complementary action
    4. Consider HQ priorities
    """
    
    # Get faction context
    faction_strategy = self.agent.faction.current_strategy
    faction_needs = self.assess_faction_needs()
    
    # Check teammate actions for coordination
    nearby_teammates = self.get_nearby_teammates(agents)
    teammate_actions = [t.current_action for t in nearby_teammates]
    
    # Select action that complements team
    valid_actions = self.get_valid_action_indices(resource_manager, agents)
    coordinated_action = self.select_coordinated_action(
        valid_actions, faction_strategy, teammate_actions, faction_needs
    )
    
    return self.perform_action(coordinated_action, state, resource_manager, agents)
```

---

## Phase 3: Hierarchical Learning Integration

### 3.1 Two-Level Reward System ✅ COMPLETED

**Problem**: HQ and agent rewards are disconnected.

**Solution**: Hierarchical reward propagation where HQ success depends on agent execution quality, and agents get feedback from mission outcomes.

**New file: `AGENT/agent_hierarchical_rewards.py`**:

```python
class HierarchicalRewardSystem:
    """
    Connects HQ strategy success to agent execution:
    
    HQ Reward = base_strategy_reward + agent_coordination_bonus + mission_success_rate
    Agent Reward = task_completion + mission_contribution + team_coordination
    """
    
    def compute_mission_contribution(self, agent, mission):
        # How much did this agent contribute to mission success?
        pass
    
    def compute_coordination_quality(self, agents_on_mission):
        # Did agents work together effectively?
        pass
```

**Integration in `AGENT/agent_factions.py` - `compute_hq_reward()`**:

- Add mission success rate
- Measure coordination quality across agents
- Reward efficient resource use and agent survival

**Integration in `AGENT/agent_behaviours.py` - `assign_reward()`**:

- Add mission progress component
- Reward coordination with teammates
- Bonus for adaptive behavior (handling unexpected situations)

### 3.2 Shared Experience Learning ✅ COMPLETED

**Problem**: Agents learn independently; don't benefit from teammates' experiences.

**Solution**: Implement experience sharing and team-level learning.

**Changes to `NEURAL_NETWORK/PPO_Agent_Network.py`**:

```python
class PPOModel(nn.Module):
    def train_with_team_experience(self, own_memory, team_memory, alpha=0.3):
        """
        Learn from both own experiences and filtered team experiences.
        Similar agents can learn from each other's successes/failures.
        """
        
        # Combine memories with importance weighting
        combined_memory = self.merge_experiences(own_memory, team_memory, alpha)
        
        # Train on combined experience
        return self.train(combined_memory)
```

### 3.3 Temporal Strategy Learning

**Problem**: HQ makes one-shot decisions; doesn't learn temporal strategy patterns.

**Solution**: Add sequence modeling to learn "when to do what" over time.

**Changes to `NEURAL_NETWORK/HQ_Network.py`**:

```python
class HQ_Network(nn.Module):
    def __init__(self, ...):
        # Add temporal module
        self.strategy_memory = nn.LSTM(hidden_size, hidden_size, num_layers=2)
        
    def forward(self, state, history):
        """
        Consider recent strategy history when making decisions.
        Learn patterns like: "If recruited gatherers, now assign them" 
        """
        # Encode current state
        state_features = self.feature_extractor(state)
        
        # Consider recent strategies
        temporal_context, _ = self.strategy_memory(history)
        
        # Combine for decision
        combined = torch.cat([state_features, temporal_context[-1]], dim=-1)
        ...
```

---

## Phase 4: Emergent Behavior Enablement

### 4.1 Meta-Learning for Strategy Discovery

**Problem**: Strategies are predefined; can't discover novel approaches.

**Solution**: Implement meta-learning to discover and refine new strategies.

**New file: `NEURAL_NETWORK/Meta_Strategy_Learning.py`**:

```python
class MetaStrategyLearner:
    """
    Learns to discover and evaluate new strategies.
    Uses evolutionary approach + gradient-based meta-learning.
    """
    
    def discover_strategies(self, past_episodes):
        # Identify patterns in successful episodes
        # Generate strategy variations
        # Test and evaluate new strategies
        pass
```

### 4.2 Opponent Modeling

**Problem**: HQ doesn't learn opponent patterns or adapt to counter-strategies.

**Solution**: Add opponent modeling and adaptive counter-strategy.

**New file: `AGENT/agent_opponent_modeling.py`**:

```python
class OpponentModel:
    """
    Learns opponent faction patterns:
    - Typical strategies
    - Weaknesses
    - Response patterns
    
    Enables adaptive counter-strategies.
    """
    
    def learn_opponent_pattern(self, opponent_actions, outcomes):
        pass
    
    def suggest_counter_strategy(self, current_opponent_state):
        pass
```

### 4.3 Emergent Role Specialization

**Problem**: Agents are hardcoded into gatherer/peacekeeper roles.

**Solution**: Let agents develop specialized sub-roles through learning.

**Changes to agent initialization**:

- Remove strict role constraints
- Let agents access all actions but learn preferences
- Reward specialization that benefits team
- Allow dynamic role adaptation based on situation

---

## Phase 5: Testing & Visualization

### 5.1 Strategy Interpretability

**New file: `UTILITIES/strategy_interpreter.py`**:

```python
class StrategyInterpreter:
    """
    Visualize and explain learned strategies:
    - What is HQ trying to accomplish?
    - Why did it make this decision?
    - How are agents coordinating?
    """
```

### 5.2 Performance Metrics

**Changes to `GAME/game_manager.py`**:

- Track coordination quality metrics
- Measure strategy effectiveness over time
- Log emergent behavior patterns
- Compare to baseline (current system)

### 5.3 Documentation Updates

**Update `Documentation/ARCHITECTURE.md`**:

- Explain new hierarchical learning system
- Document emergent behavior capabilities
- Provide examples of learned coordination
- Show strategy discovery process

---

## Implementation Priority

**High Priority** (Core functionality):

1. Phase 1.1: Parametric strategies (enables flexibility) ✅ COMPLETED
2. Phase 2.1: Mission-oriented tasks (enables autonomy) ✅ COMPLETED
3. Phase 2.3: Adaptive action selection (enables robustness) ✅ COMPLETED
4. Phase 3.1: Hierarchical rewards (connects systems) ✅ COMPLETED

**Medium Priority** (Coordination):

5. Phase 2.2: Multi-agent coordination ✅ COMPLETED
6. Phase 3.2: Shared experience learning ✅ COMPLETED
7. Phase 1.3: Better state understanding ✅ COMPLETED

**Lower Priority** (Advanced features):

8. Phase 1.2: Strategy composition
9. Phase 4.1-4.3: Meta-learning & opponent modeling
10. Phase 5: Testing & visualization

---

## Expected Outcomes

After implementation, the system will demonstrate:

1. **Strategic Intelligence**: HQ learns when and how to deploy different strategy approaches based on game state
2. **Tactical Autonomy**: Agents execute missions with flexibility, adapting to obstacles and opportunities
3. **Emergent Coordination**: Agents learn to work together without explicit coordination commands
4. **Adaptive Behavior**: Both HQ and agents adjust to opponent strategies and unexpected situations
5. **Specialization**: Agents develop preferences and expertise while maintaining flexibility

This transforms the system from "rule-based task scheduler with RL agents" to "true hierarchical emergent multi-agent learning system."

### To-dos

- [x] Add strategy parameter definitions to utils_config.py
- [x] Add parameter output heads to HQ_Network
- [x] Modify HQ_Network forward pass to output parameters
- [x] Implement perform_HQ_Strategy_parametric in Faction class
- [x] Migrate strategies from hardcoded to parametric one by one
- [x] Test parametric strategy system
- [x] Implement parametric HQ strategy system (replace discrete strategy selection with continuous parameters)
- [x] Create mission-oriented task framework (high-level objectives instead of specific tasks)
- [x] Implement adaptive action selection for agents (handle task failures gracefully)
- [x] Build two-level reward system connecting HQ strategy success to agent execution quality
- [x] Add multi-agent coordination learning with learned communication
- [x] Implement experience sharing between agents on same faction
- [x] Add learned state representation for better HQ understanding
- [ ] Enable HQ to compose and sequence strategies dynamically
- [ ] Implement meta-learning for strategy discovery
- [ ] Create strategy interpretability and performance visualization tools

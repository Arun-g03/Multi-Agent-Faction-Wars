"""
This file handles the communication between agents and the HQ. 


Purpose: To facilitate communication between agents and the HQ, allowing for the exchange of information and coordination of actions.

        Agents only have a local view of the game, While the HQ has a global view of the game through aggregated data from all agents.
"""

import logging
from utils_logger import Logger
from utils_config import TaskState, LOGGING_ENABLED
    



logger = Logger(log_file="CommunicationSystem_log.txt", log_level=logging.DEBUG)

class CommunicationSystem:
    def __init__(self, agents, faction):
        """
        Each faction gets its own CommunicationSystem instance.
        :param agents: List of agents in the faction.
        :param faction: The faction that owns this communication system.
        """
        self.agents = agents
        self.faction = faction  #  Only one faction

        self.report_log = {"threat": [], "resource": []}  #  Store latest reports

    def broadcast(self, sender, message):
        """
        Broadcast a message to agents **only within the same faction**.
        """
        if message["type"] == "resource_found" and message["data"] in sender.known_resources:
            return  # Skip broadcasting known resources
        if message["type"] == "enemy_spotted":
            threat_id = message["data"].get("id")
            if any(threat.get("id") == threat_id for threat in sender.known_threats):
                return  # Skip broadcasting known threats

        #  Ensure only faction members receive the message
        for agent in self.agents:
            if agent != sender:
                self.receive_message(agent, message)



    def receive_message(self, agent, message):
        """
        Process a received message.
        :param agent: The agent receiving the message.
        :param message: The message content.
        """
        message_type = message.get("type")
        data = message.get("data")

        if message_type == "enemy_spotted":
            # Check if the threat is already known by its ID
            threat_id = data.get("id")
            if threat_id and not any(threat.get("id") == threat_id for threat in agent.known_threats):
                agent.update_threat_location(data)
                if LOGGING_ENABLED: logger.debug_log(f"Agent {agent.role} added new threat with ID {threat_id}.", level=logging.INFO)

        elif message_type == "request_help":
            agent.update_help_request(data)
            if LOGGING_ENABLED: logger.debug_log(f"Agent {agent.role} received a help request.", level=logging.INFO)

        elif message_type == "status_update":
            agent.update_status(data)
            if LOGGING_ENABLED: logger.debug_log(f"Agent {agent.role} updated status with data: {data}.", level=logging.INFO)

        # Log received message
        if LOGGING_ENABLED: logger.debug_log(f"Agent {agent.role} received message: {message}", level=logging.INFO)
    
    def agent_to_agent(self, sender, receiver, message):
        """
        Handle direct communication between agents.
        :param sender: The agent sending the message
        :param receiver: The agent receiving the message 
        :param message: Dictionary containing message type and data
        """
        message_type = message.get("type")
        data = message.get("data")

        if message_type == "resource_info":
            receiver.known_resources.append(data)
            if LOGGING_ENABLED: logger.debug_log(f"Agent {sender.agent_id} informed {receiver.agent_id} about resource at {data['location']}", 
                           level=logging.INFO)
            
        elif message_type == "threat_info":
            receiver.known_threats.append(data)
            if LOGGING_ENABLED: logger.debug_log(f"Agent {sender.agent_id} informed {receiver.agent_id} about threat at {data['location']}", 
                           level=logging.INFO)
            
        elif message_type == "help_request":
            receiver.ally_requests.append({
                "sender": sender.agent_id,
                "location": data["location"],
                "type": data["request_type"]
            })
            if LOGGING_ENABLED: logger.debug_log(f"Agent {sender.agent_id} requested help from {receiver.agent_id} at {data['location']}", 
                           level=logging.INFO)

        # Log received message
        if LOGGING_ENABLED: logger.debug_log(f"Agent {sender.agent_id} sent message to {receiver.agent_id}: {message}", level=logging.INFO)

        
    def agent_to_hq(self, agent, report):
        faction = agent.faction  # Assuming agents have a reference to their faction
        faction.receive_report(report)
        #if LOGGING_ENABLED: logger.debug_log(f"Agent {agent.role} sent report to Faction {faction.id}: {report}", level=logging.INFO)

    def Communicate_Task_to_Agent(self):
        """
        Each faction's HQ processes reports and dynamically assigns tasks to agents.
        """
        for faction in self.faction:
            # Ensure global state is updated before assigning tasks
            faction.clean_global_state()

            if LOGGING_ENABLED: logger.debug_log(f"[HQ] Faction {faction.id} cleaned global state.", level=logging.INFO)

            # Use HQ's neural network to prioritise tasks dynamically
            faction.assign_high_level_tasks()  # This will internally handle the task assignment

            # Now, just iterate through agents to make sure the tasks are properly assigned
            for agent in faction.agents:
                # Check if the agent has a task already
                if agent.current_task is None:
                    # If the agent does not have a task, log this and assign a new task
                    if LOGGING_ENABLED: logger.debug_log(f"[HQ] No task assigned to agent {agent.agent_id} ({agent.role}). No valid resources or threats.", level=logging.WARNING)
                else:
                    if LOGGING_ENABLED: logger.debug_log(f"[HQ] Agent {agent.agent_id} already has a task: {agent.current_task}", level=logging.DEBUG)



        

        
        



    def clear_messages(self):
        """
        Clear any message queues or shared data between agents.
        """
        for agent in self.agents:
            agent.known_resources = []
            agent.known_threats = []
            agent.ally_requests = []

    def get_latest_reports(self, report_type):
        """
        Retrieve the latest reports of a given type ('threat' or 'resource').
        """
        return self.report_log.get(report_type, [])  # Return stored reports or empty list




"""
This file handles the communication between agents and the HQ. 


Purpose: To facilitate communication between agents and the HQ, allowing for the exchange of information and coordination of actions.

        Agents only have a local view of the game, While the HQ has a global view of the game through aggregated data from all agents.
"""

import logging
from utils_logger import Logger



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
                logger.debug_log(f"Agent {agent.role} added new threat with ID {threat_id}.", level=logging.INFO)

        elif message_type == "request_help":
            agent.update_help_request(data)
            logger.debug_log(f"Agent {agent.role} received a help request.", level=logging.INFO)

        elif message_type == "status_update":
            agent.update_status(data)
            logger.debug_log(f"Agent {agent.role} updated status with data: {data}.", level=logging.INFO)

        # Log received message
        logger.debug_log(f"Agent {agent.role} received message: {message}", level=logging.INFO)
    
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
            logger.debug_log(f"Agent {sender.agent_id} informed {receiver.agent_id} about resource at {data['location']}", 
                           level=logging.INFO)
            
        elif message_type == "threat_info":
            receiver.known_threats.append(data)
            logger.debug_log(f"Agent {sender.agent_id} informed {receiver.agent_id} about threat at {data['location']}", 
                           level=logging.INFO)
            
        elif message_type == "help_request":
            receiver.ally_requests.append({
                "sender": sender.agent_id,
                "location": data["location"],
                "type": data["request_type"]
            })
            logger.debug_log(f"Agent {sender.agent_id} requested help from {receiver.agent_id} at {data['location']}", 
                           level=logging.INFO)

        # Log received message
        logger.debug_log(f"Agent {sender.agent_id} sent message to {receiver.agent_id}: {message}", level=logging.INFO)

        
    def agent_to_hq(self, agent, report):
        faction = agent.faction  # Assuming agents have a reference to their faction
        faction.receive_report(report)
        logger.debug_log(f"Agent {agent.role} sent report to Faction {faction.id}: {report}", level=logging.INFO)

    def hq_to_agents(self):
        """
        Each faction's HQ processes reports and dynamically assigns tasks to agents.
        """
        for faction in self.faction:
            #  Ensure global state is updated before assigning tasks
            faction.clean_global_state()

            #  Use HQ's neural network to prioritize tasks dynamically
            faction.assign_high_level_tasks()

            #  Iterate through agents and assign the best available task only if needed
            for agent in faction.agents:
                if agent.current_task is not None:  
                    continue  #  Skip if agent already has an active task

                task = faction.assign_task(agent)  #  Assign a new task only if needed
                if task:
                    agent.current_task = task
                    logger.debug_log(f"HQ assigned task to {agent.role}: {task}", level=logging.INFO)
                else:
                    logger.debug_log(f"No task assigned to {agent.role} (ID: {agent.agent_id}).", level=logging.WARNING)




    def send_task_to_agent(self, agent, global_state):
        if not agent.current_task:  # Skip if the agent is already busy
            if agent.role == "gatherer" and global_state["resources"]:
                # Assign the first available resource
                resource = global_state["resources"][0]
                agent.update_resource_location(resource["location"])
                agent.current_task = self.create_task(
                    task_type="gather",
                    target=resource["location"],
                    target_id=f"Resource-{resource['location']}",
                    target_type="resource"
                )
                logger.debug_log(f"Sent Task to {agent.role}: {agent.current_task}", level=logging.INFO)

            elif agent.role == "peacekeeper" and global_state["threats"]:
                # Assign the first available threat based on its ID
                threat = global_state["threats"][0]
                agent.update_threat_location(threat["location"])
                agent.current_task = self.create_task(
                    task_type="eliminate",
                    target=threat["location"],
                    target_id=threat["id"],
                    target_type=threat["type"]
                )
                logger.debug_log(f"Sent Task to {agent.role}: {agent.current_task}", level=logging.INFO)

    
    



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




import os
import numpy
import yaml
import yamlloader
from core.service_client import ServiceClient
from core.cognitive_node import CognitiveNode
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from core.service_client import ServiceClient
from core_interfaces.srv import LoadConfig, CreateNode
from core.utils import class_from_classname
from std_msgs.msg import Float32

BLUE = "\033[34m"
RESET = "\033[0m"
RED = "\033[31m"
PURPLE = "\033[35m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"


class BartenderSim(Node):
    """
    BartenderSim simulator class.
    """
    def __init__(self):
        """
        Constructor of the BartenderSim simulator class.
        Initializes the simulator with parameters, publishers, and perception messages.
        """
        super().__init__("BartenderSim")
        self.rng = None
        self.ident = None
        self.base_messages = {}
        self.perceptions = {}
        self.sim_publishers = {}
        self.bar_clients = []
        self.bottles = []
        self.glass = []
        self.know_preference = {}
        self.picked_bottle = None
        self.generate_clients()
        # Training mode control: "deterministic" or "random"
        self.training_mode = "deterministic"
        self.get_logger().info(f"Training mode set to: {self.training_mode}")


        self.random_seed = self.declare_parameter('random_seed', value = 0).get_parameter_value().integer_value
        self.config_file = self.declare_parameter('config_file', descriptor=ParameterDescriptor(dynamic_typing=True)).get_parameter_value().string_value
        
        # Fixing the serving position to a specific distance and angle
        self.serving_pos = {"distance":0.8, "angle":0.0}
        self.original_glass_pos = {}  # Store original preparation table position
        self.last_glass_pos = {}     # Store last position where glass was placed

        self.normal_inner = numpy.poly1d(
        numpy.polyfit([0.0, 0.3925, 0.785, 1.1775, 1.57], [0.45, 0.47, 0.525, 0.65, 0.9], 3)
        )
        self.normal_outer = numpy.poly1d(
        numpy.polyfit([0.0, 0.3925, 0.785, 1.1775, 1.57], [1.15, 1.25, 1.325, 1.375, 1.375], 3)
        )

        self.collection_area = {"x_min":-0.6, "x_max":0.6, "y_min":0.9, "y_max":1.1, "object":"bottles"}
        self.weighing_area = {"x_min":-0.5, "x_max":0.5, "y_min":0.5, "y_max":0.7, "object":"glass"}

        self.gripper_max = 0.085

        self.iteration = 0
        self.change_reward_iterations = {}

        self.cbgroup_server = MutuallyExclusiveCallbackGroup()
        self.cbgroup_client = MutuallyExclusiveCallbackGroup()

        self.load_client = ServiceClient(LoadConfig, 'commander/load_experiment')

        self.drink_was_served = False
        self.reward_given_for_current_service = False  # Track if reward was already given for current service cycle

        # Client lifecycle tracking
        self.client_drinking_countdown = 0
        self.client_will_drink = False
        self.client_departure_countdown = 0
        self.client_will_leave = False
        self.no_client_countdown = 0
        self.waiting_for_new_client = False

        # Store agent's bottle choice from topic
        self.agent_bottle_choice = None
        
        # Subscription to the agent's bottle decision topic
        self.agent_bottle_subscription = self.create_subscription(
            Float32,
            'cognitive_node/world_model/last_bottle',
            self.agent_bottle_callback,
            10
        )

    def agent_bottle_callback(self, msg):
        """
        Callback for receiving agent's bottle choice from the topic.
        
        :param msg: Float32 message containing the agent's bottle choice (normalized value)
        :type msg: Float32
        """
        self.agent_bottle_choice = msg.data
        self.get_logger().debug(f"Received agent bottle choice from topic: {self.agent_bottle_choice}")

    def random_position(self, area):
        """
        Generate a random position within the specified area.

        :param area: The area where the position should be generated.
        :type area: dict
        :return: A tuple containing the distance and angle of the generated position.
        :rtype: tuple
        """
        valid = False
        while not valid:
            x = self.rng.uniform(low=area["x_min"], high=area["x_max"])
            y = self.rng.uniform(low=area["y_min"], high=area["y_max"])

            dist = numpy.linalg.norm([x,y])
            ang = numpy.arctan(x/y)

            valid = True
            
            for object in self.perceptions[area["object"]].data:
                if abs(object.distance - dist) < 0.1 and abs(object.angle - ang) < 0.09:
                    valid = False
                    break
            
        return dist, ang

    def generate_clients(self):
        """
        Generate clients with random bottles preferences.
        """
        self.get_logger().info("Generating clients...")
        ids = numpy.random.choice(range(1, 4), size=3, replace=False)
        ident = 1
        for id in ids:
            client = dict(
                id = ident,
                beverage = id,
            )
            self.bar_clients.append(client)
            ident += 1
        self.get_logger().info(f"Generated clients: {self.bar_clients}")

    def generate_bottles(self,n_bottles=3):
        """
        Generate bottles in random positions, within the bar area, only one beverage per kind.
        :param n_bottles: Number of bottles to generate.
        :type n_bottles: int
        """
        self.bottles = []
        self.get_logger().info("Generating bottles...")
        for i in range(1, n_bottles + 1):
            distance, angle = self.random_position(self.collection_area)
            bottle = dict(distance=distance, angle=angle, id=i)
            self.bottles.append(bottle)

    def generate_glass(self):
        """
        generates one glass in random position within the bar area
        """
        self.get_logger().info("Generating glass...")
        distance, angle = self.random_position(self.collection_area)
        glass = dict(distance=distance, angle=angle, state=False)
        self.glass.append(glass)

    def perceive_bottles(self):
        """
        Perceive all bottles and update the bottle perceptions accordingly.
        """
        self.get_logger().info("Perceiving all bottles...")

        self.perceptions["bottles"].data = []

        if self.bottles:
            # Add all bottles to perception
            for bottle in self.bottles:
                # Create a new bottle perception message
                bottle_msg = self.base_messages["bottles"]()
                
                # Update the perception data with current bottle
                bottle_msg.distance = bottle["distance"]
                bottle_msg.angle = bottle["angle"]
                if hasattr(bottle_msg, 'id'):
                    bottle_msg.id = bottle["id"]
                
                # Add bottle to perception list
                self.perceptions["bottles"].data.append(bottle_msg)
                
            self.get_logger().info(f"Perceived {len(self.bottles)} bottles")
        else:
            # If no bottles, create default perception data
            self.perceptions["bottles"].data = [self.base_messages["bottles"]()]
            self.get_logger().info("No bottles available, created default perception")

    def perceive_last_bottle(self):
        """
        Choose the bottle, update the bottle perceptions accordingly.
        """
        self.get_logger().info("Perceiving last bottle...")
        
        # Add client WorldModel processing logic here

        # Use existing last_bottle perception or generate random if not available 
        if ("last_bottle" in self.perceptions and self.perceptions["last_bottle"].data is not None) and (self.perceptions["last_bottle"].data != False):
            bottle_id = self.perceptions["last_bottle"].data
        else:
            if self.training_mode == "deterministic":
                bottle_id = 1  # siempre selecciona la botella 1
            else:
                bottle_id = int(self.rng.integers(0, 3))

            if "last_bottle" in self.perceptions:
                self.perceptions["last_bottle"].data = bottle_id
        
        self.get_logger().info(f"Using bottle ID: {bottle_id}")

    def perceive_glass(self):
        """
        Perceive the glass and update the perception accordingly.
        """
        self.get_logger().info("Perceiving glass...")
        
        if self.glass:
            glass = self.glass[0]
            self.perceptions["glass"].data[0].distance = glass["distance"]
            self.perceptions["glass"].data[0].angle = glass["angle"]


    def start_client_drinking(self):
        """
        Start the client drinking simulation after correct drink is served.
        """
        if self.training_mode == "deterministic":
            drinking_delay = 3
        else:
            drinking_delay = self.rng.integers(3, 6)
        self.client_drinking_countdown = drinking_delay
        self.client_will_drink = True
        self.get_logger().info(f"{PURPLE}Client will start drinking in {drinking_delay} iterations{RESET}")

    def simulate_client_drinking(self):
        """
        Simulate client drinking behavior - called every iteration in update_reward_sensor.
        """
        if self.client_will_drink and self.client_drinking_countdown > 0:
            self.client_drinking_countdown -= 1
            self.get_logger().info(f"{PURPLE}Client drinking in {self.client_drinking_countdown} iterations...{RESET}")
            
            if self.client_drinking_countdown == 0:
                # Client finished drinking
                if self.glass_is_in_serving_position() and self.perceptions["glass"].data[0].state:
                    # Empty the glass - client drank it
                    self.perceptions["glass"].data[0].state = False
                    self.get_logger().info(f"{GREEN}CLIENT FINISHED DRINKING! Glass is now empty.{RESET}")
                    
                    # Update internal glass state
                    if self.glass:
                        self.glass[0]["state"] = False
                    
                    # Reset drinking flags and start departure countdown
                    self.client_will_drink = False
                    self.start_client_departure()

    def start_client_departure(self):
        """
        Start client departure simulation after drinking.
        """
        if self.training_mode == "deterministic":
            departure_delay = 2
        else:
            departure_delay = self.rng.integers(2, 5)
        self.client_departure_countdown = departure_delay
        self.client_will_leave = True
        self.get_logger().info(f"{YELLOW}Client will leave in {departure_delay} iterations{RESET}")

    def simulate_client_departure(self):
        """
        Simulate client leaving and new client arrival cycle.
        """
        # Handle client departure
        if self.client_will_leave and self.client_departure_countdown > 0:
            self.client_departure_countdown -= 1
            self.get_logger().info(f"{YELLOW}Client leaving in {self.client_departure_countdown} iterations...{RESET}")
            
            if self.client_departure_countdown == 0:
                # Client leaves - set ID to 0 (no client)
                self.perceptions["client"].data[0].id = 0
                self.perceptions["client"].data[0].preference = 0
                self.get_logger().info(f"{RED}CLIENT LEFT! No client present (ID=0){RESET}")
                
                # Reset client departure flags and start waiting period
                self.client_will_leave = False
                self.start_waiting_for_new_client()
        
        # Handle waiting period and new client arrival
        if self.waiting_for_new_client and self.no_client_countdown > 0:
            self.no_client_countdown -= 1
            self.get_logger().info(f"{BLUE}No client present, new client arriving in {self.no_client_countdown} iterations...{RESET}")
            
            if self.no_client_countdown == 0:
                # New client arrives
                self.generate_new_client()
                self.waiting_for_new_client = False

    def start_waiting_for_new_client(self):
        """
        Start waiting period before new client arrives.
        """
        if self.training_mode == "deterministic":
            waiting_delay = 4
        else:
            waiting_delay = self.rng.integers(3, 8)
        self.no_client_countdown = waiting_delay
        self.waiting_for_new_client = True
        self.get_logger().info(f"{BLUE}Waiting {waiting_delay} iterations before new client arrives{RESET}")

    def generate_new_client(self):
        """
        Generate a new client with random ID and beverage preference.
        """
        # Generate new client with random ID from available clients
        if self.bar_clients:
            # Pick a random client from the generated clients
            if self.training_mode == "deterministic":
                new_client = self.bar_clients[0]
            else:
                new_client = self.rng.choice(self.bar_clients)

            new_client_id = new_client["id"]
            
            # Set the new client
            self.perceptions["client"].data[0].id = new_client_id
            self.perceptions["client"].data[0].preference = 0  # Client hasn't been asked yet
            
            self.get_logger().info(f"{GREEN}NEW CLIENT ARRIVED! Client ID: {new_client_id}, Beverage preference: {new_client['beverage']}{RESET}")
            
            # Reset service tracking for new client
            self.drink_was_served = False
            self.reward_given_for_current_service = False
            
            # Clear asking iterations for fresh start
            if hasattr(self, 'asking_iterations'):
                self.asking_iterations = {}
            
            self.get_logger().info(f"{CYAN}Ready to serve new client!{RESET}")

    def reward_serve_the_drink_goal(self):
        """
        Gives a reward of 1.0 if the glass with the CORRECT drink is placed in the serving table.
        Only gives reward once per service cycle until glass returns to preparation area.
        """
        reward = 0.0
        if self.iteration < self.change_reward_iterations['stage0']:
            self.get_logger().info("Checking serve_glass_goal...")
            
            client_id = self.perceptions["client"].data[0].id
            
            # Skip if no valid client
            if client_id == 0:
                self.get_logger().info(f"{BLUE}No client present (ID=0), no service reward{RESET}")
                self.perceptions["serve_the_drink_goal"].data = reward
                return
            
            # Check if glass is in serving position and has drink
            if self.glass_is_in_serving_position() and self.perceptions["glass"].data[0].state:
                self.get_logger().info(f"{BLUE}GLASS IN SERVING POSITION WITH DRINK{RESET}")
                
                # Check if reward was already given for this service cycle
                if self.reward_given_for_current_service:
                    self.get_logger().info(f"{YELLOW}Reward already given for current service cycle{RESET}")
                    reward = 1.0
                else:
                    # Check if the drink is correct for the client
                    client_preference = self.know_preference.get(client_id, None)
                    last_bottle_used = self.perceptions["last_bottle"].data

                    if client_preference is not None and self.picked_bottle == client_preference:
                        self.get_logger().info(f"{GREEN}CORRECT DRINK SERVED! Client {client_id} wanted {client_preference}, got {self.picked_bottle}{RESET}")
                        reward = 1.0
                        self.reward_given_for_current_service = True  # Mark reward as given
                        
                        # IMPORTANT: Mark that drink was served - enables return reward later
                        self.drink_was_served = True
                        self.get_logger().info(f"{PURPLE}Drink service completed - return reward now available{RESET}")
                        
                        # Start client drinking simulation
                        self.start_client_drinking()
                        
                    else:
                        self.get_logger().info(f"{RED}WRONG DRINK SERVED! Client {client_id} wanted {client_preference}, got {self.picked_bottle}{RESET}")
                        reward = 1.0
                        self.perceptions["glass"].data[0].state = False  # Glass is emptied (client refuses drink)
            else:
                reward = 0.0
                self.get_logger().info(f"{BLUE}GLASS NOT IN SERVING POSITION OR NO DRINK{RESET}")

            self.perceptions["serve_the_drink_goal"].data = reward


    def glass_is_in_the_original_position(self):
        """
        Check if the glass is in the original position (preparation table).
        """
        glass = self.perceptions['glass'].data[0]
        
        # Check if we have saved the original position
        if not self.original_glass_pos:
            self.get_logger().warn(f"{YELLOW}No original position saved yet{RESET}")
            return False
        
        # Compare with original position with some tolerance
        distance_match = abs(glass.distance - self.original_glass_pos['distance']) < 0.1
        angle_match = abs(glass.angle - self.original_glass_pos['angle']) < 0.1
        
        self.get_logger().info(f"{CYAN}Comparing positions:{RESET}")
        self.get_logger().info(f"{CYAN}  Current: distance={glass.distance:.3f}, angle={glass.angle:.3f}{RESET}")
        self.get_logger().info(f"{CYAN}  Original: distance={self.original_glass_pos['distance']:.3f}, angle={self.original_glass_pos['angle']:.3f}{RESET}")
        self.get_logger().info(f"{CYAN}  Match: distance={distance_match}, angle={angle_match}{RESET}")
        
        return distance_match and angle_match

    def update_reward_sensor(self):
        """
        Update goal sensors' values.
        Also runs client drinking and departure simulation.
        """
        # First run client lifecycle simulations
        self.simulate_client_drinking()
        self.simulate_client_departure()
        
        # Then update all reward sensors
        for sensor in self.perceptions:
            reward_method = getattr(self, "reward_" + sensor, None)
            if callable(reward_method):
                reward_method()

    def random_perceptions(self):
        """
        Generate random perceptions when the world is reset.
        """
        self.perceptions["client"].data = []
        self.perceptions["client"].data.append(self.base_messages["client"]())
        self.perceptions["client"].data[0].id = int(self.rng.integers(1, 4))  # Random client ID
        self.perceptions["client"].data[0].preference = 0
        self.perceptions["bottles"].data = []
        self.perceptions["bottles"].data.append(self.base_messages["bottles"]())

        # Generate glass
        self.perceptions["glass"].data = []
        self.perceptions["glass"].data.append(self.base_messages["glass"]())
        distance, angle = self.random_position(self.weighing_area)
        self.perceptions["glass"].data[0].distance = distance
        self.perceptions["glass"].data[0].angle = angle
        self.perceptions["glass"].data[0].state = False

        self.perceptions["robot_position"].data = 0.0 # Random initial position of the robot, 0 = preparation table, 1 = serving table

        # Generate bottles and glass
        self.generate_bottles()
        self.generate_glass()

        # Perceive the environment
        self.perceive_bottles()
        self.perceive_last_bottle()
        self.perceive_glass()


        self.perceptions["glass_in_left_hand"].data = False
        self.perceptions["bottle_in_right_hand"].data = False

        # Reset all client lifecycle tracking
        self.drink_was_served = False
        self.reward_given_for_current_service = False
        self.client_drinking_countdown = 0
        self.client_will_drink = False
        self.client_departure_countdown = 0
        self.client_will_leave = False
        self.no_client_countdown = 0
        self.waiting_for_new_client = False
        
        # Reset glass and position tracking
        self.original_glass_pos = {}
        self.last_glass_pos = {}
        
        # Reset asking iterations
        if hasattr(self, 'asking_iterations'):
            self.asking_iterations = {}

        self.update_reward_sensor()

    def pick_glass_policy(self):
        """
        Pick the glass.
        """
        if not self.perceptions["glass_in_left_hand"].data and self.glass:
            self.perceptions["glass_in_left_hand"].data = True
            
            # Save the ORIGINAL position (preparation table) only the first time
            if not self.original_glass_pos:
                self.original_glass_pos['distance'] = self.perceptions["glass"].data[0].distance
                self.original_glass_pos['angle'] = self.perceptions["glass"].data[0].angle
                self.get_logger().info(f"{CYAN}Saved original glass position: distance={self.original_glass_pos['distance']}, angle={self.original_glass_pos['angle']}{RESET}")
            
            # Also update last_glass_pos for reference
            self.last_glass_pos['distance'] = self.perceptions["glass"].data[0].distance
            self.last_glass_pos['angle'] = self.perceptions["glass"].data[0].angle
            
            # Move glass to a special "in_hand" position that doesn't match serving or preparation positions
            # This ensures glass_is_in_serving_position() returns False when glass is being held
            IN_HAND_DISTANCE = -1.0  # Special value indicating "in hand"
            IN_HAND_ANGLE = -1.0     # Special value indicating "in hand"
            
            self.get_logger().info(f"{CYAN}Moving glass to 'in hand' position to avoid false serving position detection{RESET}")
            
            # Update perception position to special "in hand" coordinates
            self.perceptions["glass"].data[0].distance = IN_HAND_DISTANCE
            self.perceptions["glass"].data[0].angle = IN_HAND_ANGLE
            
            # Update internal glass state
            if self.glass:
                self.glass[0]["distance"] = IN_HAND_DISTANCE
                self.glass[0]["angle"] = IN_HAND_ANGLE
                
            self.get_logger().info(f"{GREEN}Glass picked up successfully! Moved to in-hand position.{RESET}")

    def pick_bottle_policy(self):
        """
        Pick the bottle that the agent has chosen.
        The agent's choice is already reflected in last_bottle and the filtered bottles perception.
        """
        # Don't pick a bottle if already holding one
        if self.perceptions["bottle_in_right_hand"].data:
            self.get_logger().info("Bottle already in hand. Exiting policy.")
            return

        client_id = self.perceptions["client"].data[0].id
        
        # Skip if no valid client
        if client_id == 0:
            self.get_logger().info("No client present (ID=0), cannot pick bottle")
            return

        # Get agent's choice from topic (comes directly from agent decision)
        if self.agent_bottle_choice is None:
            self.get_logger().info("No agent bottle choice available yet, cannot pick bottle")
            return
            
        agent_choice_normalized = self.agent_bottle_choice
        
        # Convert normalized choice to bottle preference ID (0-3 where 0=unknown, 1-3=bottle IDs)
        # Normalized range: 0.0-0.98 maps to preference IDs 0,1,2,3
        # 4 preference states: 0 (unknown), 1 (bottle 1), 2 (bottle 2), 3 (bottle 3)
        n_preferences = 4  # 0, 1, 2, 3
        
        # Map 0.0-0.98 to preference indices 0-3
        # Since max normalized is 0.98, we need to scale properly
        max_normalized = 0.98
        scaled_choice = agent_choice_normalized / max_normalized  # Scale to 0.0-1.0 range
        
        preference_index = int(scaled_choice * n_preferences)
        preference_index = min(preference_index, n_preferences - 1)  # Clamp to 0-3
        
        # The preference_index IS the bottle preference ID (0=unknown, 1-3=bottle IDs)
        agent_bottle_preference = preference_index

        self.get_logger().info(f"Normalized choice {agent_choice_normalized} -> scaled {scaled_choice:.3f} -> preference ID {agent_bottle_preference} (0=unknown, 1-3=bottles)")
        
        # Check if agent doesn't know preference yet (ID = 0)
        if agent_bottle_preference == 0:
            self.get_logger().info("Agent chose preference ID 0 (doesn't know preference yet), cannot pick bottle")
            return
        
        # Convert preference ID to actual bottle ID (preference 1-3 maps to bottle IDs 1-3)
        agent_bottle_id = agent_bottle_preference
        
        # Verify the bottle exists in the simulator
        if not self.bottles:
            self.get_logger().error("No bottles available to pick from!")
            return
            
        # Find the bottle that matches agent's choice in internal bottles
        bottle_found = False
        for bottle in self.bottles:
            self.get_logger().info(f"Checking bottle ID: {bottle['id']} against agent choice ID: {agent_bottle_id}")
            if bottle["id"] == agent_bottle_id:
                # Execute the pick action
                self.perceptions["bottle_in_right_hand"].data = True
                self.picked_bottle = bottle["id"]
                
                self.get_logger().info(f"Agent chose bottle {agent_bottle_id}, picked successfully!")
                bottle_found = True
                break
        
        if not bottle_found:
            self.get_logger().error(f"Agent's bottle choice {agent_bottle_id} not found!")
            self.perceptions["bottle_in_right_hand"].data = False
        else:
            self.get_logger().info(f"{GREEN}Bottle picked successfully!{RESET}")

        # NOTE: We DO NOT modify last_bottle here - it should remain as the agent's original decision
        # The reward system will compare client_preference vs the agent's choice in last_bottle

    def prepare_drink_policy(self):
        """
        Prepare the beverage for the client.
        """
        # check if is bottle and glass in the hands
        self.get_logger().info(f"{GREEN}Trying to prepare drink... glass in the left hand: {self.perceptions['glass_in_left_hand'].data}, bottle in right hand: {self.perceptions['bottle_in_right_hand'].data}{RESET}")
        if self.perceptions["glass_in_left_hand"].data and self.perceptions["bottle_in_right_hand"].data:
            self.get_logger().info(f"{GREEN}Preparing drink... Glass now contains beverage!{RESET}")
            
            # Glass now has drink
            self.perceptions["glass"].data[0].state = True
            
            # Update internal glass state
            if self.glass:
                self.glass[0]["state"] = True

    # --- Robot position helpers ------------------------------------------------
    def is_at_preparation_table(self) -> bool:
        """
        Returns True if the robot is at the preparation table.
        Convention in this sim: robot_position ~ 0.0 -> preparation; ~0.95 -> serving.
        We use a simple threshold at 0.5 for robustness.
        """
        try:
            pos = float(self.perceptions["robot_position"].data)
        except Exception:
            pos = 0.0
        at_prep = pos < 0.5
        self.get_logger().info(f"Robot at preparation table: {at_prep} (pos={pos})")
        return at_prep

    def is_at_serving_table(self) -> bool:
        """
        Returns True if the robot is at the serving table.
        Uses the same thresholding as preparation helper.
        """
        try:
            pos = float(self.perceptions["robot_position"].data)
        except Exception:
            pos = 0.0
        at_serv = pos >= 0.5
        self.get_logger().info(f"Robot at serving table: {at_serv} (pos={pos})")
        return at_serv


    def place_glass_serving_policy(self):
        """
        Place the glass on the serving table, only if the robot is at the serving table
        AND the glass is currently in the left hand.
        Does not affect bottle state.
        """
        if not self.is_at_serving_table():
            self.get_logger().info(f"{YELLOW}Not at serving table - cannot place glass on serving table{RESET}")
            return
        if not self.perceptions["glass_in_left_hand"].data:
            self.get_logger().info(f"{YELLOW}Glass not in hand - nothing to place on serving table{RESET}")
            return

        # Release glass from hand
        self.perceptions["glass_in_left_hand"].data = False

        # Place on serving position
        self.get_logger().info(f"{GREEN}Placing glass on serving table...{RESET}")
        self.get_logger().info(f"{GREEN}Serving position: distance={self.serving_pos['distance']}, angle={self.serving_pos['angle']}{RESET}")

        self.perceptions["glass"].data[0].distance = self.serving_pos['distance']
        self.perceptions["glass"].data[0].angle = self.serving_pos['angle']

        # Update internal glass state
        if self.glass:
            self.glass[0]["distance"] = self.serving_pos['distance']
            self.glass[0]["angle"] = self.serving_pos['angle']

        # Update last position to serving position
        self.last_glass_pos['distance'] = self.serving_pos['distance']
        self.last_glass_pos['angle'] = self.serving_pos['angle']

    def place_glass_return_policy(self):
        """
        Return the glass to its ORIGINAL preparation position, only if the robot
        is at the preparation table AND the glass is currently in the left hand.
        Resets reward flags to enable a new service cycle.
        """
        if not self.is_at_preparation_table():
            self.get_logger().info(f"{YELLOW}Not at preparation table - cannot return glass to original position{RESET}")
            return
        if not self.perceptions["glass_in_left_hand"].data:
            self.get_logger().info(f"{YELLOW}Glass not in hand - nothing to return to preparation table{RESET}")
            return
        if not self.original_glass_pos:
            self.get_logger().warn(f"{YELLOW}Original glass position unknown - cannot return{RESET}")
            return

        # Release glass from hand
        self.perceptions["glass_in_left_hand"].data = False

        # Place back on original preparation position
        self.get_logger().info(f"{YELLOW}Placing glass back on preparation table...{RESET}")
        self.get_logger().info(f"{YELLOW}Original preparation position: distance={self.original_glass_pos['distance']}, angle={self.original_glass_pos['angle']}{RESET}")
        self.picked_bottle = None  # Reset picked bottle after placing glass
        self.perceptions["glass"].data[0].distance = self.original_glass_pos['distance']
        self.perceptions["glass"].data[0].angle = self.original_glass_pos['angle']

        # Update internal glass state
        if self.glass:
            self.glass[0]["distance"] = self.original_glass_pos['distance']
            self.glass[0]["angle"] = self.original_glass_pos['angle']

        # Update last position to original preparation position
        self.last_glass_pos['distance'] = self.original_glass_pos['distance']
        self.last_glass_pos['angle'] = self.original_glass_pos['angle']

        # Reset reward flag when glass returns to preparation area
        self.reward_given_for_current_service = False
        self.get_logger().info(f"{CYAN}Service cycle completed - ready for new reward{RESET}")

    def place_bottle_policy(self):
        """
        Place the bottle back (i.e., release it). Current simulator behavior only
        toggles the bottle-in-hand flag; bottle positions are managed by perceive_bottles.
        Does not affect glass state.
        """
        if not self.is_at_preparation_table():
            self.get_logger().info(f"{YELLOW}Not at preparation table - cannot place bottle{RESET}")
            return
        if self.perceptions["bottle_in_right_hand"].data:
            self.perceptions["bottle_in_right_hand"].data = False
            self.get_logger().info(f"{GREEN}Bottle released at preparation table{RESET}")

    def move_to_policy(self):
        """
        Move the robot to the desired position.
        """
        if self.perceptions["robot_position"].data:
            self.perceptions["robot_position"].data = 0.0
        else:
            self.perceptions["robot_position"].data = 0.95



    def ask_nicely_policy(self):
        """
        Ask the client the preferred beverage.
        Always sets the preference when executed, maintains it for 2 iterations, then resets to 0.
        """
        client_id = self.perceptions["client"].data[0].id
        
        # Skip if no valid client (ID 0 means no client)
        if client_id == 0:
            self.get_logger().info("No client present (ID 0), nothing to ask")
            return
        
        # Initialize the iteration count for this client if it doesn't exist
        if not hasattr(self, 'asking_iterations'):
            self.asking_iterations = {}
        
        # Always find and set the client preference when this policy is executed
        client_data = None
        for client in self.bar_clients:
            self.get_logger().info(f'Checking client: {client}')
            if client["id"] == client_id:
                client_data = client
                break
        
        if client_data is None:
            self.get_logger().error(f"Client with ID {client_id} not found in bar_clients")
            return
        
        # Check if this is the first time asking this client
        if client_id not in self.asking_iterations:
            # First time - set preference and start counting
            self.know_preference[client_id] = client_data["beverage"]
            self.perceptions["client"].data[0].preference = int(self.know_preference[client_id])
            self.asking_iterations[client_id] = self.iteration
            self.get_logger().info(f"Client {client_id} preference set to {self.know_preference[client_id]} (first time asking)")
        else:
            # Check how many iterations have passed since first asking
            iterations_passed = self.iteration - self.asking_iterations[client_id]
            
            if iterations_passed < 2:
                # Still within the 2 iteration window - maintain preference
                self.know_preference[client_id] = client_data["beverage"]
                self.perceptions["client"].data[0].preference = int(self.know_preference[client_id])
                self.get_logger().info(f"Client {client_id} preference maintained at {self.know_preference[client_id]} (iteration {iterations_passed + 1}/2)")
            else:
                # 2 iterations have passed - reset preference to 0
                self.know_preference[client_id] = 0
                self.perceptions["client"].data[0].preference = 0
                self.get_logger().info(f"Client {client_id} preference reset to 0 after {iterations_passed} iterations (client forgot)")
                
                # Clear the iteration log for this client so it can be asked again
                del self.asking_iterations[client_id]

    def glass_is_in_serving_position(self):
        """
        Check if the glass is in the serving position.
        """
        glass = self.perceptions['glass'].data[0]
        
        # If glass is "in hand" (negative coordinates), it's not in serving position
        if glass.distance < 0 or glass.angle < 0:
            self.get_logger().info(f"{BLUE}Glass is in hand, not in serving position{RESET}")
            return False
        
        # Fixing the serving position to a specific distance and angle
        serving_pos = self.serving_pos
        self.get_logger().info(f"{BLUE}Checking if glass is in serving position...{RESET}")
        glass.distance = round(glass.distance, 1)
        glass.angle = round(glass.angle, 1)
        serving_pos['distance'] = round(serving_pos['distance'], 2)
        serving_pos['angle'] = round(serving_pos['angle'], 2)
        self.get_logger().info(f"Rounded glass position: distance={glass.distance}, angle={glass.angle}")
        self.get_logger().info(f"Rounded expected serving position: distance={serving_pos['distance']}, angle={serving_pos['angle']}")
        return (glass.distance == serving_pos['distance']) and (abs(glass.angle) == abs(serving_pos['angle']))


    def reward_left_the_glass_goal(self):
        """
        Gives a reward of 1.0 if the glass is correctly placed back in the original position
        AFTER the client has drunk the beverage and LEFT. Prevents cheating by requiring 
        that the full service cycle was completed.
        """
        reward = 0.0
        self.get_logger().info(f"{BLUE}Checking left_the_glass_goal...{RESET}")
        
        if self.iteration > self.change_reward_iterations['stage0']:
            self.get_logger().info("STAGE 2 REWARD: LEFT GLASS")
            
            # Check if glass is in original position
            if self.glass_is_in_the_original_position():
                self.get_logger().info(f"{CYAN}Glass is in original position{RESET}")
                
                # Key conditions: 
                # 1. Glass must be empty (client drank it)
                # 2. We must have served a drink before
                # 3. Client must have left (ID = 0) OR we're waiting for new client
                glass_is_empty = not self.perceptions["glass"].data[0].state
                service_was_completed = self.drink_was_served
                client_has_left = (self.perceptions["client"].data[0].id == 0) or self.waiting_for_new_client
                
                self.get_logger().info(f"{CYAN}Glass is empty: {glass_is_empty}{RESET}")
                self.get_logger().info(f"{CYAN}Service was completed: {service_was_completed}{RESET}")
                self.get_logger().info(f"{CYAN}Client has left: {client_has_left} (current client ID: {self.perceptions['client'].data[0].id}){RESET}")
                
                if glass_is_empty and service_was_completed and client_has_left:
                    reward = 1.0
                    self.get_logger().info(f"{GREEN}FULL CYCLE COMPLETED! Client drank, left, and glass returned to preparation table{RESET}")
                    
                    # Reset the flag to prevent multiple rewards for same cycle
                    self.drink_was_served = False
                    self.get_logger().info(f"{YELLOW}Service cycle reset - ready for new client{RESET}")
                else:
                    if not glass_is_empty:
                        self.get_logger().info(f"{YELLOW}Glass still has drink - client hasn't drunk it yet{RESET}")
                    if not service_was_completed:
                        self.get_logger().info(f"{YELLOW}No drink was served in this cycle - cannot get return reward{RESET}")
                    if not client_has_left:
                        self.get_logger().info(f"{YELLOW}Client hasn't left yet - wait for client to leave{RESET}")
                    reward = 0.0
            else:
                self.get_logger().info(f"{BLUE}Glass not in original position{RESET}")
                reward = 0.0
    
        self.perceptions["left_the_glass_goal"].data = reward


    def update_reward_sensor(self):
        """
        Update goal sensors' values.
        Also runs client drinking and departure simulation.
        """
        # First run client lifecycle simulations
        self.simulate_client_drinking()
        self.simulate_client_departure()
        
        # Then update all reward sensors
        for sensor in self.perceptions:
            reward_method = getattr(self, "reward_" + sensor, None)
            if callable(reward_method):
                reward_method()

    def publish_perceptions(self):
        """
        Publish the current perceptions to the corresponding topics.
        """
        for ident, publisher in self.sim_publishers.items():
            self.get_logger().debug("Publishing " + ident + " = " + str(self.perceptions[ident].data))
            publisher.publish(self.perceptions[ident])

    def reset_world(self, request=None):
        """
        Reset the world to initial state.
        
        :param request: The request message (optional)
        :type request: ROS msg or None
        """
        self.get_logger().info(f"{YELLOW}Resetting world...{RESET}")
        
        try:
            # Reset iteration counter
            self.iteration = 0
            
            # Clear all existing data
            #self.bar_clients = []
            self.bottles = []
            self.glass = []
            #self.know_preference = {}
            
            # Reset all client lifecycle tracking
            self.drink_was_served = False
            self.reward_given_for_current_service = False
            self.client_drinking_countdown = 0
            self.client_will_drink = False
            self.client_departure_countdown = 0
            self.client_will_leave = False
            self.no_client_countdown = 0
            self.waiting_for_new_client = False
            
            # Reset glass and position tracking
            self.original_glass_pos = {}
            self.last_glass_pos = {}

            # Reset objects in the hands
            self.perceptions["glass_in_left_hand"].data = False
            self.perceptions["bottle_in_right_hand"].data = False
            
            # Reset asking iterations
            if hasattr(self, 'asking_iterations'):
                self.asking_iterations = {}
            
            # Generate new random perceptions
            self.random_perceptions()
            
            self.get_logger().info(f"{GREEN}World reset completed successfully{RESET}")
            
        except Exception as e:
            self.get_logger().error(f"{RED}Error resetting world: {str(e)}{RESET}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")
            raise

    def world_reset_service_callback(self, request, response):
        """
        Callback for the world reset service.

        :param request: The message that contains the request to reset the world.
        :type request: ROS msg defined in the config file Typically cognitive_processes_interfaces.srv.WorldReset.Request
        :param response: Response of the world reset service.
        :type response: ROS msg defined in the config file. Typically cognitive_processes_interfaces.srv.WorldReset.Response
        :return: Response indicating the success of the world reset.
        :rtype: ROS msg defined in the config file. Typically cognitive_processes_interfaces.srv.WorldReset.Response
        """
        self.reset_world(request)
        response.success=True
        return response

    def new_command_callback(self, data):
        """
        Process a command received

        :param data: The message that contais the command received.
        :type data: ROS msg defined in the config file. Typically cognitive_processes_interfaces.msg.ControlMsg
        """
        self.get_logger().debug(f"Command received... ITERATION: {data.iteration}")
        self.iteration = data.iteration
        self.update_reward_sensor()
        if data.command == "reset_world":
            self.reset_world(data)
        elif data.command == "end":
            self.get_logger().info("Ending simulator as requested by LTM...")
            rclpy.shutdown()

    def new_action_service_callback(self, request, response):
        """
        Execute a policy and publish new perceptions.

        :param request: The message that contains the policy to execute.
        :type request: ROS srv defined in the config file. Typically cognitive_node_interfaces.srv.Policy.Request
        :param response: Response of the success of the execution of the action.
        :type response: ROS srv defined in the config file. Typically cognitive_node_interfaces.srv.Policy.Response
        :return: Response indicating the success of the action execution.
        :rtype: ROS srv defined in the config file. Typically cognitive_node_interfaces.srv.Policy.Response
        """
        try:
            self.get_logger().info("Executing policy " + str(request.policy))
            self.get_logger().info(f"ITERATION: {self.iteration}")
            
            self.perceive_bottles()
            self.perceive_glass()
            self.get_logger().info(f"PERCEPTIONS BEFORE: {self.perceptions}")
            self.get_logger().info(f"POLICY TO EXECUTE: {request.policy}")
            
            # Execute the requested policy
            if hasattr(self, request.policy + "_policy"):
                getattr(self, request.policy + "_policy")()
            else:
                self.get_logger().error(f"Policy {request.policy}_policy does not exist!")
                response.success = False
                return response
            
            self.perceive_bottles()
            self.perceive_glass()
            self.get_logger().info(f"PERCEPTIONS AFTER: {self.perceptions}")
            self.update_reward_sensor()
            self.publish_perceptions()
            
            response.success = True
            return response
            
        except Exception as e:
            self.get_logger().error(f"Error executing policy {request.policy}: {str(e)}")
            response.success = False
            return response
    

    def setup_experiment_stages(self, stages):
        """
        Setup the stages of the experiment with their corresponding iterations.

        :param stages: A dictionary where keys are stage names and values are the iterations at which the stage starts.
        :type stages: dict
        """
        for stage in stages:
            self.change_reward_iterations[stage] = stages[stage]

    def setup_perceptions(self, perceptions):
        """
        Configure the ROS topics where the simulator will publish the perceptions.

        :param perceptions: A list of dictionaries where each dictionary contains the name, perception topic, and perception message class.
        :type perceptions: list
        """
        for perception in perceptions:
            sid = perception["name"]
            topic = perception["perception_topic"]
            classname = perception["perception_msg"]
            message = class_from_classname(classname)
            self.perceptions[sid] = message()
            if "List" in classname:
                self.perceptions[sid].data = []
                self.base_messages[sid] = class_from_classname(classname.replace("List", ""))
            elif "Float" in classname:
                self.perceptions[sid].data = 0.0
            else:
                self.perceptions[sid].data = False
            self.get_logger().info("I will publish to... " + str(topic))
            self.sim_publishers[sid] = self.create_publisher(message, topic, 0)

    def setup_control_channel(self, simulation):
        """
        Configure the ROS topic/service where listen for commands to be executed.

        :param simulation: The params from the config file to setup the control channel.
        :type simulation: dict
        """
        self.ident = simulation["id"]
        topic = simulation["control_topic"]
        classname = simulation["control_msg"]
        message = class_from_classname(classname)
        self.get_logger().info("Subscribing to... " + str(topic))
        self.create_subscription(message, topic, self.new_command_callback, 0)
        service_policy = simulation.get("executed_policy_service")
        service_world_reset = simulation.get("world_reset_service")

        if service_policy:
            self.get_logger().info("Creating server... " + str(service_policy))
            classname = simulation["executed_policy_msg"]
            message_policy_srv = class_from_classname(classname)
            self.create_service(message_policy_srv, service_policy, self.new_action_service_callback, callback_group=self.cbgroup_server)
            self.get_logger().info("Creating perception publisher timer... ")
            self.perceptions_timer = self.create_timer(0.01, self.publish_perceptions, callback_group=self.cbgroup_server)

        if service_world_reset:
            self.message_world_reset = class_from_classname(simulation["world_reset_msg"])
            self.create_service(self.message_world_reset, service_world_reset, self.world_reset_service_callback, callback_group=self.cbgroup_server)     

    def load_experiment_file_in_commander(self):
        """
        Load the configuration file in the commander node.

        :return: Response from the commander node indicating the success of the loading.
        :rtype: core_interfaces.srv.LoadConfig.Response
        """
        loaded = self.load_client.send_request(file = self.config_file)
        return loaded

    def load_configuration(self):
        """
        Load the configuration file and setup the simulator.
        It is configured the random number generator, the stages of the experiment,
        the perceptions, and the control channel.
        """
        if self.random_seed:
            self.rng = numpy.random.default_rng(self.random_seed)
            self.get_logger().info(f"Setting random number generator with seed {self.random_seed}")
        else:
            self.rng = numpy.random.default_rng()

        if self.config_file is None:
            self.get_logger().error("No configuration file for the LTM simulator specified!")
            rclpy.shutdown()
        else:
            if not os.path.isfile(self.config_file):
                self.get_logger().error(self.config_file + " does not exist!")
                rclpy.shutdown()
            else:
                self.get_logger().info(f"Loading configuration from {self.config_file}...")
                config = yaml.load(
                    open(self.config_file, "r", encoding="utf-8"),
                    Loader=yamlloader.ordereddict.CLoader,
                )
                self.setup_experiment_stages(config["DiscreteEventSimulator"]["Stages"])
                self.setup_perceptions(config["DiscreteEventSimulator"]["Perceptions"])
                # Be ware, we can not subscribe to control channel before creating all sensor publishers.
                self.setup_control_channel(config["Control"])
        
        self.load_experiment_file_in_commander()

def main(args=None):
    rclpy.init(args=args)
    sim = BartenderSim()
    sim.load_configuration()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=8)
    executor.add_node(sim)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print('Keyboard Interrupt Detected: Shutting down simulator...')
    finally:
        sim.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

import os
import numpy
import yaml
import yamlloader
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import Float32
from core.service_client import ServiceClient
from core_interfaces.srv import LoadConfig
from core.utils import class_from_classname
import numpy as np
import random

class BartenderSim:
    """
    BartenderSim simulator class - Pure simulation logic without ROS communication.
    """
    def __init__(self, random_seed=1000):
        """
        Constructor of the BartenderSim simulator class.
        Initializes the simulator with state tracking only.
        """
        self.rng = None
        self.random_seed = random_seed
        
        # Simulation steps
        self.steps = ["on_prep","on_prep_with_glass"]

        self.bottles = []
        self.glass = None
        self.original_glass_pos = {}
        self.picked_bottle = 0
        self.agent_bottle_choice = None
        self.know_preference = {}
        self.once_at_prep = True
        self.once_at_serv = True

        self.prep_area = {"x_min": 0.0, "x_max": 0.6, "y_min": 0.9, "y_max": 1.1, "object": "bottles"}
        self.serv_area = {"x_min": 0.4, "x_max": 0.7, "y_min": 0.5, "y_max": 0.9, "object": "glass"}
        self.serving_pos = {"distance": 0.8, "angle": 0.0}

        # Simulation state
        self.robot_position = 0.0
        self.glass_in_left_hand = False
        self.bottle_in_right_hand = False
        self.client_id = 1
        self.client_preference = 0
        
        self.iteration = 0
        self.last_step = -1.0
        self.last_policy_executed = None
        self.prev_policy_executed = None
        self.policy_sequence = []  # Track sequence of policies for detecting loops
        self.sequence_repeat_count = 0  # Count how many times a pattern repeats
        
        # Initialize RNG
        if self.random_seed:
            self.rng = numpy.random.default_rng(self.random_seed)
        else:
            self.rng = numpy.random.default_rng()

    def set_agent_bottle_choice(self, bottle_id):
        """Set the agent's bottle choice."""
        self.agent_bottle_choice = float(bottle_id)

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
            ang = numpy.arctan2(x, y)

            valid = True
            
        return dist, ang

    def generate_bottles(self, n_bottles=3):
        """
        Generate a number of bottles with random positions.
        """
        self.bottles = []
        for i in range(1, n_bottles + 1):
            distance, angle = self.random_position(self.prep_area)
            bottle = dict(distance=distance, angle=angle, id=i)
            self.bottles.append(bottle)

    def generate_glass(self):
        """
        Generate a glass with random position.
        """
        distance, angle = 0.0, 0.0
        self.glass = dict(distance=distance, angle=angle, state=False, drink_type=0.0, was_used=False)
        self.original_glass_pos = {"distance": distance, "angle": angle}

    def get_bottles_state(self):
        """
        Get the current state of bottles.
        Returns a list of bottle dictionaries.
        """
        if not self.bottles:
            return []
        return [{"distance": float(b["distance"]), "angle": float(b["angle"]), "id": int(b["id"])} for b in self.bottles]

    def get_glass_state(self):
        """
        Get the current state of the glass.
        Returns a dictionary with glass properties.
        """
        if not self.glass:
            return {"distance": 0.0, "angle": 0.0, "state": False, "drink_type": 0.0, "was_used": False}
        return {
            "distance": float(self.glass["distance"]),
            "angle": float(self.glass["angle"]),
            "state": bool(self.glass["state"]),
            "drink_type": float(self.glass["drink_type"]),
            "was_used": bool(self.glass["was_used"])
        }

    def reset_world(self):
        """
        Reset the world to a new state.
        """
        self.picked_bottle = 0
        self.agent_bottle_choice = None
        self.last_step = -1.0
        self.prev_policy_executed = None
        self.last_policy_executed = None
        self.policy_sequence = []  # Reset policy sequence on world reset
        self.sequence_repeat_count = 0
        
        # Generate environment
        self.generate_bottles()
        self.generate_glass()

        step = random.choice(self.steps)

        if step == "on_prep":
            # Robot at prep, no glass, no bottle
            self.robot_position = 0.0
            self.glass_in_left_hand = False
            self.bottle_in_right_hand = False

        elif step == "on_prep_with_glass":
            self.robot_position = 0.0
            self.glass_in_left_hand = True
            self.bottle_in_right_hand = False
            self.glass["distance"] = 0.0
            self.glass["angle"] = 0.0
            self.once = False

        elif step == "on_prep_with_bottle":
            self.robot_position = 0.0
            self.glass_in_left_hand = False
            self.bottle_in_right_hand = True
            self.once = False

        elif step == "on_prep_with_both":
            self.robot_position = 0.0
            self.glass_in_left_hand = True
            self.bottle_in_right_hand = True
            self.glass["distance"] = 0.0
            self.glass["angle"] = 0.0
            self.once = False
        
        elif step == "on_prep_with_glass_served":
            self.robot_position = 0.0
            self.glass_in_left_hand = True
            self.bottle_in_right_hand = False
            self.glass["distance"] = 0.0
            self.glass["angle"] = 0.0
            self.glass["state"] = True
            self.glass["drink_type"] = 1.0
            self.once = False

        elif step == "holding_glass_at_serv":
            self.robot_position = 0.95
            self.glass_in_left_hand = True
            self.bottle_in_right_hand = False
            self.glass["distance"] = 0.0
            self.glass["angle"] = 0.0
            self.glass["state"] = True
            self.glass["drink_type"] = 1.0
            self.once = False

        elif step == "holding_both_at_serv":
            self.robot_position = 0.95
            self.glass_in_left_hand = True
            self.bottle_in_right_hand = True
            self.glass["distance"] = 0.0
            self.glass["angle"] = 0.0
            self.glass["state"] = True
            self.glass["drink_type"] = 1.0
            self.once = False
    
        # Client
        cid = 1
        self.client_id = cid
        if cid in self.know_preference:
            self.client_preference = 1
        else:
            self.client_preference = 0
    
    def is_at_preparation_table(self):
        return self.robot_position < 0.2

    def is_at_serving_table(self):
        return self.robot_position >= 0.8

    def glass_is_in_serving_position(self):
        if not self.glass:
            return False
        if self.glass["distance"] >= 0.7:
            return True
        else:
            return False
        # if not self.glass:
        #     return False
        # g = self.glass
        # if float(g["distance"]) < 0 or float(g["angle"]) < 0:
        #     return False
        # # rounded equality check
        # gd = round(float(g["distance"]), 1)
        # ga = round(float(g["angle"]), 1)
        # sd = round(float(self.serving_pos["distance"]), 2)
        # sa = round(float(self.serving_pos["angle"]), 2)
        # return (gd == sd) and (abs(ga) == abs(sa))

    def glass_is_in_preparation_area(self):
        if not self.glass:
            return False
        if self.glass["distance"] < 0.2:
            return True
        else:
            return False
        # if not self.glass:
        #     return False
        # g = self.glass
        # d = float(g["distance"])
        # a = float(g["angle"])
        # # Convert polar (d, a) back to Cartesian (x, y) assuming a = arctan2(x, y)
        # x = d * np.sin(a)
        # y = d * np.cos(a)
        # area = self.prep_area
        # return (area["x_min"] <= x <= area["x_max"]) and (area["y_min"] <= y <= area["y_max"])


    def pick_glass_policy(self):
        if self.glass_in_left_hand:
            return
        
        if self.robot_position < 0.2 and self.glass_is_in_preparation_area():
            self.glass_in_left_hand = True
            if self.glass:
                self.glass["distance"] = 0.5
                self.glass["angle"] = 0.0
        if self.robot_position >= 0.8 and self.glass_is_in_serving_position():
            self.glass_in_left_hand = True
            if self.glass:
                self.glass["distance"] = 0.5
                self.glass["angle"] = 0.0

    def pick_bottle_policy(self):
        """
        Pick a bottle based on agent choice.
        """
        if self.bottle_in_right_hand:
            return
        
        if not self.is_at_preparation_table():
            return
            
        bottle_id = 1
        if bottle_id == 0:
            return

        self.bottle_in_right_hand = True
        self.picked_bottle = int(bottle_id)
        for b in self.bottles:
            if int(b["id"]) == self.picked_bottle:
                b["distance"] = 0.0
                b["angle"] = 1.4
                break

    def prepare_drink_policy(self):
        """
        Prepare the drink if holding glass and bottle.
        """
        if not self.glass_in_left_hand:
            return
        if not self.bottle_in_right_hand:
            return
        if self.glass and self.glass["state"]:
            return

        if self.glass:
            self.glass["state"] = True
            self.glass["drink_type"] = float(self.picked_bottle)

    def place_glass_policy(self):
        """
        Place glass in appropriate location based on context.
        - At serving table with prepared drink: place on serving area
        - At prep table with used glass: return to original position
        """
        if not self.glass_in_left_hand:
            return
        
        if self.is_at_serving_table():
            # Place glass on serving table if holding it
            self.glass_in_left_hand = False
            if self.glass:
                self.glass["distance"] = self.serving_pos["distance"]
                self.glass["angle"] = self.serving_pos["angle"]
                if self.glass["state"]:
                    self.glass["was_used"] = True
                    self.glass["state"] = False  # After placing, glass is empty (client will drink)
        
        # elif self.is_at_preparation_table():
        #     # Return glass to preparation area if holding used glass
        #     self.glass_in_left_hand = False
        #     if self.glass:
        #         self.glass["distance"] = self.original_glass_pos["distance"]
        #         self.glass["angle"] = self.original_glass_pos["angle"]

    # def place_bottle_policy(self):
    #     """
    #     Place bottle back.
    #     """
    #     if not self.is_at_preparation_table():
    #         return
    #     if not self.bottle_in_right_hand:
    #         return
            
    #     self.bottle_in_right_hand = False
    #     self.picked_bottle = 0
    #     for b in self.bottles:
    #         distance, angle = self.random_position(self.prep_area)
    #         b["distance"] = distance
    #         b["angle"] = angle  
            

    # def change_position_policy(self):
    #     """
    #     Toggle robot position between prep and serving.
    #     Restricted: Can only go to serving if holding glass with prepared drink
    #     """
    #     if self.is_at_preparation_table():
    #         # Only allow moving to serving if holding glass with prepared drink
    #         if self.glass_in_left_hand and self.glass and self.glass["state"]:
    #             self.robot_position = 0.95
    #     # elif self.is_at_serving_table():
    #     #     # Allow returning to prep if: not holding glass, OR holding used glass
    #     #     if not self.glass_in_left_hand:
    #     #         self.robot_position = 0.0
    #     #     elif self.glass and self.glass["was_used"]:
    #     #         self.robot_position = 0.0

    def ask_nicely_policy(self):
        """
        Ask client for preference.
        """
        cid = int(self.client_id)
        pref = cid  # Simple mapping
        self.know_preference[cid] = pref
        self.client_preference = pref

    def pick_bottle_policy(self):
        """
        Pick a bottle based on agent choice.
        """
        if self.bottle_in_right_hand:
            return
        
        if not self.is_at_preparation_table():
            return
            
        bottle_id = 1
        if bottle_id == 0:
            return

        self.bottle_in_right_hand = True
        self.picked_bottle = int(bottle_id)
        for b in self.bottles:
            if int(b["id"]) == self.picked_bottle:
                b["distance"] = 0.0
                b["angle"] = 1.4
                break

    def prepare_drink_policy(self):
        """
        Prepare the drink if holding glass and bottle.
        """
        if not self.glass_in_left_hand:
            return
        if not self.bottle_in_right_hand:
            return
        if self.glass and self.glass["state"]:
            return

        if self.glass:
            self.glass["state"] = True
            self.glass["drink_type"] = float(self.picked_bottle)

    def place_glass_policy(self):
        """
        Place glass in appropriate location based on context.
        - At serving table with prepared drink: place on serving area
        - At prep table with used glass: return to original position
        """
        # Backward-compatible wrapper: delegate to serving or return policies
        if self.glass and self.glass_in_left_hand and self.glass["state"]:
            return self.place_glass_serving_policy()
        return self.return_glass_policy()

    def place_glass_serving_policy(self):
        """
        Move to serving area and place the glass there if it is prepared.
        """
        if not self.glass_in_left_hand:
            return
        if not self.glass:
            return
        if not self.glass.get("state", False):
            return

        self.robot_position = 0.95  # move to serving area
        self.glass_in_left_hand = False
        self.glass["distance"] = self.serving_pos["distance"]
        self.glass["angle"] = self.serving_pos["angle"]
        self.glass["was_used"] = True
        self.glass["state"] = False  # client takes the drink

    def return_glass_policy(self):
        """
        Move to the return/prep area and place the glass back in its original spot.
        """
        if not self.glass_in_left_hand:
            return
        if not self.glass:
            return

        self.robot_position = 0.0  # move back to prep/return area
        self.glass_in_left_hand = False
        self.glass["distance"] = self.original_glass_pos["distance"]
        self.glass["angle"] = self.original_glass_pos["angle"]

    def place_bottle_policy(self):
        """
        Place bottle back.
        """
        if not self.is_at_preparation_table():
            return
        if not self.bottle_in_right_hand:
            return
            
        self.bottle_in_right_hand = False
        self.picked_bottle = 0
        for b in self.bottles:
            distance, angle = self.random_position(self.prep_area)
            b["distance"] = distance
            b["angle"] = angle  
            

    def change_position_policy(self):
        """
        Toggle robot position between prep and serving.
        Restricted: Can only go to serving if holding glass with drink,
        and can only return to prep if at serving without anything problematic.
        """
        if self.is_at_preparation_table():
            # Only allow moving to serving if holding glass with prepared drink
            if self.glass_in_left_hand and self.glass and self.glass["state"]:
                self.robot_position = 0.95
        elif self.is_at_serving_table():
            # Allow returning to prep if: not holding glass, OR holding used glass
            if not self.glass_in_left_hand:
                self.robot_position = 0.0
            elif self.glass and self.glass["was_used"]:
                self.robot_position = 0.0

    def ask_nicely_policy(self):
        """
        Ask client for preference.
        """
        cid = int(self.client_id)
        pref = cid  # Simple mapping
        self.know_preference[cid] = pref
        self.client_preference = pref

    def get_progress_goal(self):
        """
        Calculate reward based on the immediate current state (no accumulated progress).
        """
        has_glass = self.glass_in_left_hand
        has_bottle = self.bottle_in_right_hand
        
        glass_state = self.glass["state"] if self.glass else False
        was_used = self.glass["was_used"] if self.glass else False
        
        at_prep = self.is_at_preparation_table()
        at_serv = self.is_at_serving_table()
        
        # Check glass position
        g_dist = self.glass["distance"] if self.glass else 0.0
        g_ang = self.glass["angle"] if self.glass else 0.0
        
        glass_at_serving = (abs(g_dist - self.serving_pos["distance"]) < 0.1 and 
                            abs(g_ang - self.serving_pos["angle"]) < 0.1)
                            
        glass_at_original = (abs(g_dist - self.original_glass_pos["distance"]) < 0.1 and 
                             abs(g_ang - self.original_glass_pos["angle"]) < 0.1)

        # Determine current state step
        current_step = 0.0
        
        if glass_at_original and not has_glass and was_used:
            current_step = 1.0
        elif at_prep and has_glass and not glass_state and was_used:
            current_step = 0.9
        elif has_glass and not glass_state and was_used:
            current_step = 0.85
        elif glass_at_serving and not glass_state and not has_glass:
            current_step = 0.8
        elif glass_at_serving and glass_state and not has_glass:
            current_step = 0.6
        elif at_serv and has_glass and glass_state:
            current_step = 0.5
        elif at_prep and has_glass and glass_state:
            current_step = 0.4
        elif has_glass and has_bottle and at_prep:
            current_step = 0.25
        elif has_glass and at_prep:
            current_step = 0.15
        elif has_bottle and at_prep:
            current_step = 0.1
        elif at_prep and not has_glass and not has_bottle:
            current_step = 0.05
    
        reward = current_step

        # Zero out reward if the agent repeats the exact same policy consecutively
        repeated_policy = (
            self.last_policy_executed is not None and
            self.prev_policy_executed == self.last_policy_executed
        )

        if repeated_policy:
            reward = 0.0
            # Detect looping patterns (e.g., [pick_glass, place_glass_return, pick_glass, ...])
            if len(self.policy_sequence) >= 2:
                # Check if last 2 policies form a repeating pattern
                recent_pattern = tuple(self.policy_sequence[-2:])
                if len(self.policy_sequence) >= 4:
                    prev_pattern = tuple(self.policy_sequence[-4:-2])
                    if recent_pattern == prev_pattern:
                        self.sequence_repeat_count += 1
                        # Penalize based on how many times we've repeated the pattern
                        penalty = min(0.3 * self.sequence_repeat_count, 1.0)
                        reward = max(0.0 - penalty, -0.5)
                    else:
                        self.sequence_repeat_count = 0

        self.last_step = current_step

        return float(reward)

    def get_serve_the_drink_goal(self):
        """
        Calculate serve drink goal reward.
        """
        reward = 0.0
        glass_at_serving = self.glass_is_in_serving_position()
        glass_at_original = self.glass_is_in_preparation_area()
                            
        if glass_at_original and self.glass and self.glass["was_used"]:
             reward = 1.0
        
        return reward

    def get_return_the_glass_goal(self):
        """
        Calculate return glass goal reward.
        """
        reward = 0.0
        if not self.glass:
            return reward
            
        g_dist = self.glass["distance"]
        g_ang = self.glass["angle"]
        glass_at_original = (abs(g_dist - self.original_glass_pos["distance"]) < 0.1 and 
                             abs(g_ang - self.original_glass_pos["angle"]) < 0.1)
                             
        if glass_at_original and self.glass["was_used"] and not self.glass_in_left_hand:
            reward = 1.0
            
        return reward

    def execute_policy(self, policy_name):
        """
        Execute a policy by name.
        """
        policy_method = getattr(self, policy_name + "_policy", None)
        if policy_method and callable(policy_method):
            self.prev_policy_executed = self.last_policy_executed
            self.last_policy_executed = policy_name
            self.policy_sequence.append(policy_name)  # Track policy sequence
            # Keep only last 10 policies to avoid memory issues
            if len(self.policy_sequence) > 10:
                self.policy_sequence.pop(0)
            policy_method()
            return True
        return False


class BartenderSimNode(Node):
    """
    ROS2 Node wrapper for BartenderSim that handles perceptions and communication.
    """
    def __init__(self):
        """
        Constructor of the BartenderSimNode class.
        Initializes ROS communication and wraps the simulator.
        """
        super().__init__("BartenderSim")
        
        self.ident = None
        self.base_messages = {}
        self.perceptions = {}
        self.sim_publishers = {}
        self.change_reward_iterations = {}

        self.random_seed = self.declare_parameter('random_seed', value=1000).get_parameter_value().integer_value
        self.config_file = self.declare_parameter('config_file', descriptor=ParameterDescriptor(dynamic_typing=True)).get_parameter_value().string_value
        
        # Create the pure simulator
        self.simulator = BartenderSim(random_seed=self.random_seed)
        
        self.cbgroup_server = MutuallyExclusiveCallbackGroup()
        self.cbgroup_client = MutuallyExclusiveCallbackGroup()

        self.load_client = ServiceClient(LoadConfig, 'commander/load_experiment')
        
        self.agent_bottle_subscription = self.create_subscription(
            Float32,
            "cognitive_node/world_model/last_bottle",
            self.agent_bottle_callback,
            10
        )

    def agent_bottle_callback(self, msg):
        self.simulator.set_agent_bottle_choice(float(msg.data))

    def perceive_bottles(self):
        """
        Update the bottles perceptions from simulator state.
        """
        self.perceptions["bottles"].data = []
        bottles_state = self.simulator.get_bottles_state()
        if bottles_state:
            for b in bottles_state:
                msg = self.base_messages["bottles"]()
                msg.distance = float(b["distance"])
                msg.angle = float(b["angle"])
                if hasattr(msg, "id"):
                    msg.id = int(b["id"])
                self.perceptions["bottles"].data.append(msg)
        else:
            self.perceptions["bottles"].data.append(self.base_messages["bottles"]())

    def perceive_glass(self):
        """
        Update the glass perceptions from simulator state.
        """
        self.perceptions["glass"].data = []
        msg = self.base_messages["glass"]()
        glass_state = self.simulator.get_glass_state()
        msg.distance = float(glass_state["distance"])
        msg.angle = float(glass_state["angle"])
        msg.state = bool(glass_state["state"])
        msg.drink_type = float(glass_state["drink_type"])
        msg.was_used = bool(glass_state["was_used"])
        self.perceptions["glass"].data.append(msg)

    def update_perceptions_from_simulator(self):
        """
        Update all perceptions from simulator state.
        """
        self.perceive_bottles()
        self.perceive_glass()
        
        self.perceptions["robot_position"].data = float(self.simulator.robot_position)
        self.perceptions["glass_in_left_hand"].data = bool(self.simulator.glass_in_left_hand)
        self.perceptions["bottle_in_right_hand"].data = bool(self.simulator.bottle_in_right_hand)
        
        # Client
        self.perceptions["client"].data = []
        client_msg = self.base_messages["client"]()
        client_msg.id = int(self.simulator.client_id)
        client_msg.preference = int(self.simulator.client_preference)
        self.perceptions["client"].data.append(client_msg)

    def reset_world(self, data=None):
        """
        Reset the world to a new state.
        """
        self.get_logger().info("Resetting world...")
        self.simulator.reset_world()
        self.update_perceptions_from_simulator()
        self.update_reward_sensor()
        self.publish_perceptions()

    def update_reward_sensor(self):
        """
        Update goal sensors' values from simulator.
        """
        if "progress_goal" in self.perceptions:
            progress = self.simulator.get_progress_goal()
            self.perceptions["progress_goal"].data = progress
            self.get_logger().info(f"Progress reward: {progress}")
        if "serve_the_drink_goal" in self.perceptions:
            self.perceptions["serve_the_drink_goal"].data = self.simulator.get_serve_the_drink_goal()
        if "return_the_glass_goal" in self.perceptions:
            self.perceptions["return_the_glass_goal"].data = self.simulator.get_return_the_glass_goal()
    
    def publish_perceptions(self):
        """
        Publish the current perceptions to the corresponding topics.
        """
        for ident, publisher in self.sim_publishers.items():
            self.get_logger().debug("Publishing " + ident + " = " + str(self.perceptions[ident].data))
            publisher.publish(self.perceptions[ident])

    def world_reset_service_callback(self, request, response):
        """
        Callback for the world reset service.
        """
        self.reset_world(request)
        response.success = True
        return response

    def new_command_callback(self, data):
        """
        Process a command received.
        """
        self.get_logger().debug(f"Command received... ITERATION: {data.iteration}")
        self.simulator.iteration = data.iteration
        self.update_reward_sensor()
        if data.command == "reset_world":
            self.reset_world(data)
        elif data.command == "end":
            self.get_logger().info("Ending simulator as requested by LTM...")
            rclpy.shutdown()

    def new_action_service_callback(self, request, response):
        """
        Execute a policy and publish new perceptions.
        """
        self.get_logger().info("Executing policy " + str(request.policy))
        self.get_logger().info(f"ITERATION: {self.simulator.iteration}")
        
        # Pre-perception update
        self.update_perceptions_from_simulator()
        
        self.get_logger().info(f"PERCEPTIONS BEFORE: {self.perceptions}")
        self.get_logger().info(f"POLICY TO EXECUTE: {request.policy}")
        
        # Execute policy in simulator
        self.simulator.execute_policy(request.policy)
        
        # Post-perception update
        self.update_perceptions_from_simulator()
        
        self.get_logger().info(f"PERCEPTIONS AFTER: {self.perceptions}")
        self.update_reward_sensor()
        self.publish_perceptions()
        
        response.success = True
        return response

    def setup_experiment_stages(self, stages):
        """
        Setup the stages of the experiment with their corresponding iterations.
        """
        for stage in stages:
            self.change_reward_iterations[stage] = stages[stage]

    def setup_perceptions(self, perceptions):
        """
        Configure the ROS topics where the simulator will publish the perceptions.
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
        """
        loaded = self.load_client.send_request(file=self.config_file)
        return loaded

    def load_configuration(self):
        """
        Load the configuration file and setup the simulator.
        """
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
                self.setup_control_channel(config["Control"])
        
        self.load_experiment_file_in_commander()



def main(args=None):
    rclpy.init(args=args)
    sim = BartenderSimNode()
    sim.load_configuration()

    try:
        rclpy.spin(sim)
    except KeyboardInterrupt:
        print('Keyboard Interrupt Detected: Shutting down simulator...')
    finally:
        sim.destroy_node()

if __name__ == '__main__':
    main()
 
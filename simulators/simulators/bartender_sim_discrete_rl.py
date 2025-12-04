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
        self.steps = ["on_prep","on_prep_with_glass","on_prep_with_bottle","on_prep_with_both", "on_prep_with_glass_served",
                      "holding_glass_at_serv","holding_both_at_serv"]

        self.random_seed = self.declare_parameter('random_seed', value = 0).get_parameter_value().integer_value
        self.config_file = self.declare_parameter('config_file', descriptor=ParameterDescriptor(dynamic_typing=True)).get_parameter_value().string_value
        
        self.bottles = []
        self.glass = None
        self.original_glass_pos = {}
        self.picked_bottle = 0
        self.agent_bottle_choice = None
        self.know_preference = {}

        self.prep_area = {"x_min": 0.0, "x_max": 0.6, "y_min": 0.9, "y_max": 1.1, "object": "bottles"}
        self.serv_area = {"x_min": 0.4, "x_max": 0.7, "y_min": 0.5, "y_max": 0.9, "object": "glass"}
        self.serving_pos = {"distance": 0.8, "angle": 0.0}

        self.iteration = 0
        self.change_reward_iterations = {}

        self.cbgroup_server=MutuallyExclusiveCallbackGroup()
        self.cbgroup_client=MutuallyExclusiveCallbackGroup()

        self.load_client=ServiceClient(LoadConfig, 'commander/load_experiment')
        
        self.agent_bottle_subscription = self.create_subscription(
            Float32,
            "cognitive_node/world_model/last_bottle",
            self.agent_bottle_callback,
            10
        )

    def agent_bottle_callback(self, msg):
        self.agent_bottle_choice = float(msg.data)

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
            
            key = area.get("object")
            if key and key in self.perceptions and hasattr(self.perceptions[key], "data"):
                # Check collision with existing objects of the same type
                # This is a simplification from fruit_shop which checked against all objects in the area
                # For bartender, we mainly care about bottles not overlapping
                pass 
            
        return dist, ang

    def generate_bottles(self, n_bottles=3):
        """
        Generate a number of bottles with random positions.
        """
        self.get_logger().info("Generating bottles...")
        self.bottles = []
        for i in range(1, n_bottles + 1):
            distance, angle = self.random_position(self.prep_area)
            bottle = dict(distance=distance, angle=angle, id=i)
            self.bottles.append(bottle)

    def generate_glass(self):
        """
        Generate a glass with random position.
        """
        self.get_logger().info("Generating glass...")
        distance, angle = self.random_position(self.prep_area)
        self.glass = dict(distance=distance, angle=angle, state=False, drink_type=0.0, was_used=False)
        self.original_glass_pos = {"distance": distance, "angle": angle}

    def perceive_bottles(self):
        """
        Update the bottles perceptions.
        """
        self.perceptions["bottles"].data = []
        if self.bottles:
            for b in self.bottles:
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
        Update the glass perceptions.
        """
        self.perceptions["glass"].data = []
        msg = self.base_messages["glass"]()
        if self.glass:
            msg.distance = float(self.glass["distance"])
            msg.angle = float(self.glass["angle"])
            msg.state = bool(self.glass["state"])
            msg.drink_type = float(self.glass["drink_type"])
            msg.was_used = bool(self.glass["was_used"])
        self.perceptions["glass"].data.append(msg)

    def random_perceptions(self):
        """
        Generate random perceptions when the world is reset.
        """
        self.picked_bottle = 0
        self.agent_bottle_choice = None
        
        # Generate environment
        self.generate_bottles()
        self.generate_glass()
        
        # Update perceptions
        self.perceive_bottles()
        self.perceive_glass()

        step = random.choice(self.steps)
        self.get_logger().info(f"Randomly selected step: {step}")

        if step == "on_prep":
            # Robot at prep, no glass, no bottle
            self.perceptions["robot_position"].data = 0.0
            self.perceptions["glass_in_left_hand"].data = False
            self.perceptions["bottle_in_right_hand"].data = False

        elif step == "on_prep_with_glass":
            self.perceptions["robot_position"].data = 0.0
            self.perceptions["glass_in_left_hand"].data = True
            self.perceptions["bottle_in_right_hand"].data = False
            self.perceptions["glass"].data[0].distance = 0.0
            self.perceptions["glass"].data[0].angle = 0.0

        elif step == "on_prep_with_bottle":
            self.perceptions["robot_position"].data = 0.0
            self.perceptions["glass_in_left_hand"].data = False
            self.perceptions["bottle_in_right_hand"].data = True

        elif step == "on_prep_with_both":
            self.perceptions["robot_position"].data = 0.0
            self.perceptions["glass_in_left_hand"].data = True
            self.perceptions["bottle_in_right_hand"].data = True
            self.perceptions["glass"].data[0].distance = 0.0
            self.perceptions["glass"].data[0].angle = 0.0
        
        elif step == "on_prep_with_glass_served":
            self.perceptions["robot_position"].data = 0.0
            self.perceptions["glass_in_left_hand"].data = True
            self.perceptions["bottle_in_right_hand"].data = False
            self.perceptions["glass"].data[0].distance = 0.0
            self.perceptions["glass"].data[0].angle = 0.0
            self.perceptions["glass"].data[0].state = True
            self.perceptions["glass"].data[0].drink_type = 1.0

        elif step == "holding_glass_at_serv":
            self.perceptions["robot_position"].data = 0.95
            self.perceptions["glass_in_left_hand"].data = True
            self.perceptions["bottle_in_right_hand"].data = False
            self.perceptions["glass"].data[0].distance = 0.0
            self.perceptions["glass"].data[0].angle = 0.0
            self.perceptions["glass"].data[0].state = True
            self.perceptions["glass"].data[0].drink_type = 1.0

        elif step == "holding_both_at_serv":
            self.perceptions["robot_position"].data = 0.95
            self.perceptions["glass_in_left_hand"].data = True
            self.perceptions["bottle_in_right_hand"].data = True
            self.perceptions["glass"].data[0].distance = 0.0
            self.perceptions["glass"].data[0].angle = 0.0
            self.perceptions["glass"].data[0].state = True
            self.perceptions["glass"].data[0].drink_type = 1.0
    
        
        # Client
        self.perceptions["client"].data = []
        self.perceptions["client"].data.append(self.base_messages["client"]())
        cid = 1
        self.perceptions["client"].data[0].id = cid
        if cid in self.know_preference:
            self.perceptions["client"].data[0].preference = 1
        else:
            self.perceptions["client"].data[0].preference = 0

        self.update_reward_sensor()
    
    def is_at_preparation_table(self):
        return bool(self.perceptions["robot_position"].data) < 0.2

    def is_at_serving_table(self):
        return bool(self.perceptions["robot_position"].data) >= 0.8

    def glass_is_in_serving_position(self):
        g = self.perceptions["glass"].data[0]
        if float(g.distance) < 0 or float(g.angle) < 0:
            return False
        # rounded equality check
        gd = round(float(g.distance), 1)
        ga = round(float(g.angle), 1)
        sd = round(float(self.serving_pos["distance"]), 2)
        sa = round(float(self.serving_pos["angle"]), 2)
        return (gd == sd) and (abs(ga) == abs(sa))

    def glass_is_in_preparation_area(self) :
        g = self.perceptions["glass"].data[0]
        d = float(g.distance)
        a = float(g.angle)
        # Convert polar (d, a) back to Cartesian (x, y) assuming a = arctan2(x, y)
        x = d * np.sin(a)
        y = d * np.cos(a)
        area = self.prep_area
        return (area["x_min"] <= x <= area["x_max"]) and (area["y_min"] <= y <= area["y_max"])


    def pick_glass_policy(self):
        if self.perceptions["glass_in_left_hand"].data:
            return False
        if not self.glass:
            return False

        # Check spatial consistency: Robot and Glass must be at the same table (Prep or Serv)
        at_prep = self.is_at_preparation_table()
        at_serv = self.is_at_serving_table()
        glass_in_prep = self.glass_is_in_preparation_area()
        glass_in_serv = self.glass_is_in_serving_position()

        if not ((at_prep and glass_in_prep) or (at_serv and glass_in_serv)):
            self.get_logger().info(f"[BLOCKED] pick_glass: spatial mismatch. Robot(prep={at_prep}, serv={at_serv}) vs Glass(prep={glass_in_prep}, serv={glass_in_serv})")
            return False

        cur = self.perceptions["glass"].data[0]
        self.perceptions["glass_in_left_hand"].data = True
        self.perceptions["glass"].data[0].distance = 0.0
        self.perceptions["glass"].data[0].angle = 0.0
        if not self.original_glass_pos:
            self.original_glass_pos = {"distance": float(cur.distance), "angle": float(cur.angle)}
        self.last_glass_pos = {"distance": float(cur.distance), "angle": float(cur.angle)}

        IN_HAND_DISTANCE = 0.0
        IN_HAND_ANGLE = 0.0
        self.perceptions["glass"].data[0].distance = IN_HAND_DISTANCE
        self.perceptions["glass"].data[0].angle = IN_HAND_ANGLE
        if self.glass:
            self.glass["distance"] = IN_HAND_DISTANCE
            self.glass["angle"] = IN_HAND_ANGLE
        self.get_logger().info(f"Glass picked")
        return True

    def pick_bottle_policy(self):
        """
        Pick a bottle based on agent choice.
        """
        self.get_logger().info(f"Agent bottle choice: {self.agent_bottle_choice}")
        if self.perceptions["bottle_in_right_hand"].data:
            return
        
        if not self.is_at_preparation_table():
            return
            
        bottle_id = 1
        if bottle_id == 0:
            return

        self.perceptions["bottle_in_right_hand"].data = True
        self.picked_bottle = int(bottle_id)

    def prepare_drink_policy(self):
        """
        Prepare the drink if holding glass and bottle.
        """
        if not self.perceptions["glass_in_left_hand"].data:
            return
        if not self.perceptions["bottle_in_right_hand"].data:
            return
        if self.perceptions["glass"].data[0].state:
            return

        self.perceptions["glass"].data[0].state = True
        self.perceptions["glass"].data[0].drink_type = float(self.picked_bottle)
        if self.glass:
            self.glass["state"] = True
            self.glass["drink_type"] = float(self.picked_bottle)

    def place_glass_serving_policy(self):
        """
        Place glass on serving table.
        """
        if not self.is_at_serving_table():
            return
        if not self.perceptions["glass_in_left_hand"].data:
            return
        
        self.perceptions["glass_in_left_hand"].data = False
        self.glass["distance"] = self.serving_pos["distance"]
        self.glass["angle"] = self.serving_pos["angle"]
        self.perceptions["glass"].data[0].distance = self.serving_pos["distance"]
        self.perceptions["glass"].data[0].angle = self.serving_pos["angle"]
        
        if self.perceptions["glass"].data[0].state:
             self.glass["state"] = False
             self.glass["drink_type"] = 0.0
             self.glass["was_used"] = True
             self.perceptions["glass"].data[0].state = False
             self.perceptions["glass"].data[0].drink_type = 0.0
             self.perceptions["glass"].data[0].was_used = True

    def place_glass_return_policy(self):
        """
        Return glass to preparation table.
        """
        if not self.is_at_preparation_table():
            return
        if not self.perceptions["glass_in_left_hand"].data:
            return
            
        self.perceptions["glass_in_left_hand"].data = False
        self.glass["distance"] = self.original_glass_pos["distance"]
        self.glass["angle"] = self.original_glass_pos["angle"]
        self.perceptions["glass"].data[0].distance = self.original_glass_pos["distance"]
        self.perceptions["glass"].data[0].angle = self.original_glass_pos["angle"]

    def place_bottle_policy(self):
        """
        Place bottle back.
        """
        if not self.is_at_preparation_table():
            return
        if not self.perceptions["bottle_in_right_hand"].data:
            return
            
        self.perceptions["bottle_in_right_hand"].data = False
        self.picked_bottle = 0

    def change_position_policy(self):
        """
        Toggle robot position between prep and serving.
        """
        if self.is_at_preparation_table():
            self.perceptions["robot_position"].data = 0.95
        else:
            self.perceptions["robot_position"].data = 0.0

    def ask_nicely_policy(self):
        """
        Ask client for preference.
        """
        cid = int(self.perceptions["client"].data[0].id)
        # In a real sim we would look up the client's preference.
        # Here we generate it or retrieve it.
        # For simplicity, let's say preference is same as ID for now or random
        # But we need to be consistent.
        # In random_perceptions we set preference to 0 if unknown.
        # We need a source of truth for clients.
        # Let's just say preference = cid for simplicity in this "fruit shop style" rewrite
        # unless we want to keep the "bar_clients" list.
        # I'll use a simple mapping.
        pref = cid # Simple mapping
        self.know_preference[cid] = pref
        self.perceptions["client"].data[0].preference = pref

    def reward_progress_goal(self):
        """
        Calculate progress reward.
        """
        # Logic copied/adapted from bartender_sim_discrete_rl.py
        has_glass = self.perceptions["glass_in_left_hand"].data
        has_bottle = self.perceptions["bottle_in_right_hand"].data
        
        glass_state = self.perceptions["glass"].data[0].state
        was_used = self.perceptions["glass"].data[0].was_used
        
        at_prep = self.is_at_preparation_table()
        at_serv = self.is_at_serving_table()
        
        # Check glass position
        g_dist = self.perceptions["glass"].data[0].distance
        g_ang = self.perceptions["glass"].data[0].angle
        
        glass_at_serving = (abs(g_dist - self.serving_pos["distance"]) < 0.1 and 
                            abs(g_ang - self.serving_pos["angle"]) < 0.1)
                            
        glass_at_original = (abs(g_dist - self.original_glass_pos["distance"]) < 0.1 and 
                             abs(g_ang - self.original_glass_pos["angle"]) < 0.1)

        step = 0.0
        
        if glass_at_original and not has_glass and was_used:
            step = 1.0
        elif at_prep and has_glass and not glass_state and was_used:
            step = 0.9
        elif has_glass and not glass_state and was_used:
            step = 0.85
        elif glass_at_serving and not glass_state and not has_glass and was_used:
            step = 0.8 # Client drank
        elif glass_at_serving and glass_state and not has_glass:
            step = 0.6
        elif at_serv and has_glass and glass_state:
            step = 0.5
        elif at_prep and has_glass and glass_state:
            step = 0.4
        elif has_glass and has_bottle and at_prep:
            step = 0.25
        elif has_glass and at_prep:
            step = 0.15
        elif has_bottle and at_prep:
            step = 0.1
        elif at_prep and not has_glass and not has_bottle:
            step = 0.05
            
        self.perceptions["progress_goal"].data = float(step)

    def reward_serve_the_drink_goal(self):
        reward = 0.0
        # If glass is at serving and has drink (or was just drunk?)
        # The original code gave 1.0 if glass.state is True (has drink) and is at serving pos.
        # But wait, if client drinks, state becomes False.
        # So we should reward when it is placed.
        # In fruit shop, rewards are calculated based on state.
        
        g_dist = self.perceptions["glass"].data[0].distance
        g_ang = self.perceptions["glass"].data[0].angle
        glass_at_serving = (abs(g_dist - self.serving_pos["distance"]) < 0.1 and 
                            abs(g_ang - self.serving_pos["angle"]) < 0.1)
                            
        if glass_at_serving and self.perceptions["glass"].data[0].was_used:
             reward = 1.0
        
        self.perceptions["serve_the_drink_goal"].data = reward

    def reward_return_the_glass_goal(self):
        reward = 0.0
        g_dist = self.perceptions["glass"].data[0].distance
        g_ang = self.perceptions["glass"].data[0].angle
        glass_at_original = (abs(g_dist - self.original_glass_pos["distance"]) < 0.1 and 
                             abs(g_ang - self.original_glass_pos["angle"]) < 0.1)
                             
        if glass_at_original and self.perceptions["glass"].data[0].was_used and not self.perceptions["glass_in_left_hand"].data:
            reward = 1.0
            
        self.perceptions["return_the_glass_goal"].data = reward

    def reset_world(self, data):
        """
        Reset the world to a new state.
        """
        self.get_logger().info("Resetting world...")
        self.random_perceptions()
        self.publish_perceptions()

    def update_reward_sensor(self):
        """
        Update goal sensors' values.
        """
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

    def world_reset_service_callback(self, request, response):
        """
        Callback for the world reset service.
        """
        self.reset_world(request)
        response.success=True
        return response

    def new_command_callback(self, data):
        """
        Process a command received
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
        """
        self.get_logger().info("Executing policy " + str(request.policy))
        self.get_logger().info(f"ITERATION: {self.iteration}")
        
        # Pre-perception update (in case things changed externally, though here it's static)
        self.perceive_bottles()
        self.perceive_glass()
        
        self.get_logger().info(f"PERCEPTIONS BEFORE: {self.perceptions}")
        self.get_logger().info(f"POLICY TO EXECUTE: {request.policy}")
        
        getattr(self, request.policy + "_policy")()
        
        # Post-perception update
        self.perceive_bottles()
        self.perceive_glass()
        
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
        loaded = self.load_client.send_request(file = self.config_file)
        return loaded

    def load_configuration(self):
        """
        Load the configuration file and setup the simulator.
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
                self.setup_control_channel(config["Control"])
        
        self.load_experiment_file_in_commander()

def main(args=None):
    rclpy.init(args=args)
    sim = BartenderSim()
    sim.load_configuration()

    try:
        rclpy.spin(sim)
    except KeyboardInterrupt:
        print('Keyboard Interrupt Detected: Shutting down simulator...')
    finally:
        sim.destroy_node()

if __name__ == '__main__':
    main()

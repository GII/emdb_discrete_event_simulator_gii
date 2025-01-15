import os
import numpy
import yaml
import yamlloader
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from core.service_client import ServiceClient
from core_interfaces.srv import LoadConfig
from core.utils import class_from_classname

class IJCNNSim(Node):
    def __init__(self):
        super().__init__("IJCNNSim")
        self.rng = None
        self.ident = None
        self.base_messages = {}
        self.perceptions = {}
        self.sim_publishers = {}

        self.random_seed = self.declare_parameter('random_seed', value = 0).get_parameter_value().integer_value
        self.config_file = self.declare_parameter('config_file', descriptor=ParameterDescriptor(dynamic_typing=True)).get_parameter_value().string_value
        self.fruits = []
        self.catched_fruit = None

        self.normal_inner = numpy.poly1d(
        numpy.polyfit([0.0, 0.3925, 0.785, 1.1775, 1.57], [0.45, 0.47, 0.525, 0.65, 0.9], 3)
        )
        self.normal_outer = numpy.poly1d(
        numpy.polyfit([0.0, 0.3925, 0.785, 1.1775, 1.57], [1.15, 1.25, 1.325, 1.375, 1.375], 3)
        )

        self.collection_area = {"x_min":-0.6, "x_max":0.6, "y_min":0.9, "y_max":1.1, "object":"fruits"}
        self.weighing_area = {"x_min":-0.5, "x_max":0.5, "y_min":0.5, "y_max":0.7, "object":"scales",}
        self.accepted_fruit_pos = {"distance":1.10, "angle":-75*(numpy.pi/180)}
        self.rejected_fruit_pos = {"distance":1.10, "angle":75*(numpy.pi/180)}
        self.fruit_right_side_pos = {"distance":0.8, "angle":5*(numpy.pi/360)}
        self.fruit_left_side_pos = {"distance":0.8, "angle":-5*(numpy.pi/360)}

        self.fruit_tested = False
        self.fruit_correctly_accepted = False
        self.fruit_correctly_rejected = False

        self.gripper_max = 0.085

        self.iteration = 0
        self.change_reward_iterations = []

        self.cbgroup_server=MutuallyExclusiveCallbackGroup()
        self.cbgroup_client=MutuallyExclusiveCallbackGroup()

        self.load_client=ServiceClient(LoadConfig, 'commander/load_experiment')

    def random_position(self, area):
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

    def generate_fruits(self, n_fruits):
        self.get_logger().info("Generating fruits...")
        for _ in range(n_fruits):
            distance, angle = self.random_position(self.collection_area)
            dim_max = self.rng.uniform(low=0.03, high=0.1)
            fruit = dict(distance = distance, angle = angle, dim_max = dim_max, state = 0)
            self.fruits.append(fruit)


    def perceive_closest_fruit(self):
        if self.fruits:
            distances = numpy.array([fruit["distance"] for fruit in self.fruits])
            closest_fruit_index = numpy.argmin(distances)
            self.closest_fruit = self.fruits[closest_fruit_index]
            self.perceptions["fruits"].data[0].distance = self.closest_fruit["distance"]
            self.perceptions["fruits"].data[0].angle = self.closest_fruit["angle"]
            self.perceptions["fruits"].data[0].dim_max = self.closest_fruit["dim_max"]
            self.perceptions["fruits"].data[0].state = self.closest_fruit["state"]
        else:
            self.perceptions["fruits"].data[0].distance = 1.9
            self.perceptions["fruits"].data[0].angle = 1.4
            self.perceptions["fruits"].data[0].dim_max = 0.1
            self.perceptions["fruits"].data[0].state = 0

    def random_perceptions(self):
        self.catched_fruit = None
        # Generate fruits
        n_fruits = self.rng.integers(0,4)
        self.perceptions["fruits"].data = []
        self.perceptions["fruits"].data.append(self.base_messages["fruits"]())

        self.generate_fruits(n_fruits)
        self.perceive_closest_fruit()

        # Generate scale
        self.perceptions["scales"].data = []
        self.perceptions["scales"].data.append(self.base_messages["scales"]())
        distance, angle = self.random_position(self.weighing_area)
        self.perceptions["scales"].data[0].distance = distance
        self.perceptions["scales"].data[0].angle = angle

        self.perceptions["fruit_in_right_hand"].data = False
        self.perceptions["fruit_in_left_hand"].data = False

        self.perceptions["button_light"].data = False if self.rng.uniform() > 0.5 else True

        self.update_reward_sensor()
    
    def pick_fruit_policy(self):
        if not self.catched_fruit and self.fruits:
            if self.closest_fruit["angle"] > 0.0:
                self.perceptions["fruit_in_right_hand"].data = True
            else:
                self.perceptions["fruit_in_left_hand"].data = True

            #self.perceive_closest_fruit() #TODO Necesario?
            self.catched_fruit = self.closest_fruit
    
    def change_hands_policy(self):
        if self.catched_fruit:
            if self.catched_fruit["dim_max"] <= self.gripper_max:
                if self.perceptions["fruit_in_left_hand"].data:
                    self.perceptions["fruit_in_left_hand"].data = False
                    self.perceptions["fruit_in_right_hand"].data = True
                    self.catched_fruit["angle"] = -self.catched_fruit["angle"]
                elif self.perceptions["fruit_in_right_hand"].data:
                    self.perceptions["fruit_in_left_hand"].data = True
                    self.perceptions["fruit_in_right_hand"].data = False
                    self.catched_fruit["angle"] = -self.catched_fruit["angle"]
    
    def place_fruit_policy(self):
        if self.catched_fruit:
            if self.perceptions["fruit_in_left_hand"].data:
                self.catched_fruit["distance"] = self.fruit_right_side_pos["distance"]
                self.catched_fruit["angle"] = self.fruit_right_side_pos["angle"]
                self.perceptions["fruit_in_left_hand"].data = False
                self.catched_fruit = None

            elif self.perceptions["fruit_in_right_hand"].data:
                self.catched_fruit["distance"] = self.fruit_left_side_pos["distance"]
                self.catched_fruit["angle"] = self.fruit_left_side_pos["angle"]
                self.perceptions["fruit_in_right_hand"].data = False
                self.catched_fruit = None

    
    def test_fruit_policy(self):
        if self.catched_fruit:
            scale = self.perceptions["scales"].data[0]
            if (self.perceptions["fruit_in_left_hand"].data and scale.angle <= 0.0) or (
                self.perceptions["fruit_in_right_hand"].data and scale.angle > 0.0):
                self.catched_fruit["distance"] = scale.distance
                self.catched_fruit["angle"] = scale.angle
                self.catched_fruit["state"] = 1 if self.rng.uniform() > 0.5 else 2
                self.perceptions["fruit_in_left_hand"].data = False
                self.perceptions["fruit_in_right_hand"].data = False
                self.catched_fruit = None
                self.fruit_tested = True

    def ask_nicely_policy(self):
        if not self.fruits:
            n_fruits = self.rng.integers(1,4)
            self.generate_fruits(n_fruits)

    def accept_fruit_policy(self):
        if self.catched_fruit:
            if self.perceptions["fruit_in_left_hand"].data:
                self.catched_fruit["distance"] = self.accepted_fruit_pos["distance"]
                self.catched_fruit["angle"] = self.accepted_fruit_pos["angle"]
                self.perceptions["fruit_in_left_hand"].data = False
                if self.catched_fruit["state"] == 1:
                    self.fruit_correctly_accepted = True
                self.fruits.remove(self.catched_fruit)
                self.catched_fruit = None
    
    def discard_fruit_policy(self):
        if self.catched_fruit:
            if self.perceptions["fruit_in_right_hand"].data:
                self.catched_fruit["distance"] = self.rejected_fruit_pos["distance"]
                self.catched_fruit["angle"] = self.rejected_fruit_pos["angle"]
                self.perceptions["fruit_in_right_hand"].data = False
                if self.catched_fruit["state"] == 2:
                    self.fruit_correctly_rejected = True
                self.fruits.remove(self.catched_fruit)
                self.catched_fruit = None
    
    def press_button_policy(self):
        if self.perceptions["button_light"].data:
            self.perceptions["button_light"].data = False
        else:
            self.perceptions["button_light"].data = True

    def reward_iteration_dependent_goal(self):
        reward = 0.0
        if self.iteration <= self.change_reward_iterations[0]:
            self.get_logger().info("STAGE 1 REWARD: TEST FRUIT")
            if self.fruit_tested:
                reward = 1.0
        else:
            self.get_logger().info("STAGE 2 REWARD: CLASSIFY FRUIT")
            if self.fruit_correctly_accepted or self.fruit_correctly_rejected:
                reward = 1.0
        self.perceptions["iteration_dependent_goal"].data = reward

    def reward_test_fruit_goal(self):
        reward = 0.0
        if self.iteration <= self.change_reward_iterations[0]:
            self.get_logger().info("STAGE 1 REWARD: TEST FRUIT")
            if self.fruit_tested:
                reward = 1.0
        self.perceptions["test_fruit_goal"].data = reward

    def reward_classify_fruit_goal(self):
        reward = 0.0
        if self.iteration > self.change_reward_iterations[0]:
            self.get_logger().info("STAGE 2 REWARD: CLASSIFY FRUIT")
            if self.fruit_correctly_accepted or self.fruit_correctly_rejected:
                reward = 1.0
        self.perceptions["classify_fruit_goal"].data = reward

    def reset_world(self, data):
        self.get_logger().info("Resetting world...")
        self.fruit_tested = False
        self.fruit_correctly_accepted = False
        self.fruit_correctly_rejected = False
        if self.iteration <= self.change_reward_iterations[1]:
            self.fruits = []
            self.random_perceptions()
        self.publish_perceptions()
        if (not self.catched_fruit) and (
            self.perceptions["fruit_in_left_hand"].data
            or self.perceptions["fruit_in_right_hand"].data
        ):
            self.get_logger().error("Critical error: catched_object is empty and it should not!!!")

    def update_reward_sensor(self):
        """Update goal sensors' values."""
        for sensor in self.perceptions:
            reward_method = getattr(self, "reward_" + sensor, None)
            if callable(reward_method):
                reward_method()
    
    def publish_perceptions(self):
        for ident, publisher in self.sim_publishers.items():
            self.get_logger().debug("Publishing " + ident + " = " + str(self.perceptions[ident].data))
            publisher.publish(self.perceptions[ident])

    def world_reset_service_callback(self, request, response):
        self.reset_world(request)
        response.success=True
        return response

    def new_command_callback(self, data):
        """
        Process a command received

        :param data: The message that contais the command received
        :type data: ROS msg defined in setup_control_channel
        """
        self.get_logger().debug(f"Command received... ITERATION: {data.iteration}")
        self.iteration = data.iteration
        if data.command == "reset_world":
            self.reset_world(data)
        elif data.command == "end":
            self.get_logger().info("Ending simulator as requested by LTM...")
            rclpy.shutdown()

    def new_action_service_callback(self, request, response):
        """
        Execute a policy and publish new perceptions.

        :param request: The message that contains the policy to execute
        :type request: ROS srv defined in setup_control_channel
        :param response: Response of the success of the execution of the action
        :type response: ROS srv defined in setup_control_channel
        """
        #self.get_logger().info("Executing policy " + str(request.policy))
        self.get_logger().info(f"ITERATION: {self.iteration}")
        signo = (self.perceptions["scales"].data[0].angle * self.perceptions["fruits"].data[0].angle) > 0
        self.get_logger().info(f"OPA RACING: {signo}")
        self.perceive_closest_fruit() # TODO Necesario antes y despues?
        self.get_logger().info(f"FRUITS BEFORE POLICY: {self.fruits}")
        self.get_logger().info(f"CATCHED FRUIT BEFORE: {self.catched_fruit}")
        self.get_logger().info(f"PERCEPTIONS BEFORE: {self.perceptions}")
        self.get_logger().info(f"POLICY TO EXECUTE: {request.policy}")
        getattr(self, request.policy + "_policy")()
        self.perceive_closest_fruit()
        self.get_logger().info(f"FRUITS AFTER POLICY: {self.fruits}")
        self.get_logger().info(f"CATCHED FRUIT AFTER: {self.catched_fruit}")
        self.get_logger().info(f"PERCEPTIONS AFTER: {self.perceptions}")
        self.update_reward_sensor()
        self.publish_perceptions()
        if (not self.catched_fruit) and (
            self.perceptions["fruit_in_left_hand"].data
            or self.perceptions["fruit_in_right_hand"].data
        ):
            self.get_logger().error("Critical error: catched_object is empty and it should not!!!")
            rclpy.shutdown()
        response.success = True
        return response

    def setup_experiment_stages(self, stages):
        for stage in stages:
            self.change_reward_iterations.append(stages[stage])

    def setup_perceptions(self, perceptions):
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

        :param simulation: The params from the config file to setup the control channel
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
        loaded = self.load_client.send_request(file = self.config_file)
        return loaded

    def load_configuration(self):
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
    sim = IJCNNSim()
    sim.load_configuration()

    try:
        rclpy.spin(sim)
    except KeyboardInterrupt:
        print('Keyboard Interrupt Detected: Shutting down simulator...')
    finally:
        sim.destroy_node()

if __name__ == '__main__':
    main()       

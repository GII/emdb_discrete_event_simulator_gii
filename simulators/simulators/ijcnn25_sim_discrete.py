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
        self.sim_publishers = None
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
        self.accepted_fruit_pos = {"r":1.10, "ang":-60*(numpy.pi/360)}
        self.rejected_fruit_pos = {"r":1.10, "ang":60*(numpy.pi/360)}

        self.cbgroup_server=MutuallyExclusiveCallbackGroup()
        self.cbgroup_client=MutuallyExclusiveCallbackGroup()

        self.load_client=ServiceClient(LoadConfig, 'commander/load_experiment')

    def random_position(self, area):
        valid = False
        while not valid:
            x = self.rng.uniform(low=area["x_min"], high=area["x_max"])
            y = self.rng.uniform(low=area["y_min"], high=area["y_max"])

            dist = numpy.linalg.norm(x, y)
            ang = numpy.arctan(x/y)

            for object in self.perceptions[area["object"]].data:
                if abs(object.distance - dist < 0.1) and abs(object.angle - ang < 0.045):
                    valid = False
                    break
    
    def random_perceptions(self):
        self.catched_fruit = None

        # Generate fruits
        n_fruits = self.rng.integers(0,4)
        self.perceptions["fruits"].data = []
        for i in n_fruits:
            distance, angle = self.random_position(self.collection_area)
            self.perceptions["fruits"].data.append(self.base_messages["fruits"]())
            self.perceptions["fruits"].data[i].distance = distance
            self.perceptions["fruits"].data[i].angle = angle
            if self.rng.uniform() > 0.5:
                self.perceptions["fruits"].data[i].dim_max = 0.3
            else:
                self.perceptions["fruits"].data[i].dim_max = 1.0
            
            self.perceptions["fruits"].data[i].state = 0

        # Generate scale
        self.perceptions["scales"].data = []
        distance, angle = self.random_position(self.weighing_area)
        self.perceptions["scales"].data.append(self.base_messages["scales"]())
        self.perceptions["scales"].data[0].distance = distance
        self.perceptions["scales"].data[0].angle = angle

        self.perceptions["fruit_in_right_hand"].data = False
        self.perceptions["fruit_in_left_hand"].data = False
    
    def pick_fruit_policy(self):
        raise NotImplementedError
    
    def change_hands_policy(self):
        raise NotImplementedError
    
    def place_fruit_policy(self):
        raise NotImplementedError
    
    def test_fruit_policy(self):
        raise NotImplementedError
    
    def accept_fruit_policy(self):
        raise NotImplementedError
    
    def discard_fruit_policy(self):
        raise NotImplementedError
    
    def ask_nicely_policy(self):
        raise NotImplementedError
    
    def press_button_policy(self):
        raise NotImplementedError

    def reset_world(self):
        self.get_logger().info("Resetting world...")
        self.random_perceptions()
        self.publish_perceptions()
    
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
        self.get_logger().info("Executing policy " + str(request.policy))
        getattr(self, request.policy + "_policy")()
        # self.update_reward_sensor()
        self.publish_perceptions()
        # if (not self.catched_object) and (
        #     self.perceptions["ball_in_left_hand"].data
        #     or self.perceptions["ball_in_right_hand"].data
        # ):
        #     self.get_logger().error("Critical error: catched_object is empty and it should not!!!")
        #     rclpy.shutdown()
        response.success = True
        return response

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
                self.setup_perceptions(config["IJCNNExperiment"]["Perceptions"])
                # Be ware, we can not subscribe to control channel before creating all sensor publishers.
                self.setup_control_channel(config["Control"])
        if self.random_seed:
            self.rng = numpy.random.default_rng(self.random_seed)
            self.get_logger().info(f"Setting random number generator with seed {self.random_seed}")
        else:
            self.rng = numpy.random.default_rng()
        
        self.load_experiment_file_in_commander()

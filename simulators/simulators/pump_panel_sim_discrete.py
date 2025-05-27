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
from core.utils import class_from_classname, EncodableDecodableEnum, actuation_msg_to_dict

class PumpObjects(EncodableDecodableEnum):
    """
    This class defines the objects that can be found in the pump panel.
    """
    DISCHARGE_LIGHT = 0
    EMERGENCY_BUTTON = 1
    MODE_SELECTOR = 2
    OFF_BUTTON = 3
    ON_BUTTON = 4
    OUTPUT_FLOW_DIAL = 5
    START_BUTTON = 6
    SYSTEM_BACKUP_LIGHT = 7
    TEST_LIGHT = 8
    TOOL_1 = 9
    TOOL_2 = 10
    V1_BUTTON = 11
    V2_BUTTON = 12
    V3_BUTTON = 13
    VOLTAGE_DIAL = 14

class PumpPanelSim(Node):
    """
    PumpPanelSim simulator class.
    """
    def __init__(self):
        """
        Constructor of the PumpPanelSim simulator class.
        Initializes the simulator with parameters, publishers, and perception messages.
        """
        super().__init__("PumpPanelSim")
        self.rng = None
        self.ident = None
        self.base_messages = {}
        self.perceptions = {}
        self.sim_publishers = {}

        self.random_seed = self.declare_parameter('random_seed', value = 0).get_parameter_value().integer_value
        self.config_file = self.declare_parameter('config_file', descriptor=ParameterDescriptor(dynamic_typing=True)).get_parameter_value().string_value

        self.iteration = 0

        self.cbgroup_server=MutuallyExclusiveCallbackGroup()
        self.cbgroup_client=MutuallyExclusiveCallbackGroup()

        self.load_client=ServiceClient(LoadConfig, 'commander/load_experiment')

        self.pump_started = False
        self.caught_tool = False
        self.panel_on = False
        self.emergency = False

    def simulate_values(self):
        """
        Simulate the valuas of some indicators of the pump panel.
        """
        if self.panel_on:
            # Voltage
            distortion = self.rng.uniform(-2.5, 2.5)
            voltage_value = 230.0 + distortion
            
            # Output flow
            prob = self.rng.random()
            if prob < 0.5:
                output_flow_value = 0.0
                discharge = False
            elif prob < 0.8:
                output_flow_value = self.rng.uniform(1.0, 5.0)
                discharge = True
            else:
                output_flow_value = self.rng.uniform(6.0, 12.0)
                discharge = True

            # Lights
            test_light = self.rng.random() < 0.9
            system_backup_light = self.rng.random() < 0.9
        else:
            voltage_value = 0.0
            output_flow_value = 0.0
            discharge = False
            test_light = False
            system_backup_light = False
        
        obj_perc = self.perceptions["panel_objects"].data[0]
        obj_perc.voltage_dial = voltage_value
        obj_perc.output_flow_dial = output_flow_value
        obj_perc.discharge_light = discharge
        obj_perc.test_light = test_light
        obj_perc.system_backup_light = system_backup_light

    def initial_perceptions(self):
        """
        Initialize the perceptions of the pump panel simulator.
        The initial state is with all indicators off, all valves closed and the tools stored.
        """
        self.perceptions['panel_objects'].data = []
        self.perceptions['panel_objects'].data.append(self.base_messages['panel_objects']())
        obj_perc = self.perceptions["panel_objects"].data[0]
        obj_perc.discharge_light = False
        obj_perc.emergency_button = False
        obj_perc.mode_selector = True
        obj_perc.off_button = False
        obj_perc.on_button = False
        obj_perc.output_flow_dial = 0.0
        obj_perc.start_button = 0
        obj_perc.system_backup_light = False
        obj_perc.test_light = False
        obj_perc.tool_1 = 0
        obj_perc.tool_2 = 0
        obj_perc.v1_button = 0
        obj_perc.v2_button = 0
        obj_perc.v3_button = 0
        obj_perc.voltage_dial = 0.0

        self.v1_status = 1
        self.v2_status = 1
        self.v3_status = 1
        self.start_status = 1

        self.pump_started = False
        self.caught_tool = False
        self.panel_on = False
        self.emergency = False

        self.update_reward_sensor()

    def grasp_object_policy(self, obj = None):
        """
        Parametrized policy to grasp an object.

        :param obj: The object to grasp.
        :type obj: str 
        """
        obj_perc = self.perceptions["panel_objects"].data[0]
        if not self.caught_tool:
            if obj == 'TOOL_1':
                obj_perc.tool_1 = 1
            elif obj == 'TOOL_2':
                obj_perc.tool_2 = 1
            self.caught_tool = True

    def deliver_object_policy(self, obj = None):
        """
        Policy to deliver to the user the object that was caught.
        
        :param obj: The object to deliver. It is not necessary to specify it, 
            as the simulator will always deliver the tool that was caught.
        :type obj: str
        """
        obj_perc = self.perceptions["panel_objects"].data[0]
        if self.caught_tool:
            if obj_perc.tool_1 == 1:
                obj_perc.tool_1 = 2
            elif obj_perc.tool_2 == 1:
                obj_perc.tool_2 = 2
            self.caught_tool = False
    
    def store_object_policy(self, obj = None):
        """
        Policy to store in its respective box the object that was caught.

        :param obj: The object to store. It is not necessary to specify it,
            as the simulator will always store the tool that was caught.
        :type obj: str
        """
        obj_perc = self.perceptions["panel_objects"].data[0]
        if self.caught_tool:
            if obj_perc.tool_1 == 1:
                obj_perc.tool_1 = 0
            elif obj_perc.tool_2 == 1:
                obj_perc.tool_2 = 0
            self.caught_tool = False

    def press_object_policy(self, obj = None):
        """
        Parametrized policy to press an object.

        :param obj: The object to press.
        :type obj: str
        """
        obj_perc = self.perceptions["panel_objects"].data[0]
        if not self.caught_tool:
            if obj == 'EMERGENCY_BUTTON':
                if not self.emergency:
                    self.emergency = True
                    obj_perc.emergency_button = True
                    self.v1_status = 1
                    self.v2_status = 1
                    self.v3_status = 1
                    if self.panel_on:
                        obj_perc.v1_button = self.v1_status
                        obj_perc.v2_button = self.v2_status
                        obj_perc.v3_button = self.v3_status
                else:
                    self.emergency = False
                    obj_perc.emergency_button = False
            elif obj == 'ON_BUTTON':
                if not self.panel_on and not self.emergency:
                    self.panel_on = True
                    self.simulate_values()
                    obj_perc.off_button = True
                    obj_perc.on_button = True
                    obj_perc.start_button = self.start_status
                    obj_perc.v1_button = self.v1_status
                    obj_perc.v2_button = self.v2_status
                    obj_perc.v3_button = self.v3_status
            elif obj == 'OFF_BUTTON':
                if self.panel_on and not self.emergency:
                    self.panel_on = False
                    self.simulate_values()
                    obj_perc.off_button = False
                    obj_perc.on_button = False
                    obj_perc.start_button = 0
                    obj_perc.v1_button = 0
                    obj_perc.v2_button = 0
                    obj_perc.v3_button = 0
            elif obj == 'V1_BUTTON':
                if self.panel_on and not self.emergency:
                    if self.v1_status == 1:
                        self.v1_status = 2
                        obj_perc.v1_button = self.v1_status
                    elif self.v1_status == 2:
                        self.v1_status = 1
                        obj_perc.v1_button = self.v1_status
            elif obj == 'V2_BUTTON':
                if self.panel_on and not self.emergency:
                    if self.v2_status == 1:
                        self.v2_status = 2
                        obj_perc.v2_button = self.v2_status
                    elif self.v2_status == 2:
                        self.v2_status = 1
                        obj_perc.v2_button = self.v2_status
            elif obj == 'V3_BUTTON':
                if self.panel_on and not self.emergency:
                    if self.v3_status == 1:
                        self.v3_status = 2
                        obj_perc.v3_button = self.v3_status
                    elif self.v3_status == 2:
                        self.v3_status = 1
                        obj_perc.v3_button = self.v3_status
            elif obj == 'START_BUTTON':
                if (self.panel_on and not self.emergency) and (self.v1_status == 2 and self.v2_status == 2 and self.v3_status == 2):
                    self.start_status = 2
                    obj_perc.start_button = self.start_status
                    self.pump_started = True

    def reward_start_pump_goal(self):
        """
        Gives a reward of 1.0 if the pump has been started.
        """
        reward = 0.0
        if self.pump_started:
            reward = 1.0
        self.perceptions["start_pump_goal"].data = reward

    def reset_world(self, data):
        """
        Reset the world to the initial state.

        :param data: The message that contains the command to reset the world. It is not used.
        :type data: ROS msg defined in the config file. Typically cognitive_processes_interfaces.msg.ControlMsg or
        cognitive_processes_interfaces.srv.WorldReset.Request
        """
        self.get_logger().info("Resetting world...")
        self.initial_perceptions()
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
        self.simulate_values()
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

        :param request: The message that contains the policy execute and its parameter.
        :type request: ROS srv defined in the config file. Typically cognitive_node_interfaces.srv.PolicyParametrized.Request
        :param response: Response of the success of the execution of the action.
        :type response: ROS srv defined in the config file. Typically cognitive_node_interfaces.srv.PolicyParametrized.Response
        :return: Response indicating the success of the action execution.
        :rtype: ROS srv defined in the config file. Typically cognitive_node_interfaces.srv.PolicyParametrized.Response
        """
        self.get_logger().info("Executing policy " + str(request.policy))
        self.get_logger().info(f"ITERATION: {self.iteration}")
        self.get_logger().info(f"PERCEPTIONS BEFORE: {self.perceptions}")
        self.get_logger().info(f"POLICY TO EXECUTE: {request.policy}")
        param_msg=request.parameter
        param_dict = actuation_msg_to_dict(param_msg)
        self.get_logger().info(f"Params dictionary: {param_dict}")
        param_coded = param_dict.get('policy_params', [{"object":None}])[0]['object']
        if param_coded is not None:
            param = PumpObjects.decode(int(param_coded), normalized=False)
        else:
            param = None
        getattr(self, request.policy + "_policy")(param)
        self.get_logger().info(f"PERCEPTIONS AFTER: {self.perceptions}")
        self.update_reward_sensor()
        self.publish_perceptions()
        response.success = True
        return response

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
                self.setup_perceptions(config["DiscreteEventSimulator"]["Perceptions"])
                # Be ware, we can not subscribe to control channel before creating all sensor publishers.
                self.setup_control_channel(config["Control"])
        
        self.load_experiment_file_in_commander()

def main(args=None):
    rclpy.init(args=args)
    sim = PumpPanelSim()
    sim.load_configuration()

    try:
        rclpy.spin(sim)
    except KeyboardInterrupt:
        print('Keyboard Interrupt Detected: Shutting down simulator...')
    finally:
        sim.destroy_node()

if __name__ == '__main__':
    main()       

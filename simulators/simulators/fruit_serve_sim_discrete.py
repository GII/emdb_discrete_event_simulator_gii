"""Discrete event simulator of the fruit serving task of the TIAGo robot, as a ROS 2 node (``FruitServeSim``)."""

import os

import numpy
import rclpy
import yaml
import yamlloader
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
try:
    # Event-driven executor implemented in C++ (ROS 2 Jazzy and later).
    from rclpy.experimental import EventsExecutor
except ImportError:
    EventsExecutor = None

from core.service_client import ServiceClient
from core.blocking_wait_check import register_executor_thread
from core.utils import class_from_classname, resolve_seed
from core_interfaces.srv import LoadConfig


class FruitServeSim(Node):
    """
    Discrete simulator for the TIAGo fruit serving task, as a ROS 2 node.

    The robot stands between a table and a shelf and faces one of them at a time. The fruits start on the shelf. The robot has to bring a fruit to the table, weigh it on the scale with the left hand, and put it in the delivery box with the right hand. The scale and the box are on the table. The position of the objects is given as a distance and an angle with respect to the robot.

    The world changes through the policies, which are the methods named ``<name>_policy``: ``pick_fruit``, ``place_fruit``, ``place_in_box``, ``change_hands``, ``give_fruit``, ``turn_around``, ``weigh_fruit`` and ``discard_fruit``. Each one returns whether it could be executed.

    The node is named ``FruitServeSim`` and has the parameters:

    - ``random_seed``: seed of the random number generator (0 selects a random seed).
    - ``config_file``: path of the YAML experiment configuration file.

    **Topics and services.** The node creates the following from the configuration file:

    - A subscription to the control topic, whose messages have the iteration and a command. The iteration is stored and the reward sensors are updated. The command ``end`` shuts the node down.
    - A publisher for each perception, on its own topic.
    - A service to execute a policy (``executed_policy_service``). The request contains the name of the policy in ``policy``, which must be one of the policies above, and the response ``success`` is always True. A timer republishes all the perceptions every 0.01 s while this service exists.
    - A service to reset the world (``world_reset_service``).

    It is also a client of the service ``commander/load_experiment``, which it calls once the configuration is loaded to make the commander load the same file.

    **Configuration file.** The node reads these sections of the YAML file:

    - ``DiscreteEventSimulator``/``Stages``: dictionary from the name of each stage to its iteration. It is stored in ``change_reward_iterations``.
    - ``DiscreteEventSimulator``/``Perceptions``: list of perceptions, each one with ``name``, ``perception_topic`` and ``perception_msg`` (the class name of the message). The perceptions ``fruits`` (distance, angle and ``dim_max`` of the closest visible fruit) and ``scales`` (distance, angle, ``state`` and ``active``) are lists of messages. The perceptions ``fruit_in_left_hand``, ``fruit_in_right_hand``, ``holding_weighed_fruit``, ``facing_table`` and ``facing_shelf`` are boolean. All of them are required. Any perception ``<name>`` for which the node has a method ``reward_<name>`` is a reward sensor: ``progress_client_served_goal`` and ``client_served_goal``. A message whose class name contains ``List`` is published as a list of messages, one whose name contains ``Float`` as a float and any other one as a boolean.
    - ``Control``: ``id``, ``control_topic``, ``control_msg`` and, optionally, ``executed_policy_service`` with ``executed_policy_msg`` and ``world_reset_service`` with ``world_reset_msg``.
    """

    def __init__(self):
        """
        Constructor of the FruitServeSim class.

        Declares the parameters, defines the areas and positions of the table, the shelf, the box and the human, and creates the empty state of the world and the client of the commander service. The random number generator is created, and the topics, services and perceptions are set up, by ``load_configuration``.
        """
        super().__init__("FruitServeSim")
        self.rng = None
        self.ident = None
        self.base_messages = {}
        self.perceptions = {}
        self.sim_publishers = {}

        self.random_seed = self.declare_parameter("random_seed", value=0).get_parameter_value().integer_value
        self.config_file = self.declare_parameter(
            "config_file", descriptor=ParameterDescriptor(dynamic_typing=True)
        ).get_parameter_value().string_value

        self.random_table_area = {
            "x_min": -0.35,
            "x_max": 0.35,
            "y_min": 0.50,
            "y_max": 0.80,
            "object": "fruits",
        }
        self.random_shelf_area = {
            "x_min": -0.55,
            "x_max": 0.55,
            "y_min": -0.80,
            "y_max": -0.50,
            "object": "fruits",
        }
        self.weighing_area = {
            "x_min": -0.35,
            "x_max": -0.01,
            "y_min": 0.50,
            "y_max": 0.80,
            "object": "scales",
        }
        self.left_place_pos = {"distance": 0.78, "angle": -0.08}
        self.right_place_pos = {"distance": 0.78, "angle": 0.08}
        self.box_right_pos = {"distance": 0.72, "angle": 0.28}
        self.human_pos = {"distance": 0.55, "angle": 0.0}
        self.gripper_max = 0.085

        self.iteration = 0
        self.change_reward_iterations = {}

        self.fruit_types = ["apple", "kiwi"]
        self.delivered = {}

        self.fruits = []
        self.held_fruit = None
        self.held_hand = None
        self.closest_fruit = None
        self.scale = None
        self.fruit_placed_in_box = False
        self.facing_table = True
        self.facing_shelf = False

        self.cbgroup_server = MutuallyExclusiveCallbackGroup()
        self.cbgroup_client = MutuallyExclusiveCallbackGroup()

        self.load_client = ServiceClient(LoadConfig, "commander/load_experiment")

    def random_position(self, area):
        """
        Generate a random position within the specified area.

        The position is drawn uniformly inside the area and converted to a distance and an angle. It is drawn again if it is too close (less than 0.1 in distance and 0.09 in angle) to an object of the current perception of the kind of the area.

        :param area: The area where the position should be generated. It must have the keys ``x_min``, ``x_max``, ``y_min`` and ``y_max``, and ``object``, the name of the perception (``fruits`` or ``scales``) whose objects are avoided.
        :type area: dict
        :return: A tuple with the distance and the angle (in radians) of the generated position.
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

    def _make_fruit(self, fruit_type):
        distance, angle = self.random_position(self.random_shelf_area)
        dim_max = self.rng.uniform(low=0.04, high=0.09)
        return {
            "class_name": fruit_type,
            "distance": distance,
            "angle": angle,
            "dim_max": dim_max,
            "location": "shelf",
            "weighed": False,
        }

    def generate_fruits(self, n_fruits, scale=None):
        """
        Generate random fruits on the shelf.

        The fruits are added to the current ones. Each one has a random type among ``fruit_types``, a random position on the shelf and a random size. In addition, if a scale is given, with a probability of 0.5 one more fruit is added in one of these places chosen at random: on the table (in either of the two placing positions), in the box, with the human or on the scale. The fruit on the scale is already weighed.

        :param n_fruits: Number of fruits to generate on the shelf.
        :type n_fruits: int
        :param scale: The scale, a dictionary with its ``distance`` and ``angle``. If None, no additional fruit is generated.
        :type scale: dict
        """
        self.get_logger().info("Generating fruits...")
        for _ in range(n_fruits):
            self.fruits.append(self._make_fruit(self.rng.choice(self.fruit_types)))

        if self.rng.uniform() > 0.5 and scale:
            positions = ['placed_pos_l', 'placed_pos_r', 'box_right_pos', 'human_pos', 'scale_pos']

            dim_max = self.rng.uniform(low=0.03, high=0.1)
            choice = self.rng.choice(positions)
            fruit_type = self.rng.choice(self.fruit_types)

            if choice == 'placed_pos_l':
                pos = self.left_place_pos
                location = "table"
            elif choice == 'placed_pos_r':
                pos = self.right_place_pos
                location = "table"
            elif choice == 'box_right_pos':
                pos = self.box_right_pos
                location = "box"
            elif choice == 'human_pos':
                pos = self.human_pos
                location = "human"
            elif choice == 'scale_pos':
                pos = {"distance": scale["distance"], "angle": scale["angle"]}
                location = "scale"
            else:
                return

            self.fruits.append({
                "class_name": fruit_type,
                "distance": pos["distance"],
                "angle": pos["angle"],
                "dim_max": dim_max,
                "location": location,
                "weighed": location == "scale",
            })

    def get_closest_visible_fruit(self, visible_locations=None):
        """
        Get the closest fruit in the visible locations.

        The fruit held by the robot is always prioritized.

        :param visible_locations: Locations (``location`` of the fruits) where to look for the fruit. If None, ``table`` and ``scale`` are used.
        :type visible_locations: tuple
        :return: The held fruit if there is one, otherwise the closest fruit in the visible locations, or None if there is no such fruit.
        :rtype: dict
        """
        if self.held_fruit is not None:
            return self.held_fruit
        if visible_locations is None:
            visible_locations = ("table", "scale")
        candidates = [f for f in self.fruits if f["location"] in visible_locations]
        if not candidates:
            return None
        candidates.sort(key=lambda f: f["distance"])
        return candidates[0]

    def assign_fruit_to_hand(self, fruit):
        """
        Mark a fruit as being held by one hand, based on its side.

        The right hand is used if the angle of the fruit is positive and the left hand otherwise. The location of the fruit becomes ``hand_left`` or ``hand_right``.

        :param fruit: The fruit to hold.
        :type fruit: dict
        :return: The hand that holds the fruit, ``left`` or ``right``.
        :rtype: str
        """
        hand = "right" if fruit["angle"] > 0.0 else "left"
        fruit["location"] = f"hand_{hand}"
        self.held_fruit = fruit
        self.held_hand = hand
        return hand

    def clear_held_fruit(self):
        """
        Release the held fruit.

        The robot no longer holds any fruit. The location of the fruit is not changed.
        """
        self.held_fruit = None
        self.held_hand = None

    # ── Perception updates ────────────────────────────────────────────────

    def perceive_closest_fruit(self):
        """
        Update the perception of the closest visible fruit.

        The visibility depends on where the robot is facing: the fruits on the table and on the scale when facing the table, and the fruits on the shelf when facing the shelf. The fruit in a hand is always visible. The perception contains a single message with the distance, angle and ``dim_max`` of the fruit. If no fruit is visible, the message contains the values 1.9, 1.4 and 0.1.
        """
        message = self.perceptions["fruits"]
        message.data = []
        fruit_msg = self.base_messages["fruits"]()

        # Determine which fruit locations are visible
        if self.facing_table:
            visible_locations = ("table", "scale", "hand_left", "hand_right")
        else:
            visible_locations = ("shelf", "hand_left", "hand_right")

        fruit = self.get_closest_visible_fruit(visible_locations=visible_locations)
        if fruit is not None:
            fruit_msg.distance = fruit["distance"]
            fruit_msg.angle = fruit["angle"]
            fruit_msg.dim_max = fruit["dim_max"]
            self.closest_fruit = fruit
        else:
            fruit_msg.distance = 1.9
            fruit_msg.angle = 1.4
            fruit_msg.dim_max = 0.1
            self.closest_fruit = None
        message.data.append(fruit_msg)

    def update_hand_perceptions(self):
        """
        Update the perceptions of the hands.

        ``fruit_in_left_hand`` and ``fruit_in_right_hand`` tell whether a fruit is held with that hand.
        """
        self.perceptions["fruit_in_left_hand"].data = self.held_hand == "left"
        self.perceptions["fruit_in_right_hand"].data = self.held_hand == "right"

    def update_holding_weighed_fruit_perception(self):
        """
        Update the perception of holding a weighed fruit.

        It is True if the robot holds a fruit that has been weighed.
        """
        # if self.iteration <= self.change_reward_iterations.get('stage1', 0):
        #     self.perceptions["holding_weighed_fruit"].data = False
        #     return
        self.perceptions["holding_weighed_fruit"].data = (
            self.held_fruit is not None and self.held_fruit.get("weighed", False)
        )

    def update_facing_table_perception(self):
        """Update the perception of facing the table."""
        self.perceptions["facing_table"].data = self.facing_table

    def update_facing_shelf_perception(self):
        """Update the perception of facing the shelf."""
        self.perceptions["facing_shelf"].data = self.facing_shelf

    def update_scale_perception(self):
        """
        Update the perception of the scale.

        The perception contains a single message with the ``state`` and ``active`` values of the scale and, when the robot faces the table, its distance and angle. When it faces the shelf, the scale is not visible and the distance and angle are 1.9 and 1.4.
        """
        message = self.perceptions["scales"]
        message.data = []
        scale_msg = self.base_messages["scales"]()
        # before_weigh_stage = self.iteration <= self.change_reward_iterations.get('stage1', 0)
        if self.facing_table:
            scale_msg.distance = self.scale["distance"]
            scale_msg.angle = self.scale["angle"]
        else:
            scale_msg.distance = 1.9
            scale_msg.angle = 1.4

        # if before_weigh_stage:
        #     scale_msg.state = 0
        #     scale_msg.active = False
        # else:
        scale_msg.state = self.scale["state"]
        scale_msg.active = self.scale["active"]

        message.data.append(scale_msg)

    def update_simulated_perceptions(self):
        """
        Refresh all the simulated sensor outputs from the current internal state.

        This updates the closest fruit, the hands, the weighed fruit, the facing direction, the scale and the reward sensors.
        """
        self.perceive_closest_fruit()
        self.update_hand_perceptions()
        self.update_holding_weighed_fruit_perception()
        self.update_facing_table_perception()
        self.update_facing_shelf_perception()
        self.update_scale_perception()
        self.update_reward_sensor()

    # ── World reset ───────────────────────────────────────────────────────

    def reset_world(self, _data=None):
        """
        Reset the simulated world to a new randomized state.

        The robot releases any fruit, and the delivered fruits and the box are cleared. A scale is placed at a random position of the weighing area, inactive. Between 2 and 4 fruits are generated on the shelf (see ``generate_fruits``). The robot faces the table or the shelf with equal probability. With a probability of 0.5 the robot starts holding the first fruit, and with a probability of 0.5 that fruit is already weighed. Finally, the perceptions are updated and published.

        :param _data: Request of the reset service. It is not used.
        :type _data: object
        """
        self.get_logger().info("Resetting fruit serve world...")
        self.clear_held_fruit()
        self.delivered = {}
        self.fruit_placed_in_box = False

        # Generate scale at a random position in the weighing area
        self.perceptions["scales"].data = []
        self.perceptions["scales"].data.append(self.base_messages["scales"]())
        distance, angle = self.random_position(self.weighing_area)
        self.scale = {
            "distance": distance,
            "angle": angle,
            "state": 0,
            "active": False,
        }

        # Generate fruits on the shelf
        self.fruits = []
        self.perceptions["fruits"].data = []
        self.perceptions["fruits"].data.append(self.base_messages["fruits"]())
        min_fruits = len(self.fruit_types)
        n_fruits = self.rng.integers(min_fruits, min_fruits + 3)
        self.generate_fruits(n_fruits, scale=self.scale)

        # 50% chance start facing shelf, 50% facing table (mutually exclusive)
        self.facing_table = self.rng.uniform() > 0.5
        self.facing_shelf = not self.facing_table

        # 50% chance the robot starts holding a fruit (hand based on angle, like fruit_shop)
        if self.rng.uniform() > 0.5 and self.fruits:
            fruit = self.fruits[0]  # pick first shelf fruit
            if fruit["angle"] > 0.0:
                hand = "right"
            else:
                hand = "left"
            fruit["location"] = f"hand_{hand}"
            # Randomly start holding an already-weighed fruit so the
            # holding_weighed_fruit precondition gets positive samples.
            # Only from stage2 onward — the weighed perception and scale
            # are masked during stage0 (exploration) and stage1 (place-on-table).
            # if self.iteration > self.change_reward_iterations.get('stage1', 0) and 
            if self.rng.uniform() > 0.5:
                fruit["weighed"] = True
            self.held_fruit = fruit
            self.held_hand = hand

        self.update_simulated_perceptions()
        self.publish_perceptions()

    # ── Infrastructure callbacks ──────────────────────────────────────────

    def publish_perceptions(self):
        """Publish the current simulated perceptions."""
        for ident, publisher in self.sim_publishers.items():
            publisher.publish(self.perceptions[ident])

    def world_reset_service_callback(self, request, response):
        """
        Callback for the world reset service.

        :param request: Request of the service.
        :type request: object
        :param response: Response of the service. Its field ``success`` is set to True.
        :type response: object
        :return: The response.
        :rtype: object
        """
        self.reset_world(request)
        response.success = True
        return response

    def new_command_callback(self, data):
        """
        Process a command received in the control topic.

        The iteration of the message is stored and the reward sensors are updated. If the command is ``end``, the node shuts down.

        :param data: Control message, with the fields ``iteration`` and ``command``.
        :type data: object
        """
        self.get_logger().debug(f"Command received... ITERATION: {data.iteration}")
        self.iteration = data.iteration
        self.update_reward_sensor()
        # World resets arrive through the world reset service, not through this topic
        if data.command == "end":
            self.get_logger().info("Ending simulator as requested by LTM...")
            rclpy.shutdown()

    def new_action_service_callback(self, request, response):
        """
        Execute a policy and publish the new perceptions.

        The policy ``<request.policy>_policy`` is executed, and then all the perceptions are updated and published.

        :param request: Request of the service, with the name of the policy in ``policy``.
        :type request: object
        :param response: Response of the service. Its field ``success`` is set to True.
        :type response: object
        :return: The response.
        :rtype: object
        """
        self.get_logger().info("Executing policy " + str(request.policy))
        self.get_logger().info(f"ITERATION: {self.iteration}")
        policy = getattr(self, request.policy + "_policy")
        policy()
        self.update_simulated_perceptions()
        self.publish_perceptions()
        response.success = True
        return response

    # ── Policies ──────────────────────────────────────────────────────────

    def pick_fruit_policy(self):
        """
        Pick the closest visible fruit.

        It requires the robot not to hold any fruit and to see a fruit. When facing the table, a fruit on the scale is always picked first, and otherwise the closest fruit on the table. When facing the shelf, the closest fruit on the shelf is picked. The hand is chosen by the side of the fruit (see ``assign_fruit_to_hand``). A fruit picked from the scale deactivates the scale and is considered weighed.

        :return: True if a fruit was picked, False otherwise.
        :rtype: bool
        """
        if self.held_fruit is not None:
            self.get_logger().warn("Cannot pick a fruit while already holding one.")
            return False

        # When facing the table, always pick the fruit on the scale first if one is there.
        # Fall back to the closest table/scale fruit otherwise.
        if self.facing_table:
            fruit = next((f for f in self.fruits if f["location"] == "scale"), None)
            if fruit is None:
                fruit = self.get_closest_visible_fruit(visible_locations=("table", "scale"))
        else:
            fruit = self.get_closest_visible_fruit(visible_locations=("shelf",))

        if fruit is None:
            self.get_logger().warn("No fruit visible from current side.")
            return False

        from_location = fruit["location"]
        hand = self.assign_fruit_to_hand(fruit)
        # If picking from scale, deactivate it
        if from_location == "scale":
            self.scale["active"] = False
            self.held_fruit["weighed"] = True  # once picked up, the fruit is considered weighed
        self.fruit_placed_in_box = False
        self.get_logger().info(f"Picked {fruit['class_name']} from {from_location} with {hand} hand.")
        return True

    def place_fruit_policy(self):
        """
        Place the held fruit.

        It requires a fruit in the hand. The fruit is left at the placing position on the side opposite to the hand that holds it (the right placing position for a fruit in the left hand and the left one for a fruit in the right hand). It is left on the table if the robot faces the table and on the shelf otherwise, and it is no longer weighed.

        :return: True if the fruit was placed, False otherwise.
        :rtype: bool
        """
        if self.held_fruit is None:
            self.get_logger().warn("No fruit in hand to place on the table.")
            return False

        
        if self.held_hand == "left":
            self.held_fruit["distance"] = self.right_place_pos["distance"]
            self.held_fruit["angle"] = self.right_place_pos["angle"]
        else:
            self.held_fruit["distance"] = self.left_place_pos["distance"]
            self.held_fruit["angle"] = self.left_place_pos["angle"]
        if self.facing_table:
            self.held_fruit["location"] = "table"
        else:
            self.held_fruit["location"] = "shelf"
        self.held_fruit["weighed"] = False
        self.clear_held_fruit()
        return True

    def place_in_box_policy(self):
        """
        Place the held fruit in the delivery box.

        It requires a weighed fruit held with the right hand while facing the table. The fruit is left in the box and the task of serving the client is completed.

        :return: True if the fruit was placed in the box, False otherwise.
        :rtype: bool
        """
        if self.held_fruit is None:
            self.get_logger().warn("No fruit in hand to place in the box.")
            return False

        if not self.held_fruit.get("weighed", False):
            self.get_logger().warn("Fruit has not been weighed; cannot place in box.")
            return False

        if not self.facing_table:
            self.get_logger().warn("Must face the table to place in box.")
            return False

        # Check hand matches box side
        if (self.held_hand != "right"):
            self.get_logger().warn("Fruit must be in the hand closest to the box to place in box.")
            return False

        fruit_type = self.held_fruit["class_name"]

        self.held_fruit["distance"] = self.box_right_pos["distance"]
        self.held_fruit["angle"] = self.box_right_pos["angle"]
        self.held_fruit["location"] = "box"
        self.held_fruit["weighed"] = False

        self.fruit_placed_in_box = True
        self.clear_held_fruit()
        self.get_logger().info(
            f"Placed {fruit_type} in box. Delivered: {self.delivered}"
        )
        return True

    def change_hands_policy(self):
        """
        Pass the held fruit to the other hand.

        It requires a fruit in the hand whose size (``dim_max``) does not exceed the maximum opening of the gripper (``gripper_max``).

        :return: True if the fruit changed hands, False otherwise.
        :rtype: bool
        """
        if self.held_fruit is None:
            self.get_logger().warn("No fruit in hand to change hands.")
            return False
        if self.held_fruit["dim_max"] > self.gripper_max:
            self.get_logger().warn("Fruit is too large to change hands.")
            return False

        new_hand = "left" if self.held_hand == "right" else "right"
        self.held_hand = new_hand
        self.held_fruit["location"] = f"hand_{new_hand}"
        self.held_fruit["angle"] = -self.held_fruit["angle"]
        return True

    def give_fruit_policy(self):
        """
        Give the held fruit to the human.

        It requires a fruit in the hand.

        :return: True if the fruit was given, False otherwise.
        :rtype: bool
        """
        if self.held_fruit is None:
            self.get_logger().warn("No fruit in hand to give.")
            return False

        self.held_fruit["distance"] = self.human_pos["distance"]
        self.held_fruit["angle"] = self.human_pos["angle"]
        self.held_fruit["location"] = "human"
        self.clear_held_fruit()
        return True

    def turn_around_policy(self):
        """
        Turn 180 degrees to face the other side (table or shelf).

        Keeps ``facing_table`` and ``facing_shelf`` mutually exclusive and in sync.

        :return: True.
        :rtype: bool
        """
        self.facing_table = not self.facing_table
        self.facing_shelf = not self.facing_shelf
        side = "table" if self.facing_table else "shelf"
        self.get_logger().info(f"Turned around. Now facing {side}.")
        return True

    def weigh_fruit_policy(self):
        """
        Weigh the held fruit on the scale.

        It requires a fruit that has not been weighed, held with the left hand while facing the table. The fruit is left on the scale, which becomes active. The fruit is considered weighed when it is picked up again from the scale.

        :return: True if the fruit was left on the scale, False otherwise.
        :rtype: bool
        """
        if self.held_fruit is None:
            self.get_logger().warn("No fruit in hand to weigh.")
            return False

        if self.held_fruit.get("weighed", False):
            self.get_logger().warn("Fruit is already weighed; cannot weigh again.")
            return False

        if not self.facing_table:
            self.get_logger().warn("Must face the table to use the scale.")
            return False

        # Check hand matches scale side
        if (self.held_hand != "left"):
            self.get_logger().warn("Fruit must be in the hand closest to the scale to weigh.")
            return False

        self.held_fruit["distance"] = self.scale["distance"]
        self.held_fruit["angle"] = self.scale["angle"]
        self.held_fruit["location"] = "scale"
        self.scale["active"] = True
        self.get_logger().info(f"Weighed {self.held_fruit['class_name']}. Left on scale.")
        self.clear_held_fruit()
        return True

    def discard_fruit_policy(self):
        """
        Discard the held fruit.

        It requires a weighed fruit in the hand.

        :return: True if the fruit was discarded, False otherwise.
        :rtype: bool
        """
        if self.held_fruit is None:
            self.get_logger().warn("No fruit in hand to discard.")
            return False

        if not self.held_fruit.get("weighed", False):
            self.get_logger().warn("Fruit has not been weighed; cannot discard.")
            return False

        self.held_fruit["location"] = "discarded"
        self.clear_held_fruit()
        return True

    # ── Rewards ───────────────────────────────────────────────────────────

    def reward_progress_client_served_goal(self):
        """
        Update the perception ``progress_client_served_goal``, a gradual reward toward placing a weighed fruit in the delivery box.

        The sequence of policies is: ``turn_around`` (face the shelf), ``pick_fruit``, ``turn_around`` (face the table), optionally ``change_hands``, ``weigh_fruit``, ``pick_fruit`` (from the scale), optionally ``change_hands`` and ``place_in_box``.

        The value of the first situation that matches is used:

        - 1.0: a weighed fruit has been placed in the box.
        - 0.875: holding a weighed fruit with the right hand, facing the table (ready to place it in the box).
        - 0.75: holding a weighed fruit with the left hand, facing the table (it needs ``change_hands`` for the box).
        - 0.625: holding a weighed fruit facing the shelf, or the scale is active (a fruit has been weighed and is waiting to be picked up).
        - 0.5: holding an unweighed fruit with the left hand, facing the table (ready to weigh).
        - 0.375: holding an unweighed fruit with the right hand, facing the table (it needs ``change_hands`` for the scale).
        - 0.25: holding an unweighed fruit, facing the shelf (it needs ``turn_around``).
        - 0.125: not holding a fruit, facing the shelf with fruits on it (something to pick).
        - 0.0: any other situation.
        """

        progress = 0.0

        if self.fruit_placed_in_box:
            progress = 1.0

        elif self.held_fruit is not None and self.held_fruit.get("weighed", False):
            # Holding a weighed fruit — check hand/direction alignment with box (right side)
            if self.facing_table and self.held_hand == "right":
                progress = 0.875
            elif self.facing_table:
                # Facing table but left hand — needs change_hands before boxing
                progress = 0.75
            else:
                # Weighed fruit but facing shelf — need to turn around
                progress = 0.625

        elif self.scale is not None and self.scale.get("active", False):
            # Fruit has been weighed and left on the scale; robot needs to pick it up
            progress = 0.625

        elif self.held_fruit is not None:
            # Holding an unweighed fruit — check hand/direction alignment with scale (left side)
            if self.facing_table and self.held_hand == "left":
                progress = 0.5
            elif self.facing_table:
                # Facing table but right hand — needs change_hands before weighing
                progress = 0.375
            else:
                # Holding fruit but facing shelf — need to turn around
                progress = 0.25

        elif self.facing_shelf and any(f["location"] == "shelf" for f in self.fruits):
            # Facing shelf with visible fruits — ready to pick
            progress = 0.125

        self.perceptions["progress_client_served_goal"].data = progress

    def reward_client_served_goal(self):
        """
        Update the perception ``client_served_goal``, the terminal goal of the task.

        It is 1.0 when a weighed fruit has been placed in the box and 0.0 otherwise.
        """
        self.perceptions["client_served_goal"].data = 1.0 if self.fruit_placed_in_box else 0.0

    def update_reward_sensor(self):
        """
        Update the reward sensors.

        For each configured perception ``<name>``, the method ``reward_<name>`` is called if it exists.
        """
        for sensor in self.perceptions:
            reward_method = getattr(self, "reward_" + sensor, None)
            if callable(reward_method):
                reward_method()

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup_experiment_stages(self, stages):
        """
        Set up the stages of the experiment with their corresponding iterations.

        :param stages: Dictionary from the name of each stage to its iteration. It is stored in ``change_reward_iterations``.
        :type stages: dict
        """
        for stage in stages:
            self.change_reward_iterations[stage] = stages[stage]

    def setup_perceptions(self, perceptions):
        """
        Configure the ROS topics where the simulator will publish the perceptions.

        For each perception, the message is created with an initial value that depends on its class name, and a publisher is created in its topic.

        :param perceptions: List of dictionaries with the keys ``name``, ``perception_topic`` and ``perception_msg``.
        :type perceptions: list
        """
        for perception in perceptions:
            sid = perception["name"]
            topic = perception["perception_topic"]
            classname = perception["perception_msg"]
            message = class_from_classname(classname)
            self.perceptions[sid] = message()
            if hasattr(self.perceptions[sid], "visible"):
                self.perceptions[sid].visible = False
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
        Configure the ROS topic and the services where the node listens for commands to be executed.

        :param simulation: The ``Control`` section of the configuration file.
        :type simulation: dict
        """
        self.ident = simulation["id"]
        topic = simulation["control_topic"]
        message = class_from_classname(simulation["control_msg"])
        self.get_logger().info("Subscribing to... " + str(topic))
        self.create_subscription(message, topic, self.new_command_callback, 0)

        service_policy = simulation.get("executed_policy_service")
        service_world_reset = simulation.get("world_reset_service")

        if service_policy:
            self.get_logger().info("Creating server... " + str(service_policy))
            message_policy_srv = class_from_classname(simulation["executed_policy_msg"])
            self.create_service(
                message_policy_srv,
                service_policy,
                self.new_action_service_callback,
                callback_group=self.cbgroup_server,
            )
            self.get_logger().info("Creating perception publisher timer... ")
            self.perceptions_timer = self.create_timer(
                0.01, self.publish_perceptions, callback_group=self.cbgroup_server
            )

        if service_world_reset:
            self.message_world_reset = class_from_classname(simulation["world_reset_msg"])
            self.create_service(
                self.message_world_reset,
                service_world_reset,
                self.world_reset_service_callback,
                callback_group=self.cbgroup_server,
            )

    def load_experiment_file_in_commander(self):
        """
        Load the configuration file in the commander node.

        :return: The result of the request to the service ``commander/load_experiment``.
        :rtype: object
        """
        return self.load_client.send_request(file=self.config_file)

    def load_configuration(self):
        """
        Load the configuration file and set up the simulator.

        The random number generator is created with the seed of the parameter. If the configuration file is not an existing file, an error is logged and ROS is shut down. Otherwise, the stages, the perceptions and the control channel are set up from the file, the world is reset and the file is loaded in the commander node.
        """
        self.random_seed = resolve_seed(self.random_seed)
        self.rng = numpy.random.default_rng(self.random_seed)
        self.get_logger().info(f"Setting random number generator with seed {self.random_seed}")

        if self.config_file is None:
            self.get_logger().error("No configuration file for the LTM simulator specified!")
            rclpy.shutdown()
            return

        if not os.path.isfile(self.config_file):
            self.get_logger().error(self.config_file + " does not exist!")
            rclpy.shutdown()
            return

        self.get_logger().info(f"Loading configuration from {self.config_file}...")
        config = yaml.load(
            open(self.config_file, "r", encoding="utf-8"),
            Loader=yamlloader.ordereddict.CLoader,
        )
        self.setup_experiment_stages(config["DiscreteEventSimulator"]["Stages"])
        self.setup_perceptions(config["DiscreteEventSimulator"]["Perceptions"])
        self.setup_control_channel(config["Control"])
        self.reset_world()
        self.load_experiment_file_in_commander()


def main(args=None):
    """
    Run the fruit serving simulator node.

    The node is spun with an events executor if it is available, and with a single-threaded executor otherwise, until the node is shut down or interrupted from the keyboard.

    :param args: Command line arguments for ROS.
    :type args: list
    """
    rclpy.init(args=args)
    sim = FruitServeSim()
    sim.load_configuration()
    # Much cheaper than the Python executors. The simulator runs in a single thread either way, and
    # none of its callbacks waits for anything, so callback groups are not needed.
    executor = EventsExecutor() if EventsExecutor is not None else SingleThreadedExecutor()
    executor.add_node(sim)
    register_executor_thread() # The callbacks of the EventsExecutor are not visible in the Python stack

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Keyboard Interrupt Detected: Shutting down simulator...")
    finally:
        sim.destroy_node()


if __name__ == "__main__":
    main()

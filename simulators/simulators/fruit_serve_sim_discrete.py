import os

import numpy
import rclpy
import yaml
import yamlloader
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

from core.service_client import ServiceClient
from core.utils import class_from_classname, resolve_seed
from core_interfaces.srv import LoadConfig


class FruitServeSim(Node):
    """Discrete simulator for the TIAGo fruit serving task."""

    def __init__(self):
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
        """Generate random fruits on the shelf."""
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
        """Get closest fruit in visible locations. Held fruit is always prioritized."""
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
        """Mark a fruit as being held by one hand based on its side of the table."""
        hand = "right" if fruit["angle"] > 0.0 else "left"
        fruit["location"] = f"hand_{hand}"
        self.held_fruit = fruit
        self.held_hand = hand
        return hand

    def clear_held_fruit(self):
        self.held_fruit = None
        self.held_hand = None

    # ── Perception updates ────────────────────────────────────────────────

    def perceive_closest_fruit(self):
        """Publish closest visible fruit. Visibility depends on facing direction."""
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
        self.perceptions["fruit_in_left_hand"].data = self.held_hand == "left"
        self.perceptions["fruit_in_right_hand"].data = self.held_hand == "right"

    def update_holding_weighed_fruit_perception(self):
        # if self.iteration <= self.change_reward_iterations.get('stage1', 0):
        #     self.perceptions["holding_weighed_fruit"].data = False
        #     return
        self.perceptions["holding_weighed_fruit"].data = (
            self.held_fruit is not None and self.held_fruit.get("weighed", False)
        )

    def update_facing_table_perception(self):
        self.perceptions["facing_table"].data = self.facing_table

    def update_facing_shelf_perception(self):
        self.perceptions["facing_shelf"].data = self.facing_shelf

    def update_scale_perception(self):
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
        """Refresh all simulated sensor outputs from the current internal state."""
        self.perceive_closest_fruit()
        self.update_hand_perceptions()
        self.update_holding_weighed_fruit_perception()
        self.update_facing_table_perception()
        self.update_facing_shelf_perception()
        self.update_scale_perception()
        self.update_reward_sensor()

    # ── World reset ───────────────────────────────────────────────────────

    def reset_world(self, _data=None):
        """Reset the simulated world to a new randomized state."""
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
        self.reset_world(request)
        response.success = True
        return response

    def new_command_callback(self, data):
        self.get_logger().debug(f"Command received... ITERATION: {data.iteration}")
        self.iteration = data.iteration
        self.update_reward_sensor()
        if data.command == "reset_world":
            self.reset_world(data)
        elif data.command == "end":
            self.get_logger().info("Ending simulator as requested by LTM...")
            rclpy.shutdown()

    def new_action_service_callback(self, request, response):
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
        """Pick closest visible fruit, always prioritizing a fruit on the scale."""
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
        """Place held fruit on the table. Must face table."""
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
        """Place fruit in the delivery box. Fruit must be weighed first."""
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
        """Give held fruit to the human."""
        if self.held_fruit is None:
            self.get_logger().warn("No fruit in hand to give.")
            return False

        self.held_fruit["distance"] = self.human_pos["distance"]
        self.held_fruit["angle"] = self.human_pos["angle"]
        self.held_fruit["location"] = "human"
        self.clear_held_fruit()
        return True

    def turn_around_policy(self):
        """Turn 180 degrees to face the other side (table <-> shelf).

        Keeps facing_table and facing_shelf mutually exclusive and in sync.
        """
        self.facing_table = not self.facing_table
        self.facing_shelf = not self.facing_shelf
        side = "table" if self.facing_table else "shelf"
        self.get_logger().info(f"Turned around. Now facing {side}.")
        return True

    def weigh_fruit_policy(self):
        """Weigh the held fruit on the scale. Must face table. Hand must match scale side."""
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
        """Discard the held fruit. Fruit must be weighed first."""
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
        Gradual reward toward placing a weighed fruit in the delivery box.

        sequence:
          turn_around (face shelf) → pick_fruit → turn_around (face table)
          → [change_hands] → weigh_fruit → pick_fruit (from scale)
          → [change_hands] → place_in_box

        Progress levels:
          0.0   - nothing useful happening
          0.125 - fruits available on shelf (something to pick)
          0.25  - holding unweighed fruit, facing shelf (need to turn around)
          0.375 - holding unweighed fruit, facing table, wrong hand for scale (need change_hands)
          0.5   - holding unweighed fruit, facing table, left hand (ready to weigh)
          0.625 - scale active: fruit has been weighed, waiting to be picked up
          0.75  - holding weighed fruit, facing table, left hand (need change_hands for box)
          0.875 - holding weighed fruit, facing table, right hand (ready to place in box)
          1.0   - weighed fruit placed in box
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
        """Terminal mission: 1.0 when a weighed fruit is placed in the box."""
        self.perceptions["client_served_goal"].data = 1.0 if self.fruit_placed_in_box else 0.0

    def update_reward_sensor(self):
        for sensor in self.perceptions:
            reward_method = getattr(self, "reward_" + sensor, None)
            if callable(reward_method):
                reward_method()

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup_experiment_stages(self, stages):
        for stage in stages:
            self.change_reward_iterations[stage] = stages[stage]

    def setup_perceptions(self, perceptions):
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
        return self.load_client.send_request(file=self.config_file)

    def load_configuration(self):
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
    rclpy.init(args=args)
    sim = FruitServeSim()
    sim.load_configuration()

    try:
        rclpy.spin(sim)
    except KeyboardInterrupt:
        print("Keyboard Interrupt Detected: Shutting down simulator...")
    finally:
        sim.destroy_node()


if __name__ == "__main__":
    main()

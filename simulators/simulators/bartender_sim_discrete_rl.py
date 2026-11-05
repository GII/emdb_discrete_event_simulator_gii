"""Discrete event simulator of the bartender experiment with Cartesian positions, clients with preferences and a curriculum: the simulation logic (``BartenderSim``) and its ROS 2 node (``BartenderSimNode``)."""

import os
import math

import numpy
import numpy as np
import yaml
import yamlloader
import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity
from rclpy.executors import SingleThreadedExecutor
try:
    # Event-driven executor implemented in C++ (ROS 2 Jazzy and later).
    from rclpy.experimental import EventsExecutor
except ImportError:
    EventsExecutor = None
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import Float32
from scipy.spatial import distance

from core.container import Container
from core.service_client import ServiceClient
from core.blocking_wait_check import register_executor_thread
from core_interfaces.srv import LoadConfig
from core.utils import class_from_classname, resolve_seed


# Default period (seconds) of the periodic publication of perceptions.
DEFAULT_PERCEPTION_PERIOD = 0.01


# BartenderSim v2. Key fixes vs the original simulator:
# 1. reset_world: if→elif chains; no silent fall-through
# 2. ask_nicely: stable non-trivial per-client preferences (not pref=client_id)
# 3. place_glass at prep: cleans drink (replaces discard_drink_policy)
# 4. place_glass at serv: only marks served if drink matches preference
# 5. pick_glass: can always pick up used glass (for return)
# 6. place_bottle: only repositions the returned bottle, not all bottles
# 7. change_position: instant teleport
# 8. Reward hierarchy: fixed perverse incentive (wrong drink + bottle was > wrong drink)
# 9. n_bottles configurable
# 10. is_in_transit() exposed
class BartenderSim:
    """
    Discrete simulator of the bartender experiment: pure simulation logic, without ROS communication.

    The robot has two hands: the left one holds the glass and the right one holds a bottle. It works at two tables, the preparation table (position id 0) and the serving table (position id 1), and moves between them instantly. The bottles and the glass have Cartesian coordinates x and y, and the preparation area (``prep_area``) and the serving area (``serv_area``) are rectangles.

    A client asks for a drink type and may like it shaken. The preference of the client is not known by the agent until it executes ``ask_nicely_policy``. An episode is finished (``finished_signal``) when the correct drink has been served and the used glass has been cleaned.

    The world changes only through the policies: ``shake_glass``, ``pick_glass``, ``place_glass``, ``prepare_drink``, ``change_position``, ``pick_bottle``, ``place_bottle`` and ``ask_nicely``, which are the methods named ``<name>_policy``.

    **Curriculum.** The initial state of each episode is sampled by ``generate_world`` from the probabilities of the curriculum selected in ``current_curriculum``. The available curricula are ``balanced`` (the default), ``help`` and ``benchmark``. Each one has a section ``serving`` and a section ``preparing``, used when the robot starts at the serving or at the preparation table, with these probabilities:

    - ``bottle_at_serv``: each bottle is placed in the serving area instead of the preparation area.
    - ``grasped_bottle``: the right hand starts holding a bottle.
    - ``correct_bottle``: the bottle held is the one the client prefers.
    - ``grasped_glass``: the left hand starts holding the glass.
    - ``glass_at_serv``: the glass is placed in the serving area instead of the preparation area.
    - ``empty_glass``: the glass starts empty.
    - ``correct_drink``: the drink in a glass that is not empty is the one the client wants.
    - ``used_glass``: the glass starts already used.
    """

    def __init__(self, random_seed=0, n_bottles=3, single_client=True, random_preferences=False):
        """
        Constructor of the BartenderSim simulator class.

        It creates the random number generator, the clients, the areas and positions of the tables, the bottles and the glass. The state of the episode is set by ``reset_world``.

        :param random_seed: Seed of the random number generator. A value of 0 or None selects a random seed.
        :type random_seed: int
        :param n_bottles: Number of bottles. Their drink types are numbered from 1 to ``n_bottles``.
        :type n_bottles: int
        :param single_client: If True, there is only the client 1, who prefers the drink type 2 and likes it shaken. If False, there are three clients.
        :type single_client: bool
        :param random_preferences: Only used when ``single_client`` is False. If True, the preferred drink type and the taste for shaken drinks of the three clients are random (derived from the seed). If False, they are fixed: the client 1 prefers the drink type 1, the client 2 the drink type 2 shaken and the client 3 the drink type 3.
        :type random_preferences: bool
        """
        self.random_seed = resolve_seed(random_seed)
        self.rng = numpy.random.default_rng(self.random_seed)
        self.n_bottles = n_bottles
        self.current_curriculum = "balanced"
        self.curriculum = {
            "balanced": {
                "serving": {
                    "bottle_at_serv": 0.5,
                    "grasped_bottle": 0.667,
                    "correct_bottle": 0.5,
                    "grasped_glass": 0.333,
                    "glass_at_serv": 0.5,
                    "empty_glass": 0.333,
                    "correct_drink": 0.5,
                    "used_glass": 0.5,
                },
                "preparing": {
                    "bottle_at_serv": 0.5,
                    "grasped_bottle": 0.667,
                    "correct_bottle": 0.5,
                    "grasped_glass": 0.333,
                    "glass_at_serv": 0.5,
                    "empty_glass": 0.333,
                    "correct_drink": 0.5,
                    "used_glass": 0.5,
                },
            },
            "help": {
                "serving": {
                    "bottle_at_serv": 0.5,
                    "grasped_bottle": 0.8,
                    "correct_bottle": 0.75,
                    "grasped_glass": 0.5,
                    "glass_at_serv": 0.5,
                    "empty_glass": 0.25,
                    "correct_drink": 0.8,
                    "used_glass": 0.25,
                },
                "preparing": {
                    "bottle_at_serv": 0.25,
                    "grasped_bottle": 0.8,
                    "correct_bottle": 0.75,
                    "grasped_glass": 0.5,
                    "glass_at_serv": 0.5,
                    "empty_glass": 0.25,
                    "correct_drink": 0.8,
                    "used_glass": 0.25,
                },
            },
            "benchmark": {
                "serving": {
                    "bottle_at_serv": 0.1,
                    "grasped_bottle": 0.1,
                    "correct_bottle": 0.5,
                    "grasped_glass": 0.1,
                    "glass_at_serv": 0.1,
                    "empty_glass": 0.9,
                    "correct_drink": 0.5,
                    "used_glass": 0.1,
                },
                "preparing": {
                    "bottle_at_serv": 0.1,
                    "grasped_bottle": 0.1,
                    "correct_bottle": 0.5,
                    "grasped_glass": 0.1,
                    "glass_at_serv": 0.1,
                    "empty_glass": 0.9,
                    "correct_drink": 0.5,
                    "used_glass": 0.1,
                },
            },
        }
        self.client_preferences = {}

        #Client generation
        self.single_client = single_client
        
        if self.single_client:
            self.client_preferences[1] = {"preference": 2, "likes_shake": True}

        else:
            if random_preferences:
                for client_id in range(1, 4):
                    local_rng = numpy.random.default_rng(self.random_seed ^ (client_id * 0xDEAD))
                    self.client_preferences[client_id] = {
                        "preference": int(local_rng.integers(1, n_bottles + 1)),
                        "likes_shake": local_rng.choice([True, False])
                    }
            else:
                self.client_preferences[1] = {"preference": 1, "likes_shake": False} # Client 1 prefers drink type 1, no shake
                self.client_preferences[2] = {"preference": 2, "likes_shake": True} # Client 2 prefers drink type 2, shake
                self.client_preferences[3] = {"preference": 3, "likes_shake": False} # Client 3 prefers drink type 3, no shake

        self.prep_area = {"x_min": 0.0, "x_max": 0.6, "y_min": 0.9, "y_max": 1.1}
        self.serv_area = {"x_min": 0.4, "x_max": 0.7, "y_min": 0.5, "y_max": 0.9}
        prep_x = (self.prep_area["x_min"] + self.prep_area["x_max"]) / 2.0
        prep_y = (self.prep_area["y_min"] + self.prep_area["y_max"]) / 2.0
        serv_x = (self.serv_area["x_min"] + self.serv_area["x_max"]) / 2.0
        serv_y = (self.serv_area["y_min"] + self.serv_area["y_max"]) / 2.0
        self.prep_pos = {"x": float(prep_x), "y": float(prep_y)}
        self.serving_pos = {"x": float(serv_x), "y": float(serv_y)}
        self.robot_positions = {
            "prep": {"position_id": 0, "x": 0.3, "y": 0.8, "orientation": 90.0},
            "serv": {"position_id": 1, "x": 0.3, "y": 0.7, "orientation": 0.0},
        }

        # Initialize robot and hands
        self.robot = dict(self.robot_positions["prep"])
        self.left_hand = {"used": False, "contents": {}}
        self.right_hand = {"used": False, "contents": {}}

        # Initialize scenario state
        self.client = self._make_client_state(1)
        self.bottles = []
        self.glass = None
        self.generate_bottles()
        self.generate_glass()

        # Initialize tracking state
        self.correct_drink_served = False 
        self.glass_was_cleaned = False 

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_valid_bottle_ids(self):
        return [int(b["drink_type"]) for b in self.bottles] if self.bottles else list(range(1, self.n_bottles + 1))

    def _get_wrong_drink_type(self, reference_drink):
        """Return a valid drink id different from the provided reference."""
        valid = self._get_valid_bottle_ids()
        alts = [d for d in valid if d != int(reference_drink)]
        return int(self.rng.choice(alts)) if alts else int(valid[0])

    def _is_drink_matching_preference(self):
        """Check whether the current drink in the glass matches client preference."""
        if not self.glass or not self.glass["state"] or self.glass["was_used"]:
            return False
        return int(self.glass["drink_type"]) == int(self.client["preference"]) and (self.client["likes_shake"] == self.glass["is_shaken"])

    def _is_drink_wrong(self):
        """Check whether the current drink is wrong in a way that requires cleaning (wrong type or shaken when not desired)."""
        if not self.glass or not self.glass["state"] or self.glass["was_used"]:
            return False
        return int(self.glass["drink_type"]) != int(self.client["preference"]) or (self.glass["is_shaken"] and not self.client["likes_shake"])

    def is_client_preference_known(self):
        """
        Check whether the preference of the client is known by the agent in the current episode.

        :return: True if the agent has asked for the preference with ``ask_nicely_policy``.
        :rtype: bool
        """
        return bool(self.client["preference_known"])

    def _make_client_state(self, client_id, preference_known=False):
        client_data = self.client_preferences[int(client_id)]
        return {
            "id": int(client_id),
            "preference": int(client_data["preference"]),
            "likes_shake": bool(client_data["likes_shake"]),
            "preference_known": bool(preference_known),
        }

    def _set_robot_position(self, position_id):
        if isinstance(position_id, str):
            position = position_id.lower()
        elif isinstance(position_id, int):
            position = "prep" if position_id == 0 else "serv" if position_id == 1 else None
        else:
            position = None
        if position not in self.robot_positions:
            raise ValueError(f"Invalid robot position: {position_id}")
        self.robot = dict(self.robot_positions[position])
        self._sync_grasped_objects_to_robot()

    def _sync_grasped_objects_to_robot(self):
        def apply_hand_offset(hand_name):
            orientation = math.radians(float(self.robot.get("orientation", 0.0)))
            forward_x = math.cos(orientation)
            forward_y = math.sin(orientation)
            right_x = math.sin(orientation)
            right_y = -math.cos(orientation)
            forward_offset = 0.06
            side_offset = 0.05
            side_sign = -1.0 if hand_name == "left" else 1.0
            return (
                (forward_offset * forward_x) + (side_sign * side_offset * right_x),
                (forward_offset * forward_y) + (side_sign * side_offset * right_y),
            )

        if self.left_hand["used"] and self.left_hand["contents"]:
            dx, dy = apply_hand_offset("left")
            self.left_hand["contents"]["x"] = float(self.robot["x"] + dx)
            self.left_hand["contents"]["y"] = float(self.robot["y"] + dy)
        if self.right_hand["used"] and self.right_hand["contents"]:
            dx, dy = apply_hand_offset("right")
            self.right_hand["contents"]["x"] = float(self.robot["x"] + dx)
            self.right_hand["contents"]["y"] = float(self.robot["y"] + dy)

    def _set_gripper_object(self, gripper, obj):
        if not obj:
            gripper["used"] = False
            gripper["contents"] = {}
        else:
            gripper["used"] = True
            gripper["contents"] = obj
        self._sync_grasped_objects_to_robot()

    # ------------------------------------------------------------------ #
    # World generation
    # ------------------------------------------------------------------ #

    def random_position(self, area):
        """
        Generate a random position within the specified area.

        :param area: The area where the position should be generated. It must have the keys ``x_min``, ``x_max``, ``y_min`` and ``y_max``.
        :type area: dict
        :return: A tuple with the coordinates (x, y) of the position, drawn uniformly inside the area.
        :rtype: tuple
        """
        x = self.rng.uniform(low=area["x_min"], high=area["x_max"])
        y = self.rng.uniform(low=area["y_min"], high=area["y_max"])
        return float(x), float(y)

    def generate_bottles(self, area=None):
        """
        Generate the bottles with random positions.

        The bottles replace the previous ones. There are ``n_bottles`` bottles, each one a dictionary with its ``x``, its ``y`` and its ``drink_type`` (from 1 to ``n_bottles``).

        :param area: Area where the bottles are placed. If None, the preparation area is used.
        :type area: dict
        """
        if area is None:
            area = self.prep_area
        self.bottles = []
        for i in range(1, self.n_bottles + 1):
            x, y = self.random_position(area)
            self.bottles.append(dict(x=x, y=y, drink_type=i))

    def generate_glass(self, area=None, state=False, drink_type=0.0, was_used=False, is_shaken=False):
        """
        Generate the glass at a random position inside an area.

        The glass replaces the previous one. It is a dictionary with its ``x``, its ``y``, ``state`` (whether it contains a drink), ``drink_type``, ``was_used`` and ``is_shaken``.

        :param area: Area where the glass is placed. If None, the preparation area is used.
        :type area: dict
        :param state: Whether the glass contains a drink.
        :type state: bool
        :param drink_type: Type of the drink in the glass (0.0 if it is empty).
        :type drink_type: float
        :param was_used: Whether the glass has already been used.
        :type was_used: bool
        :param is_shaken: Whether the drink in the glass has been shaken.
        :type is_shaken: bool
        """
        # Place the glass at a random position inside the preparation area
        # so episodes vary spatially like the bottles.
        if area is None:
            area = self.prep_area
        x, y = self.random_position(area)
        self.glass = dict(x=x, y=y, state=state, drink_type=drink_type, was_used=was_used, is_shaken=is_shaken)

    def _get_curriculum_section(self):
        curriculum = self.curriculum.get(self.current_curriculum, self.curriculum["balanced"])
        if self.is_at_serving_table():
            return curriculum.get("serving", {})
        return curriculum.get("preparing", {})

    def _sample_curriculum_bool(self, probability, default=0.5):
        try:
            probability = float(probability)
        except (TypeError, ValueError):
            probability = float(default)
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        return bool(self.rng.random() < probability)

    def _generate_world_from_curriculum(self):
        self.correct_drink_served = False
        self.glass_was_cleaned = False

        # Define robot position
        if self._sample_curriculum_bool(0.5):
            self._set_robot_position("serv")
        else:
            self._set_robot_position("prep")

        # Get current curriculum section based on robot position (prep or serv)
        params = self._get_curriculum_section()

        # Reset grippers
        self._set_gripper_object(self.left_hand, None)
        self._set_gripper_object(self.right_hand, None)

        # Set bottles positions
        for bottle in self.bottles:
            bottle_area = self.serv_area if self._sample_curriculum_bool(params.get("bottle_at_serv", 0.5)) else self.prep_area
            bottle["x"], bottle["y"] = self.random_position(bottle_area)

        # Pick bottle according to curriculum probability
        if self._sample_curriculum_bool(params.get("grasped_bottle", 0.5)):
            valid_bottle_ids = self._get_valid_bottle_ids()
            if valid_bottle_ids:
                # Determine whether to pick the correct bottle or a wrong one based on curriculum probability
                if self._sample_curriculum_bool(params.get("correct_bottle", 0.5)):
                    bottle_id = int(self.client["preference"])
                    if bottle_id not in valid_bottle_ids:
                        bottle_id = valid_bottle_ids[0]
                else:
                    wrong_bottle_ids = [
                        bottle_id for bottle_id in valid_bottle_ids
                        if bottle_id != int(self.client["preference"])
                    ]
                    bottle_id = int(self.rng.choice(wrong_bottle_ids if wrong_bottle_ids else valid_bottle_ids))
                # Set the right hand to hold the selected bottle
                for bottle in self.bottles:
                    if int(bottle["drink_type"]) == bottle_id:
                        self._set_gripper_object(self.right_hand, bottle)
                        break

        # Set glass position
        glass_area = self.serv_area if self._sample_curriculum_bool(params.get("glass_at_serv", 0.5)) else self.prep_area
        glass_x, glass_y = self.random_position(glass_area)
        self.glass.update({"x": glass_x, "y": glass_y})

        # Set glass state based on curriculum probabilities
        glass_is_empty = self._sample_curriculum_bool(params.get("empty_glass", 0.5))
        glass_drink_type = 0.0
        glass_is_shaken = False
        if not glass_is_empty:
            # When the glass is not empty, determine whether to set the correct drink type or a wrong one based on curriculum probability
            if self._sample_curriculum_bool(params.get("correct_drink", 0.5)):
                glass_drink_type = float(self.client["preference"])
                glass_is_shaken = bool(self.client["likes_shake"])
            else:
                # If wrong drink type, randomly choose between wrong drink type or wrong shake preference
                if self._sample_curriculum_bool(0.5):
                    glass_drink_type = float(self._get_wrong_drink_type(self.client["preference"]))
                    glass_is_shaken = bool(self.client["likes_shake"])
                else:
                    glass_drink_type = float(self.client["preference"])
                    glass_is_shaken = not bool(self.client["likes_shake"])

        # Update the glass state with the determined values
        was_used_value = self._sample_curriculum_bool(params.get("used_glass", 0.5))
        if was_used_value:
            self.correct_drink_served = True

        self.glass.update({
            "state": not glass_is_empty,
            "drink_type": glass_drink_type,
            "was_used": was_used_value,
            "is_shaken": glass_is_shaken if not glass_is_empty else False,
        })

        # Set left hand to hold the glass based on curriculum probability
        if self._sample_curriculum_bool(params.get("grasped_glass", 0.5)):
            self._set_gripper_object(self.left_hand, self.glass)

    def generate_world(self):
        """
        Generate a new world state based on the curriculum.

        The state is sampled from the curriculum again until it is not a state that would already be rewarded. Those states are the ones where the glass contains the drink that the client wants and is in the serving area, and the ones where a used glass is in the preparation area.
        """
        correct_drink_served = True
        glass_to_be_cleaned = True

        while correct_drink_served or glass_to_be_cleaned:
            self._generate_world_from_curriculum()

            # Filter states that are not possible:
            # correct_drink_served: if the glass is already in the serving position and matches the client's preference, it would be considered already served and changed to used.
            correct_drink_served = self._is_drink_matching_preference() and self.obj_is_in_serving_position(self.glass)
            # glass_to_be_cleaned: if the glass was used and is in the preparation area, it would be cleaned and reset to empty.
            glass_to_be_cleaned = self.glass.get("was_used", False) and self.obj_is_in_preparation_area(self.glass)

    # ------------------------------------------------------------------ #
    # State accessors
    # ------------------------------------------------------------------ #

    def get_bottles_state(self):
        """
        Get the current state of all the bottles.

        :return: A list with a dictionary per bottle, with the keys ``x`` (float), ``y`` (float) and ``drink_type`` (int). The list is empty if there are no bottles.
        :rtype: list
        """
        return [
            {"x": float(b["x"]), "y": float(b["y"]), "drink_type": int(b["drink_type"])}
            for b in self.bottles
        ] if self.bottles else []

    def get_glass_state(self):
        """
        Get the current state of the glass.

        :return: A dictionary with the keys ``x`` (float), ``y`` (float), ``state`` (bool, whether the glass contains a drink), ``drink_type`` (float), ``was_used`` (bool) and ``is_shaken`` (bool). If there is no glass, the coordinates and the drink type are 0.0 and the flags are False.
        :rtype: dict
        """
        if not self.glass:
            return {"x": 0.0, "y": 0.0, "state": False, "drink_type": 0.0, "was_used": False, "is_shaken": False}
        return {
            "x": float(self.glass["x"]),
            "y": float(self.glass["y"]),
            "state": bool(self.glass["state"]),
            "drink_type": float(self.glass["drink_type"]),
            "was_used": bool(self.glass["was_used"]),
            "is_shaken": bool(self.glass["is_shaken"]),
        }

    def get_robot_state(self):
        """
        Get the current state of the robot.

        :return: A dictionary with the keys ``position_id`` (int, 0 at the preparation table and 1 at the serving table), ``x`` (float), ``y`` (float) and ``orientation`` (float, in degrees).
        :rtype: dict
        """
        return {
            "position_id": int(self.robot.get("position_id", -1)),
            "x": float(self.robot.get("x", 0.0)),
            "y": float(self.robot.get("y", 0.0)),
            "orientation": float(self.robot.get("orientation", 0.0)),
        }

    def get_hands_state(self):
        """
        Get the current state of the hands of the robot.

        :return: A dictionary with the keys ``left_hand`` and ``right_hand``. Each one is a dictionary with ``used`` (bool) and ``contents``, a copy of the object that the hand holds (empty if the hand is free).
        :rtype: dict
        """
        return {
            "left_hand": {
                "used": bool(self.left_hand["used"]),
                "contents": dict(self.left_hand["contents"]) if self.left_hand["used"] else {},
            },
            "right_hand": {
                "used": bool(self.right_hand["used"]),
                "contents": dict(self.right_hand["contents"]) if self.right_hand["used"] else {},
            },
        }

    # ------------------------------------------------------------------ #
    # Positional helpers
    # ------------------------------------------------------------------ #

    def is_at_preparation_table(self):
        """
        Check whether the robot is at the preparation table.

        :return: True if the position id of the robot is 0.
        :rtype: bool
        """
        return int(self.robot.get("position_id", -1)) == 0

    def is_at_serving_table(self):
        """
        Check whether the robot is at the serving table.

        :return: True if the position id of the robot is 1.
        :rtype: bool
        """
        return int(self.robot.get("position_id", -1)) == 1

    def _is_point_in_area(self, x, y, area):
        """Check whether a Cartesian point lies inside a rectangular area."""
        return (
            area["x_min"] <= x <= area["x_max"] and
            area["y_min"] <= y <= area["y_max"]
        )

    def obj_is_in_serving_position(self, obj):
        """
        Check whether an object is inside the serving area.

        :param obj: The object, a dictionary with the keys ``x`` and ``y``.
        :type obj: dict
        :return: True if the object exists and its coordinates are inside the serving area.
        :rtype: bool
        """
        return bool(obj) and self._is_point_in_area(obj["x"], obj["y"], self.serv_area)

    def obj_is_in_preparation_area(self, obj):
        """
        Check whether an object is inside the preparation area.

        :param obj: The object, a dictionary with the keys ``x`` and ``y``.
        :type obj: dict
        :return: True if the object exists and its coordinates are inside the preparation area.
        :rtype: bool
        """
        return bool(obj) and self._is_point_in_area(obj["x"], obj["y"], self.prep_area)

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset_world(self, client_ids=None):
        """
        Reset the world to a new random state.

        The client is chosen first: it is always the client 1 when ``single_client`` is True, and a random one otherwise. Its preference is not known by the agent. Then the state of the robot, the hands, the bottles and the glass is generated with ``generate_world``.

        :param client_ids: Identifiers of the clients that can be chosen. If None, the clients 1, 2 and 3 can be chosen. It is ignored if ``single_client`` is True.
        :type client_ids: tuple
        """
        if self.single_client:
            # Single Client (by default 1)
            cid = 1

        else:
            # Random client — preferences are stored in self.client_preferences and are stable per client_id
            available_client_ids = client_ids or (1, 2, 3)
            cid = int(self.rng.choice(available_client_ids))
        
        self.client = self._make_client_state(cid)
        self.generate_world()

    # ------------------------------------------------------------------ #
    # Policies
    # ------------------------------------------------------------------ #

    def shake_glass_policy(self):
        """
        Shake the glass to mix the drink.

        It requires the left hand to hold the glass, the glass to contain a drink and the drink not to be shaken yet; otherwise nothing happens. The drink is marked as shaken.
        """
        # Require: glass in hand, glass exists, glass has a drink (state==True),
        # glass not already used, and not already shaken.
        if (
            not self.left_hand["contents"] == self.glass
            or not self.glass.get("state", False)
            or self.glass.get("is_shaken", False)
        ):
            return
        self.glass.update({"is_shaken": True})

    def pick_glass_policy(self):
        """
        Pick the glass from the preparation or the serving table.

        Nothing happens if the left hand already holds the glass. Otherwise, the left hand picks the glass if the robot is at the preparation table and the glass is in the preparation area, or if the robot is at the serving table and the glass is in the serving area.
        """
        if self.left_hand["contents"] == self.glass:
            return
        
        same_side = self.is_at_preparation_table() and self.obj_is_in_preparation_area(self.glass) or self.is_at_serving_table() and self.obj_is_in_serving_position(self.glass)
        if same_side:
            self._set_gripper_object(self.left_hand, self.glass)

    def place_glass_policy(self):
        """
        Place the glass in a location that depends on where the robot is.

        It requires the left hand to hold the glass; otherwise nothing happens.

        - At the serving table, the glass is left at the serving position. If the drink matches the preference of the client (drink type and shaken or not), the drink is considered served: the glass is marked as used and becomes empty.
        - At the preparation table, the glass is left at a random position of the preparation area. If the glass was used or the drink is wrong (wrong type, or shaken when the client does not like it), the glass is cleaned: it becomes empty and not used. Cleaning a used glass is what counts as returning it.
        """
        if self.left_hand["contents"] != self.glass:
            return

        if self.is_at_serving_table():
            self._set_gripper_object(self.left_hand, None)
            # move glass to serving position
            self.glass.update({
                "x": self.serving_pos["x"],
                "y": self.serving_pos["y"],
            })
            # Serving event: only triggers if drink matches preference
            if self._is_drink_matching_preference():
                self.correct_drink_served = True
                # Now clean the glass state for the physical object
                self.glass.update({"was_used": True, "state": False, "drink_type": 0.0, "is_shaken": False})

        elif self.is_at_preparation_table():
            # Placing at prep = cleaning the glass
            self._set_gripper_object(self.left_hand, None)
            # Update glass position to the prep table
            x, y = self.random_position(self.prep_area)
            self.glass.update({"x": x, "y": y})
            # If the glass was used, mark it as cleaned and reset its state. If it has the wrong drink, also clean it.
            if self.glass.get("was_used", False) or self._is_drink_wrong():
                self.glass_was_cleaned = True if self.glass.get("was_used", False) else False # Don't provide reward for cleaning a wrong drink, only for cleaning a used glass.
                self.glass.update({
                    "state": False,
                    "drink_type": 0.0,
                    "is_shaken": False,
                    "was_used": False,
                })

    def prepare_drink_policy(self):
        """
        Prepare the drink in the glass.

        It requires both hands to be occupied and the glass to be empty; otherwise nothing happens. The glass is filled with the drink type of the bottle in the right hand, not shaken.
        """
        # Holding both glass and bottle
        if not self.left_hand["used"] or not self.right_hand["used"]:
            return
        # Glass must be empty
        if self.glass["state"]:
            return
        # Read the drink type from the bottle in the right hand
        drink_type = float(self.right_hand["contents"]["drink_type"])

        # When preparing a fresh drink, ensure the shaken flag is cleared.
        self.glass.update({"state": True, "drink_type": drink_type, "is_shaken": False})

    def change_position_policy(self):
        """
        Move the robot instantly to the other table.

        The robot goes from the preparation table to the serving table, and from the serving table to the preparation table. The objects in its hands move with it.
        """
        if self.is_at_preparation_table():
            self._set_robot_position("serv")
        else:
            self._set_robot_position("prep")
        

    def pick_bottle_policy(self, bottle_id=None):
        """
        Pick a bottle with the right hand.

        The bottle is picked if it is on the same table where the robot is: in the preparation area when the robot is at the preparation table, or in the serving area when it is at the serving table. Otherwise the hand stays empty.

        :param bottle_id: Drink type of the bottle to pick.
        :type bottle_id: int
        :return: True if the right hand was already occupied, if ``bottle_id`` is None or 0, or if a bottle with that drink type exists (even if it was not within reach). False if there is no bottle with that drink type.
        :rtype: bool
        """
        if self.right_hand["used"] or not bottle_id:
            return True
        bottle = None
        for b in self.bottles:
            if b["drink_type"] == bottle_id:
                bottle = b
        if not bottle:
            return False
        same_side = self.is_at_preparation_table() and self.obj_is_in_preparation_area(bottle) or self.is_at_serving_table() and self.obj_is_in_serving_position(bottle)
        if same_side:
            self._set_gripper_object(self.right_hand, bottle)
        return True

    def place_bottle_policy(self):
        """
        Place the held bottle at a random position of the table where the robot is.

        It requires the right hand to hold a bottle; otherwise nothing happens. The position is chosen inside the preparation area if the robot is at the preparation table and inside the serving area otherwise.
        """
        if not self.right_hand["used"]:
            return
        area = self.prep_area if self.is_at_preparation_table() else self.serv_area
        self.right_hand["contents"]["x"], self.right_hand["contents"]["y"] = self.random_position(area)
        self._set_gripper_object(self.right_hand, None)
    

    def ask_nicely_policy(self):
        """
        Ask the client for the preference.

        From now on, the preference of the client is known by the agent in the current episode.
        """
        self.client["preference_known"] = True

    # ------------------------------------------------------------------ #
    # Goals / Rewards
    # ------------------------------------------------------------------ #

    def get_progress_goal(self):
        """
        Calculate the reward of the progress of the task.

        The shaped reward of progress has not been migrated yet to the Cartesian representation of this simulator, so the reward is always 0.0.

        :return: The reward.
        :rtype: float
        """
        # TODO: MIGRATE

        # has_glass = self.left_hand["used"]
        # has_bottle = self.right_hand["used"]
        # glass_state = self.glass["state"] if self.glass else False
        # was_used = self.glass["was_used"] if self.glass else False
        # drink_ok = self._is_drink_matching_preference()
        # at_prep = self.is_at_preparation_table()
        # at_serv = self.is_at_serving_table()

        # g_x = self.glass["x"] if self.glass else 0.0
        # g_y = self.glass["y"] if self.glass else 0.0
        # glass_at_serving = (
        #     numpy.linalg.norm([g_x - self.serving_pos["x"], g_y - self.serving_pos["y"]]) < 0.1
        # )
        # glass_at_original = (
        #     numpy.linalg.norm([g_x - self.original_glass_pos["x"], g_y - self.original_glass_pos["y"]]) < 0.1
        # )

        # if glass_at_original and not has_glass and was_used:
        #     current_step = 1.0
        # elif at_prep and has_glass and not glass_state and was_used:
        #     current_step = 0.9
        # elif has_glass and not glass_state and was_used:
        #     current_step = 0.85
        # elif glass_at_serving and not glass_state and not has_glass:
        #     current_step = 0.8
        # elif glass_at_serving and glass_state and not has_glass and drink_ok:
        #     current_step = 0.6
        # elif at_serv and has_glass and glass_state and drink_ok:
        #     current_step = 0.5
        # elif at_prep and has_glass and glass_state and drink_ok:
        #     current_step = 0.4
        # elif has_glass and has_bottle and at_prep and not glass_state:
        #     # Ready to prepare: glass empty + holding bottle
        #     current_step = 0.25
        # elif at_prep and has_glass and glass_state and not drink_ok:
        #     # Wrong drink: below 0.25 to incentivise going back and cleaning
        #     current_step = 0.15
        # elif has_glass and at_prep:
        #     current_step = 0.15
        # elif has_bottle and at_prep:
        #     current_step = 0.1
        # elif at_prep:
        #     current_step = 0.05
        # else:
        #     current_step = 0.0

        # reward = current_step
        # if self._is_policy_loop():
        #     self.sequence_repeat_count += 1
        #     reward = 0.0
        # else:
        #     self.sequence_repeat_count = 0

        # self.last_step = current_step
        reward = 0
        return float(reward)

    def get_serve_the_drink_goal(self):
        """
        Calculate the reward of the goal of serving the drink.

        :return: 1.0 if the glass is marked as used, which happens when the correct drink is served, and 0.0 otherwise.
        :rtype: float
        """
        glass_state = self.get_glass_state()
        client_state = self.client.get("id", 0)
        if not glass_state["was_used"] and client_state:
            return 0.0
        else:
            return 1.0

    def get_return_the_glass_goal(self):
        """
        Calculate the reward of the goal of returning the glass.

        The goal is reached once, in an episode, when the used glass is returned to the preparation table and cleaned.

        :return: 1.0 if the goal is reached and 0.0 otherwise.
        :rtype: float
        """
        glass_state=self.get_glass_state()
        if glass_state["was_used"]:
            return 0.0
        else:
            return 1.0

    def finished_signal(self):
        """
        Check whether the episode is finished.

        :return: True if the correct drink has been served and the used glass has been cleaned at the preparation table.
        :rtype: bool
        """
        return self.correct_drink_served and self.glass_was_cleaned


# ======================================================================== #
# ROS2 Node wrapper
# ======================================================================== #

class BartenderSimNode(Node):
    """
    ROS 2 node wrapper for BartenderSim.

    It handles all the ROS communication; the logic of the simulation lives in BartenderSim. The node is named ``BartenderSim`` and has the parameters:

    - ``random_seed``: seed of the random number generator (0 selects a random seed).
    - ``config_file``: path of the YAML experiment configuration file.
    - ``single_client``: if True, there is only one client (default False).
    - ``random_preferences``: if True, the preferences of the clients are random (default False).

    **Topics and services.** The node creates the following from the configuration file:

    - A subscription to the control topic, whose messages have the iteration and a command. The iteration is stored, the reward sensors are updated and the stage is updated. The command ``end`` shuts the node down.
    - A publisher for each perception, on its own topic.
    - A service to execute a policy (``executed_policy_service``). The request contains the name of the policy in ``policy`` and the perception in ``perception``, and the response ``success`` is always True. A timer republishes all the perceptions every ``perception_period`` seconds while this service exists.
    - A service to reset the world (``world_reset_service``).

    It is also a client of the service ``commander/load_experiment``, which it calls once the configuration is loaded to make the commander load the same file.

    **Configuration file.** The node reads these sections of the YAML file:

    - ``DiscreteEventSimulator``/``perception_period``: optional period, in seconds, of the timer that republishes the perceptions. The default is 0.01. It is stored in the attribute ``perception_period``.
    - ``DiscreteEventSimulator``/``Stages``: dictionary from the name of each stage to the iteration where it starts. The stage ``stage0`` selects the curriculum ``help``, ``stage1`` the curriculum ``balanced`` and ``stage2`` the curriculum ``benchmark``.
    - ``DiscreteEventSimulator``/``ClientStages``: optional dictionary from the name of each client stage to the iteration where it starts. It is used only when there are several clients: during ``client_stage0``, ``client_stage1`` and ``client_stage2`` the clients 1 and 2 appear, during ``client_stage3`` only the client 3, and the three clients otherwise.
    - ``DiscreteEventSimulator``/``Perceptions``: list of perceptions, each one with ``name``, ``perception_topic`` and ``perception_msg`` (the class name of the message). The perceptions ``bottles``, ``glass``, ``client``, ``robot_position``, ``glass_in_left_hand`` and ``bottle_in_right_hand`` are required. The reward sensors ``progress_goal``, ``serve_the_drink_goal``, ``return_the_glass_goal`` and ``finished_signal`` are optional. A message whose class name contains ``List`` is published as a list of messages, one that ends with ``BottleMsg`` as a structured message, one whose name contains ``Float`` as a float and any other one as a boolean.
    - ``Control``: ``id``, ``control_topic``, ``control_msg`` and, optionally, ``executed_policy_service`` with ``executed_policy_msg`` and ``world_reset_service`` with ``world_reset_msg``.

    The perceptions of positions are relative to the robot: the distance and the angle (in degrees, between -180 and 180, relative to the orientation of the robot) of each object, and its x and y coordinates with respect to the robot.
    """

    def __init__(self):
        """
        Constructor of the BartenderSimNode class.

        Declares the parameters, creates the BartenderSim simulator and the client of the commander service. The topics, services and perceptions are created later, by ``load_configuration``.
        """
        super().__init__("BartenderSim")

        self.ident = None
        self.base_messages = {}
        self.perceptions = {}
        self.sim_publishers = {}
        self.change_stage_iterations = {}
        self.change_client_stage_iterations = {}

        self.random_seed = self.declare_parameter(
            'random_seed', value=0
        ).get_parameter_value().integer_value

        self.config_file = self.declare_parameter(
            'config_file',
            descriptor=ParameterDescriptor(dynamic_typing=True)
        ).get_parameter_value().string_value

        self.single_client = self.declare_parameter(
            'single_client', value=False
        ).get_parameter_value().bool_value
        self.random_preferences = self.declare_parameter(
                    'random_preferences', value=False
        ).get_parameter_value().bool_value
        self.get_logger().info(f"Single client mode: {self.single_client}")
        self.get_logger().info(f"Randomized preferences: {self.random_preferences}")


        self.simulator = BartenderSim(random_seed=self.random_seed, single_client=self.single_client)
        self.random_seed = self.simulator.random_seed
        self.get_logger().info(f"Setting random number generator with seed {self.random_seed}")

        self.cbgroup_server = MutuallyExclusiveCallbackGroup()
        self.cbgroup_client = MutuallyExclusiveCallbackGroup()

        self.load_client = ServiceClient(LoadConfig, 'commander/load_experiment')

        self.iteration = 0
        self.current_stage = ""
        self.current_client_stage = ""

        # self.agent_bottle_subscription = self.create_subscription(
        #     Float32,
        #     "cognitive_node/world_model/last_bottle",
        #     self.agent_bottle_callback,
        #     1,
        # )

    # def agent_bottle_callback(self, msg):
    #     self.simulator.set_agent_bottle_choice(float(msg.data))

    # ------------------------------------------------------------------ #
    # Perception updates
    # ------------------------------------------------------------------ #

    def perceive_bottles(self):
        """
        Update the bottles perception based on the simulator state.

        The perception contains a message for each bottle, with its distance and angle to the robot, its drink type and its x and y coordinates relative to the robot. If there are no bottles, it contains a single message with the default values.
        """
        # Read states from simulator
        robot_state = self.simulator.get_robot_state()
        bottles_state = self.simulator.get_bottles_state()
        robot_xy = (robot_state["x"], robot_state["y"])
        robot_orientation = float(robot_state.get("orientation", 0.0))

        # Assign the perception data for bottles
        self.perceptions["bottles"].data = []
        for b in bottles_state:
            bottle_xy = (float(b["x"]), float(b["y"]))
            msg = self.base_messages["bottles"]()
            msg.distance = float(self.get_distance(robot_xy, bottle_xy))
            msg.angle = self.get_relative_angle_to_robot(robot_xy, bottle_xy, robot_orientation)
            msg.drink_type = int(b["drink_type"])
            msg.x = float(bottle_xy[0] - robot_xy[0])
            msg.y = float(bottle_xy[1] - robot_xy[1])
            self.perceptions["bottles"].data.append(msg)
        if not self.perceptions["bottles"].data:
            self.perceptions["bottles"].data.append(self.base_messages["bottles"]())

    def perceive_glass(self):
        """
        Update the glass perception based on the simulator state.

        The perception contains a single message with the distance and angle to the robot, the x and y coordinates relative to the robot, and the state, drink type, used flag and shaken flag of the glass.
        """
        # Read states from simulator
        robot_state = self.simulator.get_robot_state()
        robot_xy = (robot_state["x"], robot_state["y"])
        robot_orientation = float(robot_state.get("orientation", 0.0))
        gs = self.simulator.get_glass_state()
        glass_xy = (float(gs["x"]), float(gs["y"]))

        # Assign the perception data for the glass
        self.perceptions["glass"].data = []
        msg = self.base_messages["glass"]()
        msg.distance = float(self.get_distance(robot_xy, glass_xy))
        msg.angle = self.get_relative_angle_to_robot(robot_xy, glass_xy, robot_orientation)
        msg.state = bool(gs["state"])
        msg.drink_type = float(gs["drink_type"])
        msg.was_used = bool(gs["was_used"])
        msg.is_shaken = bool(gs["is_shaken"])
        msg.x = float(glass_xy[0] - robot_xy[0])
        msg.y = float(glass_xy[1] - robot_xy[1])
        self.perceptions["glass"].data.append(msg)

    def perceive_client(self):
        """
        Update the client perception based on the simulator state.

        The perception contains a single message with the id of the client. The preference and the taste for shaken drinks are only given when the agent knows the preference of the client; otherwise they are 0 and False. If the message has the field ``preference_known``, it is also filled.
        """
        self.perceptions["client"].data = []
        client_msg = self.base_messages["client"]()
        client_msg.id = int(self.simulator.client["id"])
        client_msg.preference = int(self.simulator.client["preference"]) if self.simulator.client["preference_known"] else 0
        client_msg.likes_shake = bool(self.simulator.client["likes_shake"]) if self.simulator.client["preference_known"] else False
        if hasattr(client_msg, "preference_known"):
            client_msg.preference_known = bool(self.simulator.client["preference_known"])
        self.perceptions["client"].data.append(client_msg)

    def update_perceptions_from_simulator(self):
        """
        Update all the perceptions from the simulator state.

        This updates the bottles, the glass and the client, the position of the robot (0.0 at the preparation table and 0.95 at the serving table) and the hands. For each hand, the perception tells whether it is occupied, with the kind of object (``glass`` or ``bottle``) and its drink type in ``contents`` and ``contents_id``.
        """
        self.perceive_bottles()
        self.perceive_glass()
        self.perceive_client()

        # Perceive robot position and hand states
        robot_state = self.simulator.get_robot_state()
        hands_state = self.simulator.get_hands_state()

        self.perceptions["robot_position"].data = min(float(robot_state["position_id"]), 0.95)
        self.perceptions["glass_in_left_hand"].data = bool(hands_state["left_hand"]["used"])
        self.perceptions["glass_in_left_hand"].contents = "glass" if hands_state["left_hand"]["used"] else ""
        self.perceptions["glass_in_left_hand"].contents_id = int(hands_state["left_hand"]["contents"].get("drink_type", -1))

        self.perceptions["bottle_in_right_hand"].data = bool(hands_state["right_hand"]["used"])
        self.perceptions["bottle_in_right_hand"].contents = "bottle" if hands_state["right_hand"]["used"] else ""
        self.perceptions["bottle_in_right_hand"].contents_id = int(hands_state["right_hand"]["contents"].get("drink_type", -1))


    @staticmethod
    def get_relative_angle(x1_y1, x2_y2):
        """
        Return the relative angle between two points.

        :param x1_y1: Tuple with the coordinates of the first point (x1, y1).
        :type x1_y1: tuple
        :param x2_y2: Tuple with the coordinates of the second point (x2, y2).
        :type x2_y2: tuple
        :return: Relative angle in degrees between the two points. If both positions are equal,
                 returns 0.0.
        :rtype: float
        """
        (x1, y1) = x1_y1
        (x2, y2) = x2_y2
        if math.isclose(x1, x2) and math.isclose(y1, y2):
            return 0.0
        return math.atan2(y2 - y1, x2 - x1) * 180 / math.pi

    @staticmethod
    def get_relative_angle_to_robot(robot_xy, target_xy, robot_orientation):
        """
        Return the angle of a target in the frame of the robot.

        :param robot_xy: Tuple with the coordinates of the robot (x, y).
        :type robot_xy: tuple
        :param target_xy: Tuple with the coordinates of the target (x, y).
        :type target_xy: tuple
        :param robot_orientation: Orientation of the robot, in degrees.
        :type robot_orientation: float
        :return: Angle of the target relative to the orientation of the robot, in degrees, normalized to the range [-180, 180).
        :rtype: float
        """
        return float(
            ((BartenderSimNode.get_relative_angle(robot_xy, target_xy) - robot_orientation + 180.0) % 360.0) - 180.0
        )

    @staticmethod
    def get_distance(x1_y1, x2_y2):
        """
        Return the Euclidean distance between two points.

        :param x1_y1: Tuple with the coordinates of the first point (x1, y1).
        :type x1_y1: tuple
        :param x2_y2: Tuple with the coordinates of the second point (x2, y2).
        :type x2_y2: tuple
        :return: Euclidean distance between the two points.
        :rtype: float
        """
        return distance.euclidean(x1_y1, x2_y2)

    # ------------------------------------------------------------------ #
    # World control
    # ------------------------------------------------------------------ #

    def reset_world(self, data=None):
        """
        Reset the world to a new state.

        The simulator is reset, using the clients of the current client stage when there are several clients. Then the perceptions and the reward sensors are updated and published.

        :param data: Request of the reset service. It is not used.
        :type data: object
        """
        self.get_logger().info("Resetting world...")
        client_ids = None
        if not self.single_client:
            client_ids = self._get_client_ids_for_stage()
        self.simulator.reset_world(client_ids=client_ids)
        self.update_perceptions_from_simulator()
        self.update_reward_sensor()
        self.publish_perceptions()
        self.get_logger().info("AFTER WORLD RESET:")
        self._log_simulator_state()

    def update_stage(self):
        """
        Update the stage and the curriculum based on the current iteration.

        The current stage and client stage are the last ones, in the order of the configuration, whose start iteration has been reached. The stages ``stage0``, ``stage1`` and ``stage2`` select the curricula ``help``, ``balanced`` and ``benchmark`` of the simulator.
        """

        for stage, start_iter in self.change_stage_iterations.items():
            if self.iteration >= int(start_iter):
                self.current_stage = stage

        for stage, start_iter in self.change_client_stage_iterations.items():
            if self.iteration >= int(start_iter):
                self.current_client_stage = stage

        if self.current_stage == "stage0":
            self.simulator.current_curriculum = "help"
        if self.current_stage == "stage1":
            self.simulator.current_curriculum = "balanced"
        if self.current_stage == "stage2":
            self.simulator.current_curriculum = "benchmark"
        self.get_logger().info(
            f"Current stage: {self.current_stage}, client stage: {self.current_client_stage}, "
            f"curriculum: {self.simulator.current_curriculum}"
        )

    def _get_client_ids_for_stage(self):
        if self.single_client:
            return None
        if self.current_client_stage in ("client_stage0", "client_stage1", "client_stage2"):
            return (1, 2)
        if self.current_client_stage == "client_stage3":
            return (3,)
        return (1, 2, 3)

    def update_reward_sensor(self):
        """
        Update the goal sensors' values from the simulator.

        Only the sensors present in the configured perceptions (``progress_goal``, ``serve_the_drink_goal``, ``return_the_glass_goal`` and ``finished_signal``) are updated.
        """
        if "progress_goal" in self.perceptions:
            progress = self.simulator.get_progress_goal()
            self.perceptions["progress_goal"].data = progress
            self.get_logger().info(f"Progress reward: {progress}")
        if "serve_the_drink_goal" in self.perceptions:
            self.perceptions["serve_the_drink_goal"].data = self.simulator.get_serve_the_drink_goal()
            self.get_logger().info(f"Serve the drink reward: {self.perceptions['serve_the_drink_goal'].data}")
        if "return_the_glass_goal" in self.perceptions:
            self.perceptions["return_the_glass_goal"].data = self.simulator.get_return_the_glass_goal()
            self.get_logger().info(f"Return the glass reward: {self.perceptions['return_the_glass_goal'].data}")
        if "finished_signal" in self.perceptions:
            self.perceptions["finished_signal"].data = self.simulator.finished_signal()
            self.get_logger().info(f"Finished signal: {self.perceptions['finished_signal'].data}")

    def publish_perceptions(self):
        """Publish the current perceptions to the corresponding topics."""
        # The debug texts are built only if they are going to be logged: perceptions are published every period
        debug = self.get_logger().is_enabled_for(LoggingSeverity.DEBUG)
        for ident, publisher in self.sim_publishers.items():
            perception = self.perceptions[ident]
            if debug:
                if hasattr(perception, "data"):
                    debug_value = perception.data
                elif hasattr(perception, "id"):
                    debug_value = {
                        "id": perception.id,
                        "distance": perception.distance,
                        "angle": perception.angle,
                        "x": perception.x,
                        "y": perception.y,
                    }
                else:
                    debug_value = perception
                self.get_logger().debug(f"Publishing {ident} = {debug_value}")
            publisher.publish(perception)

    # ------------------------------------------------------------------ #
    # Policy execution
    # ------------------------------------------------------------------ #

    def execute_policy(self, policy_name, perception):
        """
        Execute a policy by name.

        The policy is the method ``<policy_name>_policy`` of the node, which is called with the perception.

        :param policy_name: Name of the policy, without the ``_policy`` suffix.
        :type policy_name: str
        :param perception: Perception sent with the request of the policy.
        :type perception: object
        :return: True if the policy exists and was executed, False otherwise.
        :rtype: bool
        """
        method = getattr(self, policy_name + "_policy", None)
        if method and callable(method):
            method(perception=perception)
            return True
        return False

    def shake_glass_policy(self, perception=None):
        """
        Shake the glass to mix the drink.

        :param perception: Perception sent with the request. It is not used.
        :type perception: object
        """
        self.simulator.shake_glass_policy()

    def pick_glass_policy(self, perception=None):
        """
        Pick the glass from the preparation or the serving table.

        :param perception: Perception sent with the request. It is not used.
        :type perception: object
        """
        self.simulator.pick_glass_policy()

    def place_glass_policy(self, perception=None):
        """
        Place the glass in a location that depends on where the robot is.

        :param perception: Perception sent with the request. It is not used.
        :type perception: object
        """
        self.simulator.place_glass_policy()

    def prepare_drink_policy(self, perception=None):
        """
        Prepare the drink if the robot holds the glass (empty) and a bottle.

        :param perception: Perception sent with the request. It is not used.
        :type perception: object
        """
        self.simulator.prepare_drink_policy()

    def change_position_policy(self, perception=None):
        """
        Move the robot instantly between the preparation and the serving tables.

        :param perception: Perception sent with the request. It is not used.
        :type perception: object
        """
        self.simulator.change_position_policy()

    def pick_bottle_policy(self, perception=None):
        """
        Pick the bottle indicated in the perception.

        The drink type of the bottle is read from the feature ``bottles:drink_type`` of the perception (its last value). That value is scaled by the number of bottles plus one to obtain the drink type, except when it is close to 0.98, which selects the last bottle.

        :param perception: Perception sent with the request, as a message that can be converted into a container.
        :type perception: object
        """
        perception = Container.from_msg(perception)
        bottle_id_raw = perception.read().sel(features=["bottles:drink_type"]).values[-1]
        bottle_id = int(bottle_id_raw*(self.simulator.n_bottles+1)) if not math.isclose(bottle_id_raw, 0.98) else self.simulator.n_bottles 
        success = self.simulator.pick_bottle_policy(bottle_id=bottle_id)
        if not success:
            self.get_logger().error(f"Picked bottle with id {bottle_id} not found in the simulator state.")

    def place_bottle_policy(self, perception=None):
        """
        Place the held bottle at a random position of the table where the robot is.

        :param perception: Perception sent with the request. It is not used.
        :type perception: object
        """
        self.simulator.place_bottle_policy()

    def ask_nicely_policy(self, perception=None):
        """
        Ask the client for the preference.

        :param perception: Perception sent with the request. It is not used.
        :type perception: object
        """
        self.simulator.ask_nicely_policy()

    # ------------------------------------------------------------------ #
    # ROS callbacks
    # ------------------------------------------------------------------ #

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

        The iteration of the message is stored, and the reward sensors and the stage are updated. If the command is ``end``, the node shuts down.

        :param data: Control message, with the fields ``iteration`` and ``command``.
        :type data: object
        """
        self.get_logger().debug(f"Command received... ITERATION: {data.iteration}")
        self.iteration = data.iteration

        self.update_reward_sensor()
        self.update_stage()
        # World resets arrive through the world reset service, not through this topic
        if data.command == "end":
            self.get_logger().info("Ending simulator as requested by LTM...")
            rclpy.shutdown()

    def new_action_service_callback(self, request, response):
        """
        Execute a policy and publish the new perceptions.

        The policy ``request.policy`` is executed with ``request.perception``, and the perceptions are updated before and after. A policy that does not exist does not change the world. Then the reward sensors are updated and all the perceptions are published.

        :param request: Request of the service, with the name of the policy in ``policy`` and the perception in ``perception``.
        :type request: object
        :param response: Response of the service. Its field ``success`` is set to True.
        :type response: object
        :return: The response.
        :rtype: object
        """
        self.get_logger().info(f"Executing policy {request.policy}")
        self.get_logger().info(f"ITERATION: {self.iteration}")

        self.update_perceptions_from_simulator()
        self.get_logger().info(f"BEFORE EXECUTION:")
        self._log_simulator_state()
        self.execute_policy(request.policy, request.perception)
        self.update_perceptions_from_simulator()
        self.get_logger().info(f"AFTER EXECUTION:")
        self._log_simulator_state()

        self.update_reward_sensor()

        self.publish_perceptions()
        response.success = True
        return response

    def _log_simulator_state(self):
        self.get_logger().info(f"Robot state: {self.simulator.get_robot_state()}")
        self.get_logger().info(f"Hands state: {self.simulator.get_hands_state()}")
        self.get_logger().info(f"Bottles state: {self.simulator.get_bottles_state()}")
        self.get_logger().info(f"Glass state: {self.simulator.get_glass_state()}")
        self.get_logger().info(f"Client state: {self.simulator.client}")

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def setup_experiment_stages(self, stages):
        """
        Set up the stages of the experiment with their corresponding iterations.

        :param stages: Dictionary from the name of each stage to the iteration where it starts.
        :type stages: dict
        """
        for stage in stages:
            self.change_stage_iterations[stage] = stages[stage]

    def setup_client_stages(self, stages):
        """
        Set up the client stages of the experiment with their corresponding iterations.

        :param stages: Dictionary from the name of each client stage to the iteration where it starts.
        :type stages: dict
        """
        for stage in stages:
            self.change_client_stage_iterations[stage] = stages[stage]

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
            if "List" in classname:
                self.perceptions[sid].data = []
                self.base_messages[sid] = class_from_classname(classname.replace("List", ""))
            elif classname.endswith("BottleMsg"):
                # BottleMsg is a structured message, not a scalar wrapper.
                # Initialize all known fields so later updates can safely assign them.
                self.perceptions[sid].id = -1
                self.perceptions[sid].distance = -1.0
                self.perceptions[sid].angle = -1.0
                self.perceptions[sid].x = -1.0
                self.perceptions[sid].y = -1.0
            elif "Float" in classname:
                self.perceptions[sid].data = 0.0
            else:
                self.perceptions[sid].data = False
            self.get_logger().info(f"Publishing to: {topic}")
            self.sim_publishers[sid] = self.create_publisher(message, topic, 0)

    def setup_control_channel(self, simulation):
        """
        Configure the ROS topic and the services where the node listens for commands to be executed.

        The timer that republishes the perceptions uses ``perception_period``, which is set by ``load_configuration``.

        :param simulation: The ``Control`` section of the configuration file.
        :type simulation: dict
        """
        self.ident = simulation["id"]
        topic = simulation["control_topic"]
        classname = simulation["control_msg"]
        message = class_from_classname(classname)
        self.get_logger().info(f"Subscribing to: {topic}")
        self.create_subscription(message, topic, self.new_command_callback, 0)

        service_policy = simulation.get("executed_policy_service")
        service_world_reset = simulation.get("world_reset_service")

        if service_policy:
            self.get_logger().info(f"Creating action server: {service_policy}")
            message_policy_srv = class_from_classname(simulation["executed_policy_msg"])
            self.create_service(
                message_policy_srv, service_policy,
                self.new_action_service_callback,
                callback_group=self.cbgroup_server,
            )
            self.perceptions_timer = self.create_timer(
                self.perception_period, self.publish_perceptions,
                callback_group=self.cbgroup_server,
            )

        if service_world_reset:
            self.message_world_reset = class_from_classname(simulation["world_reset_msg"])
            self.create_service(
                self.message_world_reset, service_world_reset,
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

        If no configuration file is given or it does not exist, an error is logged and ROS is shut down. Otherwise, the period of the perceptions, the stages, the client stages, the perceptions and the control channel are set up from the file, and the file is loaded in the commander node.
        """
        if not self.config_file:
            self.get_logger().error("No configuration file specified!")
            rclpy.shutdown()
            return
        if not os.path.isfile(self.config_file):
            self.get_logger().error(f"{self.config_file} does not exist!")
            rclpy.shutdown()
            return

        self.get_logger().info(f"Loading configuration from {self.config_file}...")
        with open(self.config_file, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)

        # Period (seconds) of the periodic publication of perceptions.
        self.perception_period = config["DiscreteEventSimulator"].get("perception_period", DEFAULT_PERCEPTION_PERIOD)
        self.setup_experiment_stages(config["DiscreteEventSimulator"]["Stages"])
        self.setup_client_stages(config["DiscreteEventSimulator"].get("ClientStages", {}))
        self.setup_perceptions(config["DiscreteEventSimulator"]["Perceptions"])
        self.setup_control_channel(config["Control"])
        self.load_experiment_file_in_commander()


# ======================================================================== #
# Entry point
# ======================================================================== #

def main(args=None):
    """
    Run the bartender simulator node.

    The node is spun with an events executor if it is available, and with a single-threaded executor otherwise, until the node is shut down or interrupted from the keyboard.

    :param args: Command line arguments for ROS.
    :type args: list
    """
    rclpy.init(args=args)
    sim = BartenderSimNode()
    sim.load_configuration()
    # Much cheaper than the Python executors. The simulator runs in a single thread either way, and
    # none of its callbacks waits for anything, so callback groups are not needed.
    executor = EventsExecutor() if EventsExecutor is not None else SingleThreadedExecutor()
    executor.add_node(sim)
    register_executor_thread() # The callbacks of the EventsExecutor are not visible in the Python stack
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Keyboard Interrupt: shutting down simulator...")
    finally:
        sim.destroy_node()


if __name__ == "__main__":
    main()

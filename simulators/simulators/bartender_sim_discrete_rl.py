import os
import math

import numpy
import numpy as np
import yaml
import yamlloader
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import Float32
from scipy.spatial import distance

from core.container import Container
from core.service_client import ServiceClient
from core_interfaces.srv import LoadConfig
from core.utils import class_from_classname, resolve_seed


class BartenderSim:
    """
    BartenderSim v2 - Fixed + improved discrete simulator.

    Key fixes vs original:
      1. reset_world: if→elif chains; no silent fall-through
      2. ask_nicely: stable non-trivial per-client preferences (not pref=client_id)
      3. place_glass at prep: cleans drink (replaces discard_drink_policy)
      4. place_glass at serv: only marks served if drink matches preference
      5. pick_glass: can always pick up used glass (for return)
      6. place_bottle: only repositions the returned bottle, not all bottles
      7. change_position: instant teleport
      8. Reward hierarchy: fixed perverse incentive (wrong drink + bottle was > wrong drink)
      9. n_bottles configurable
     10. is_in_transit() exposed
    """

    def __init__(self, random_seed=0, n_bottles=3):
        self.random_seed = resolve_seed(random_seed)
        self.rng = numpy.random.default_rng(self.random_seed)
        self.service_world_reset = False
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
                    "empty_glass": 0.5,
                    "correct_drink": 0.75,
                    "used_glass": 0.25,
                },
                "preparing": {
                    "bottle_at_serv": 0.25,
                    "grasped_bottle": 0.667,
                    "correct_bottle": 0.75,
                    "grasped_glass": 0.5,
                    "glass_at_serv": 0.5,
                    "empty_glass": 0.333,
                    "correct_drink": 0.5,
                    "used_glass": 0.5,
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
        for client_id in range(1, 4):
            local_rng = numpy.random.default_rng(self.random_seed ^ (client_id * 0xDEAD))
            self.client_preferences[client_id] = {
                "preference": int(local_rng.integers(1, n_bottles + 1)),
                "likes_shake": bool(local_rng.integers(0, 2)),
            }

        
        

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
        """Whether client preference is known to the agent in the current episode."""
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
        """Generate a random position within the specified area."""
        x = self.rng.uniform(low=area["x_min"], high=area["x_max"])
        y = self.rng.uniform(low=area["y_min"], high=area["y_max"])
        return float(x), float(y)

    def generate_bottles(self, area=None):
        """Generate bottles with random positions."""
        if area is None:
            area = self.prep_area
        self.bottles = []
        for i in range(1, self.n_bottles + 1):
            x, y = self.random_position(area)
            self.bottles.append(dict(x=x, y=y, drink_type=i))

    def generate_glass(self, area=None, state=False, drink_type=0.0, was_used=False, is_shaken=False):
        """Generate a glass at a random position inside the preparation area."""
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
        "Generates a new world state based on the curriculum. Prevents rewarded states from being generated."
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
        """Get the current state of all bottles."""
        return [
            {"x": float(b["x"]), "y": float(b["y"]), "drink_type": int(b["drink_type"])}
            for b in self.bottles
        ] if self.bottles else []

    def get_glass_state(self):
        """Get the current state of the glass."""
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
        """Get the current state of the robot."""
        return {
            "position_id": int(self.robot.get("position_id", -1)),
            "x": float(self.robot.get("x", 0.0)),
            "y": float(self.robot.get("y", 0.0)),
            "orientation": float(self.robot.get("orientation", 0.0)),
        }

    def get_hands_state(self):
        """Get the current state of the robot's hands."""
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
        return int(self.robot.get("position_id", -1)) == 0

    def is_at_serving_table(self):
        return int(self.robot.get("position_id", -1)) == 1

    def _is_point_in_area(self, x, y, area):
        """Check whether a Cartesian point lies inside a rectangular area."""
        return (
            area["x_min"] <= x <= area["x_max"] and
            area["y_min"] <= y <= area["y_max"]
        )

    def obj_is_in_serving_position(self, obj):
        return bool(obj) and self._is_point_in_area(obj["x"], obj["y"], self.serv_area)

    def obj_is_in_preparation_area(self, obj):
        return bool(obj) and self._is_point_in_area(obj["x"], obj["y"], self.prep_area)

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset_world(self):
        """Reset the world to a new random state."""
        self.correct_drink_served = False
        self.glass_was_cleaned = False

        # Random client — preferences are stored in self.client_preferences and are stable per client_id
        cid = int(self.rng.integers(1, 4))
        self.client = self._make_client_state(cid)
        self.generate_world()

    # ------------------------------------------------------------------ #
    # Policies
    # ------------------------------------------------------------------ #

    def shake_glass_policy(self):
        """Shake the glass to mix the drink."""
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
        """Pick glass from prep or serving table."""
        if self.left_hand["contents"] == self.glass:
            return
        
        same_side = self.is_at_preparation_table() and self.obj_is_in_preparation_area(self.glass) or self.is_at_serving_table() and self.obj_is_in_serving_position(self.glass)
        if same_side:
            self._set_gripper_object(self.left_hand, self.glass)

    def place_glass_policy(self):
        """
        Place glass in context-appropriate location.
        - At serving table: places glass; marks served if drink matches preference.
        - At prep table: places glass AND cleans it (state=False, drink_type=0).
          This is the only way to clear a wrong drink — bring it back to prep.
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
        """Prepare drink if holding glass (empty, unused) and bottle."""
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
        """Instant teleport between prep and serving tables."""
        if self.is_at_preparation_table():
            self._set_robot_position("serv")
        else:
            self._set_robot_position("prep")
        

    def pick_bottle_policy(self, bottle_id=None):
        """Pick bottle by agent choice, fallback to client preference, then first available."""
        if self.right_hand["used"] or not bottle_id:
            return
        bottle = None
        for b in self.bottles:
            if b["drink_type"] == bottle_id:
                bottle = b
        if not bottle:
            return
        same_side = self.is_at_preparation_table() and self.obj_is_in_preparation_area(bottle) or self.is_at_serving_table() and self.obj_is_in_serving_position(bottle)
        if same_side:
            self._set_gripper_object(self.right_hand, bottle)

    def place_bottle_policy(self):
        """Place the held bottle back to a random position in the prep area."""
        if not self.right_hand["used"]:
            return
        area = self.prep_area if self.is_at_preparation_table() else self.serv_area
        self.right_hand["contents"]["x"], self.right_hand["contents"]["y"] = self.random_position(area)
        self._set_gripper_object(self.right_hand, None)
    

    def ask_nicely_policy(self):
        """
        Ask client for preference.
        """
        self.client["preference_known"] = True

    # ------------------------------------------------------------------ #
    # Goals / Rewards
    # ------------------------------------------------------------------ #

    def get_progress_goal(self):
        """
        Shaped reward reflecting task progress.
        Ordered strictly by achievement level; no perverse incentives.
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
        """Reward = 1.0 when correct drink is served."""
        glass_state = self.get_glass_state()
        client_state = self.client.get("id", 0)
        if not glass_state["was_used"] and client_state:
            return 0.0
        else:
            return 1.0

    def get_return_the_glass_goal(self):
        """Reward = 1.0 once per episode when used glass returns to prep."""
        glass_state=self.get_glass_state()
        if glass_state["was_used"]:
            return 0.0
        else:
            return 1.0


# ======================================================================== #
# ROS2 Node wrapper
# ======================================================================== #

class BartenderSimNode(Node):
    """
    ROS2 Node wrapper for BartenderSim.
    Handles all ROS communication; pure logic lives in BartenderSim.
    """

    def __init__(self):
        super().__init__("BartenderSim")

        self.ident = None
        self.base_messages = {}
        self.perceptions = {}
        self.sim_publishers = {}
        self.change_stage_iterations = {}

        self.random_seed = self.declare_parameter(
            'random_seed', value=0
        ).get_parameter_value().integer_value

        self.config_file = self.declare_parameter(
            'config_file',
            descriptor=ParameterDescriptor(dynamic_typing=True)
        ).get_parameter_value().string_value

        self.simulator = BartenderSim(random_seed=self.random_seed)
        self.random_seed = self.simulator.random_seed
        self.get_logger().info(f"Setting random number generator with seed {self.random_seed}")

        self.cbgroup_server = MutuallyExclusiveCallbackGroup()
        self.cbgroup_client = MutuallyExclusiveCallbackGroup()

        self.load_client = ServiceClient(LoadConfig, 'commander/load_experiment')

        self.iteration = 0
        self.current_stage = ""

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
        Update the perception of bottles based on the simulator state.
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
        Update the perception of the glass based on the simulator state.
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
        Update the perception of the client based on the simulator state.
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
        Return the target angle normalized into the robot frame.
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
        self.get_logger().info("Resetting world...")
        self.simulator.reset_world()
        self.update_perceptions_from_simulator()
        self.update_reward_sensor()
        self.publish_perceptions()
        self.get_logger().info("AFTER WORLD RESET:")
        self._log_simulator_state()

    def update_stage(self):
        """Update the stage perception based on the current iteration."""

        for stage, start_iter in self.change_stage_iterations.items():
            if self.iteration >= int(start_iter):
                self.current_stage = stage

        if self.current_stage == "stage0":
            self.simulator.current_curriculum = "help"
        if self.current_stage == "stage1":
            self.simulator.current_curriculum = "balanced"
        if self.current_stage == "stage2":
            self.simulator.current_curriculum = "benchmark"
        self.get_logger().info(f"Current stage: {self.current_stage}, curriculum: {self.simulator.current_curriculum}")

    def update_reward_sensor(self):
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

    def publish_perceptions(self):
        for ident, publisher in self.sim_publishers.items():
            perception = self.perceptions[ident]
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
        """Execute a policy by name and update tracking state."""
        method = getattr(self, policy_name + "_policy", None)
        if method and callable(method):
            method(perception=perception)
            return True
        return False

    def shake_glass_policy(self, perception=None):
        """Shake the glass to mix the drink."""
        self.simulator.shake_glass_policy()

    def pick_glass_policy(self, perception=None):
        """Pick glass from prep or serving table."""
        self.simulator.pick_glass_policy()

    def place_glass_policy(self, perception=None):
        """Place glass in context-appropriate location."""
        self.simulator.place_glass_policy()

    def prepare_drink_policy(self, perception=None):
        """Prepare drink if holding glass (empty, unused) and bottle."""
        self.simulator.prepare_drink_policy()

    def change_position_policy(self, perception=None):
        """Instant teleport between prep and serving tables."""
        self.simulator.change_position_policy()

    def pick_bottle_policy(self, perception=None):
        """Pick bottle by agent choice, fallback to client preference, then first available."""
        perception = Container.from_msg(perception)
        bottle_id_raw = perception.read().sel(features=["bottles:drink_type"]).values[-1]
        bottle_id = int(bottle_id_raw*(self.simulator.n_bottles+1)) if not math.isclose(bottle_id_raw, 0.98) else self.simulator.n_bottles 
        self.simulator.pick_bottle_policy(bottle_id=bottle_id)

    def place_bottle_policy(self, perception=None):
        """Place the held bottle back to a random position in the prep area."""
        self.simulator.place_bottle_policy()

    def ask_nicely_policy(self, perception=None):
        """Ask client for preference."""
        self.simulator.ask_nicely_policy()

    # ------------------------------------------------------------------ #
    # ROS callbacks
    # ------------------------------------------------------------------ #

    def world_reset_service_callback(self, request, response):
        self.reset_world(request)
        response.success = True
        return response

    def new_command_callback(self, data):
        self.get_logger().debug(f"Command received... ITERATION: {data.iteration}")
        self.iteration = data.iteration

        self.update_reward_sensor()
        self.update_stage()
        if data.command == "reset_world" and not self.service_world_reset:
            self.reset_world(data)
        elif data.command == "end":
            self.get_logger().info("Ending simulator as requested by LTM...")
            rclpy.shutdown()

    def new_action_service_callback(self, request, response):
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
        for stage in stages:
            self.change_stage_iterations[stage] = stages[stage]

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
                0.05, self.publish_perceptions,
                callback_group=self.cbgroup_server,
            )

        if service_world_reset:
            self.service_world_reset = True
            self.message_world_reset = class_from_classname(simulation["world_reset_msg"])
            self.create_service(
                self.message_world_reset, service_world_reset,
                self.world_reset_service_callback,
                callback_group=self.cbgroup_server,
            )

    def load_experiment_file_in_commander(self):
        return self.load_client.send_request(file=self.config_file)

    def load_configuration(self):
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

        self.setup_experiment_stages(config["DiscreteEventSimulator"]["Stages"])
        self.setup_perceptions(config["DiscreteEventSimulator"]["Perceptions"])
        self.setup_control_channel(config["Control"])
        self.load_experiment_file_in_commander()


# ======================================================================== #
# Entry point
# ======================================================================== #

def main(args=None):
    rclpy.init(args=args)
    sim = BartenderSimNode()
    sim.load_configuration()
    try:
        rclpy.spin(sim)
    except KeyboardInterrupt:
        print("Keyboard Interrupt: shutting down simulator...")
    finally:
        sim.destroy_node()


if __name__ == "__main__":
    main()

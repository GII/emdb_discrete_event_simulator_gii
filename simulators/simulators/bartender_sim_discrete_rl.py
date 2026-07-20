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

    # Discrete policy IDs with explicit "none" state.
    # Used to build a unique encoding for the (last_policy, prev_policy) pair.
    POLICIES = [
        "none",
        "pick_glass",
        "place_glass",
        "pick_bottle",
        "place_bottle",
        "change_position",
        "prepare_drink",
        "ask_nicely",
        "shake_glass",
    ]
    POLICY_TO_INDEX = {policy_name: idx for idx, policy_name in enumerate(POLICIES)}

    def __init__(self, random_seed=0, n_bottles=3):
        self.random_seed = resolve_seed(random_seed)
        self.rng = numpy.random.default_rng(self.random_seed)
        self.n_bottles = n_bottles

        self.steps = [
            "on_prep",
            "on_prep_with_wrong_bottle",
            "on_prep_with_wrong_drink_glass",
            "at_serv_with_correct_drink_and_bottle",
            "at_prep_with_correct_drink_glass",
            "at_serv_with_correct_drink",
            "holding_wrong_drink_at_serv",
            "at_serv_with_used_glass",
            "at_prep_with_already_used_glass",
            "at_prep_with_correct_glass_and_bottle",
        ]

        self.bottles = []
        self.glass = None
        self.original_glass_pos = {}
        self.picked_bottle = 0
        self.agent_bottle_choice = None
        self.know_preference = {}
        self._served_shake_latch = False

        self.prep_area = {"x_min": 0.0, "x_max": 0.6, "y_min": 0.9, "y_max": 1.1}
        self.serv_area = {"x_min": 0.4, "x_max": 0.7, "y_min": 0.5, "y_max": 0.9}
        serv_x = (self.serv_area["x_min"] + self.serv_area["x_max"]) / 2.0
        serv_y = (self.serv_area["y_min"] + self.serv_area["y_max"]) / 2.0
        self.serving_pos = {
            "distance": float(numpy.linalg.norm([serv_x, serv_y])),
            "angle": float(numpy.arctan2(serv_x, serv_y)),
        }

        self.robot_position = 0.0
        self.glass_in_left_hand = False
        self.bottle_in_right_hand = False
        self.client_id = 1
        self.client_preference = 0

        self.iteration = 0
        self.last_step = -1.0
        self.last_policy_executed = None
        self.prev_policy_executed = None
        self.policy_sequence = []
        self.sequence_repeat_count = 0
        self.correct_drink_served = False
        self.serve_reward_consumed = False
        self.return_reward_consumed = False

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def set_agent_bottle_choice(self, bottle_id):
        """Set the agent's bottle choice."""
        self.agent_bottle_choice = float(bottle_id)

    def _get_valid_bottle_ids(self):
        return [int(b["id"]) for b in self.bottles] if self.bottles else list(range(1, self.n_bottles + 1))

    def _get_wrong_drink_type(self, reference_drink):
        """Return a valid drink id different from the provided reference."""
        valid = self._get_valid_bottle_ids()
        alts = [d for d in valid if d != int(reference_drink)]
        return int(self.rng.choice(alts)) if alts else int(valid[0])

    def _is_drink_matching_preference(self):
        """Check whether the current drink in the glass matches client preference."""
        if not self.glass or not self.glass["state"]:
            return False
        if self.client_preference <= 0:
            return False
        return int(self.glass["drink_type"]) == int(self.client_preference) and (self.client_likes_shake == self.glass["is_shaken"])

    def is_client_preference_known(self):
        """Whether client preference is known to the agent in the current episode."""
        return self.client_preference > 0

    def get_policy_pair_encoding(self):
        """Unique normalized encoding for (last_policy, prev_policy) in [0, 1]."""
        n_policies = len(self.POLICIES)
        last_idx = self.POLICY_TO_INDEX.get(self.last_policy_executed or "none", 0)
        prev_idx = self.POLICY_TO_INDEX.get(self.prev_policy_executed or "none", 0)
        pair_idx = (last_idx * n_policies) + prev_idx
        return float(pair_idx) / float((n_policies * n_policies) - 1)

    # ------------------------------------------------------------------ #
    # World generation
    # ------------------------------------------------------------------ #

    def random_position(self, area):
        """Generate a random position within the specified area."""
        x = self.rng.uniform(low=area["x_min"], high=area["x_max"])
        y = self.rng.uniform(low=area["y_min"], high=area["y_max"])
        return float(numpy.linalg.norm([x, y])), float(numpy.arctan2(x, y))

    def generate_bottles(self):
        """Generate bottles with random positions."""
        self.bottles = []
        for i in range(1, self.n_bottles + 1):
            dist, ang = self.random_position(self.prep_area)
            self.bottles.append(dict(distance=dist, angle=ang, id=i))

    def generate_glass(self):
        """Generate a glass at the origin."""
        # Place the glass at a random position inside the preparation area
        # so episodes vary spatially like the bottles.
        dist, ang = self.random_position(self.prep_area)
        self.glass = dict(distance=dist, angle=ang, state=False, drink_type=0.0, was_used=False, is_shaken=False)

        self.original_glass_pos = {"distance": dist, "angle": ang}

    # ------------------------------------------------------------------ #
    # State accessors
    # ------------------------------------------------------------------ #

    def get_bottles_state(self):
        """Get the current state of all bottles."""
        return [
            {"distance": float(b["distance"]), "angle": float(b["angle"]), "id": int(b["id"])}
            for b in self.bottles
        ] if self.bottles else []

    def get_selected_bottle_state(self):
        """Return the selected bottle state, preferring the agent choice when available."""
        bottle_id = self.agent_bottle_choice
        if bottle_id is None and self.client_preference > 0:
            bottle_id = float(self.client_preference)

        if bottle_id is None:
            return None

        try:
            selected_id = int(round(float(bottle_id)))
        except (TypeError, ValueError):
            return None

        for bottle in self.bottles:
            if int(bottle["id"]) == selected_id:
                return {
                    "distance": float(bottle["distance"]),
                    "angle": float(bottle["angle"]),
                    "id": selected_id,
                }
        return None

    def get_glass_state(self):
        """Get the current state of the glass."""
        if not self.glass:
            return {"distance": 0.0, "angle": 0.0, "state": False, "drink_type": 0.0, "was_used": False, "is_shaken": False}
        return {
            "distance": float(self.glass["distance"]),
            "angle": float(self.glass["angle"]),
            "state": bool(self.glass["state"]),
            "drink_type": float(self.glass["drink_type"]),
            "was_used": bool(self.glass["was_used"]),
            "is_shaken": bool(self.glass["is_shaken"]),
        }

    # ------------------------------------------------------------------ #
    # Positional helpers
    # ------------------------------------------------------------------ #

    def is_at_preparation_table(self):
        return self.robot_position < 0.2

    def is_at_serving_table(self):
        return self.robot_position >= 0.8

    def is_in_transit(self):
        return not self.is_at_preparation_table() and not self.is_at_serving_table()
    
    def _polar_to_xy(self, distance, angle):
         """        
         Convert simulator polar coordinates back to Cartesian coordinates.
         Uses the simulator convention angle = atan2(x, y).
         """
         x = float(distance) * math.sin(float(angle))
         y = float(distance) * math.cos(float(angle))
         return x, y

    def _is_point_in_area(self, distance, angle, area):
        """Check whether a polar point lies inside a rectangular area in Cartesian space."""
        x, y = self._polar_to_xy(distance, angle)
        return (
           area["x_min"] <= x <= area["x_max"] and
            area["y_min"] <= y <= area["y_max"]
        )


    def glass_is_in_serving_position(self):
         return bool(self.glass) and self._is_point_in_area(self.glass["distance"], self.glass["angle"], self.serv_area)

    def glass_is_in_preparation_area(self):
        return bool(self.glass) and self._is_point_in_area(self.glass["distance"], self.glass["angle"], self.prep_area)

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset_world(self):
        """Reset the world to a new random state."""
        self.picked_bottle = 0
        self.agent_bottle_choice = None
        self.last_step = -1.0
        self.prev_policy_executed = None
        self.last_policy_executed = None
        self.policy_sequence = []
        self.sequence_repeat_count = 0
        self.correct_drink_served = False
        self.serve_reward_consumed = False
        self.return_reward_consumed = False
        self._served_shake_latch = False

        self.generate_bottles()
        self.generate_glass()

        # Random client — preference unknown until ask_nicely is called
        cid = int(self.rng.integers(1, 4))
        self.client_id = cid
        self.client_preference = self.know_preference.get(cid, 0)
        self.client_likes_shake = bool(self.rng.integers(0, 2))

        # Default robot state
        self.robot_position = 0.0
        self.glass_in_left_hand = False
        self.bottle_in_right_hand = False

        step = self.rng.choice(self.steps)

        if step == "on_prep":
            pass  # defaults already set

        elif step == "on_prep_with_wrong_bottle":
            self.bottle_in_right_hand = True
            self.picked_bottle = self._get_wrong_drink_type(self.client_preference)
            for b in self.bottles:
                if b["id"] == self.picked_bottle:
                    b.update({"distance": 0.0, "angle": 1.4})
                    break

        elif step == "on_prep_with_wrong_drink_glass":
            self.glass_in_left_hand = True
            self.glass.update({
                "distance": 0.0, "angle": 0.0, "state": True,
                "drink_type": float(self._get_wrong_drink_type(self.client_preference)),
                "was_used": False,
                "is_shaken": False,
            })

        elif step == "holding_wrong_drink_at_serv":
            self.robot_position = 0.95
            self.glass_in_left_hand = True
            self.glass.update({
                "distance": 0.0, "angle": 0.0, "state": True,
                "drink_type": float(self._get_wrong_drink_type(self.client_preference)),
                "was_used": False,
                "is_shaken": False,
            })

        elif step == "at_serv_with_used_glass":
            self.robot_position = 0.95
            self.glass_in_left_hand = True
            self.bottle_in_right_hand = bool(self.rng.integers(0, 2))
            if self.bottle_in_right_hand:
                self.picked_bottle = int(self.rng.choice(self._get_valid_bottle_ids()))
            # place used glass at a random serving position
            dist, ang = self.random_position(self.serv_area)
            self.glass.update({"distance": dist, "angle": ang, "state": False, "was_used": True, "is_shaken": False})

        elif step == "at_serv_with_correct_drink_and_bottle":
            self.robot_position = 0.95
            self.glass_in_left_hand = True
            self.bottle_in_right_hand = bool(self.rng.integers(0, 2))
            if self.bottle_in_right_hand:
                self.picked_bottle = int(self.client_preference) if self.client_preference > 0 else 1
            # place glass at a random serving position (still considered in-hand)
            dist, ang = self.random_position(self.serv_area)
            self.glass.update({
                "distance": dist, "angle": ang, "state": True,
                "drink_type": float(self.client_preference), "was_used": False, "is_shaken": False,
            })

        elif step == "at_prep_with_correct_drink_glass":
            self.glass_in_left_hand = True
            self.glass.update({
                "distance": 0.0, "angle": 0.0, "state": True,
                "drink_type": float(self.client_preference), "was_used": False, "is_shaken": False,
            })

        elif step == "at_serv_with_correct_drink":
            self.robot_position = 0.95
            self.glass_in_left_hand = True
            # place glass at a random serving position (still considered in-hand)
            dist, ang = self.random_position(self.serv_area)
            self.glass.update({
                "distance": dist, "angle": ang, "state": True,
                "drink_type": float(self.client_preference), "was_used": False, "is_shaken": False,
            })

        elif step == "at_prep_with_already_used_glass":
            self.glass_in_left_hand = True
            self.glass.update({
                "distance": 0.0, "angle": 0.0, "state": False,
                "drink_type": float(self.client_preference), "was_used": True,
            })

        elif step == "at_prep_with_correct_glass_and_bottle":
            self.glass_in_left_hand = True
            self.bottle_in_right_hand = True
            self.picked_bottle = int(self.client_preference) if self.client_preference > 0 else 1
            for b in self.bottles:
                if b["id"] == self.picked_bottle:
                    b.update({"distance": 0.0, "angle": 1.4})
                    break
            self.glass.update({
                "distance": 0.0, "angle": 0.0,
                "state": False, "drink_type": 0.0, "was_used": False,
            })

    # ------------------------------------------------------------------ #
    # Policies
    # ------------------------------------------------------------------ #

    def shake_glass_policy(self):
        """Shake the glass to mix the drink."""
        # Require: glass in hand, glass exists, glass has a drink (state==True),
        # glass not already used, and not already shaken.
        if (
            not self.glass_in_left_hand
            or not self.glass
            or not self.glass.get("state", False)
            or self.glass.get("was_used", False)
            or self.glass.get("is_shaken", False)
        ):
            return
        self.glass.update({"is_shaken": True})

    def pick_glass_policy(self):
        """Pick glass from prep or serving table."""
        if self.glass_in_left_hand:
            return

        if self.is_at_preparation_table() and self.glass_is_in_preparation_area():
            self.glass_in_left_hand = True
            self.glass.update({"distance": 0.5, "angle": 0.0})

        elif self.is_at_serving_table() and self.glass_is_in_serving_position():
            self.glass_in_left_hand = True
            self.glass.update({"distance": 0.5, "angle": 0.0})

    def place_glass_policy(self):
        """
        Place glass in context-appropriate location.
        - At serving table: places glass; marks served if drink matches preference.
        - At prep table: places glass AND cleans it (state=False, drink_type=0).
          This is the only way to clear a wrong drink — bring it back to prep.
        """
        if not self.glass_in_left_hand:
            return

        if self.is_at_serving_table():
            self.glass_in_left_hand = False
            # move glass to serving position
            self.glass.update({
                "distance": self.serving_pos["distance"],
                "angle": self.serving_pos["angle"],
            })
            # Serving event: only triggers if drink matches preference
            if self.glass["state"] and self._is_drink_matching_preference():
                # latch whether the drink that is being served was shaken
                self._served_shake_latch = bool(self.glass.get("is_shaken", False))
                self.correct_drink_served = True
                # Now clean the glass state for the physical object
                self.glass.update({"was_used": True, "state": False, "drink_type": 0.0, "is_shaken": False})

        elif self.is_at_preparation_table():
            # Placing at prep = cleaning the glass
            self.glass_in_left_hand = False
            self.glass.update({
                "distance": self.original_glass_pos["distance"],
                "angle": self.original_glass_pos["angle"],
                "state": False,
                "drink_type": 0.0,
                "is_shaken": False,
                # was_used stays unchanged: cleaned but still was_used if it was before
            })

    def prepare_drink_policy(self):
        """Prepare drink if holding glass (empty, unused) and bottle."""
        if not self.glass_in_left_hand or not self.bottle_in_right_hand:
            return
        if not self.glass or self.glass["state"] or self.glass["was_used"]:
            return
        if not self.picked_bottle or self.picked_bottle <= 0:
            return
        # When preparing a fresh drink, ensure the shaken flag is cleared.
        self.glass.update({"state": True, "drink_type": float(self.picked_bottle), "is_shaken": False})

    def change_position_policy(self):
        """Instant teleport between prep and serving tables."""
        if self.is_at_preparation_table():
            self.robot_position = 0.95
        else:
            self.robot_position = 0.0

    def pick_bottle_policy(self):
        """Pick bottle by agent choice, fallback to client preference, then first available."""
        if self.bottle_in_right_hand or not self.is_at_preparation_table():
            return

        valid_ids = self._get_valid_bottle_ids()
        bottle_id = int(self.agent_bottle_choice) if self.agent_bottle_choice is not None else 0

        if bottle_id == 0 and self.client_preference > 0:
            bottle_id = int(self.client_preference)
        if bottle_id not in valid_ids and valid_ids:
            bottle_id = valid_ids[0]
        if bottle_id == 0:
            return

        self.bottle_in_right_hand = True
        self.picked_bottle = bottle_id
        for b in self.bottles:
            if b["id"] == bottle_id:
                b.update({"distance": 0.0, "angle": 1.4})
                break

    def place_bottle_policy(self):
        """Place the held bottle back to a random position in the prep area."""
        if not self.is_at_preparation_table() or not self.bottle_in_right_hand:
            return
        for b in self.bottles:
            if b["id"] == self.picked_bottle:
                b["distance"], b["angle"] = self.random_position(self.prep_area)
                break
        self.bottle_in_right_hand = False
        self.picked_bottle = 0

    def ask_nicely_policy(self):
        """
        Ask client for preference.
        Preferences are stable per client but NOT trivially equal to client_id.
        Revealed only when this policy is called.
        """
        cid = int(self.client_id)
        if cid not in self.know_preference:
            # Deterministic per (seed, client_id) so preference is consistent across episodes
            local_rng = numpy.random.default_rng(self.random_seed ^ (cid * 0xDEAD))
            pref = int(local_rng.choice(self._get_valid_bottle_ids()))
            self.know_preference[cid] = pref
        self.client_preference = self.know_preference[cid]

    # ------------------------------------------------------------------ #
    # Loop detection
    # ------------------------------------------------------------------ #

    def _is_policy_loop(self):
        """Detect meaningless policy loops: A→A, A→B→A, A→B→A→B."""
        if self.last_policy_executed is None:
            return False
        # Immediate repetition
        if self.prev_policy_executed == self.last_policy_executed:
            return True
        seq = self.policy_sequence
        # Alternating: A→B→A
        if len(seq) >= 3 and seq[-3] == seq[-1] and seq[-3] != seq[-2]:
            return True
        # Repeated block: A→B→A→B
        if len(seq) >= 4 and tuple(seq[-2:]) == tuple(seq[-4:-2]):
            return True
        return False

    # ------------------------------------------------------------------ #
    # Goals / Rewards
    # ------------------------------------------------------------------ #

    def get_progress_goal(self):
        """
        Shaped reward reflecting task progress.
        Ordered strictly by achievement level; no perverse incentives.
        """
        has_glass = self.glass_in_left_hand
        has_bottle = self.bottle_in_right_hand
        glass_state = self.glass["state"] if self.glass else False
        was_used = self.glass["was_used"] if self.glass else False
        drink_ok = self._is_drink_matching_preference()
        at_prep = self.is_at_preparation_table()
        at_serv = self.is_at_serving_table()

        g_dist = self.glass["distance"] if self.glass else 0.0
        g_ang = self.glass["angle"] if self.glass else 0.0
        glass_at_serving = (
            abs(g_dist - self.serving_pos["distance"]) < 0.1 and
            abs(g_ang - self.serving_pos["angle"]) < 0.1
        )
        glass_at_original = (
            abs(g_dist - self.original_glass_pos["distance"]) < 0.1 and
            abs(g_ang - self.original_glass_pos["angle"]) < 0.1
        )

        if glass_at_original and not has_glass and was_used:
            current_step = 1.0
        elif at_prep and has_glass and not glass_state and was_used:
            current_step = 0.9
        elif has_glass and not glass_state and was_used:
            current_step = 0.85
        elif glass_at_serving and not glass_state and not has_glass:
            current_step = 0.8
        elif glass_at_serving and glass_state and not has_glass and drink_ok:
            current_step = 0.6
        elif at_serv and has_glass and glass_state and drink_ok:
            current_step = 0.5
        elif at_prep and has_glass and glass_state and drink_ok:
            current_step = 0.4
        elif has_glass and has_bottle and at_prep and not glass_state:
            # Ready to prepare: glass empty + holding bottle
            current_step = 0.25
        elif at_prep and has_glass and glass_state and not drink_ok:
            # Wrong drink: below 0.25 to incentivise going back and cleaning
            current_step = 0.15
        elif has_glass and at_prep:
            current_step = 0.15
        elif has_bottle and at_prep:
            current_step = 0.1
        elif at_prep:
            current_step = 0.05
        else:
            current_step = 0.0

        reward = current_step
        if self._is_policy_loop():
            self.sequence_repeat_count += 1
            reward = 0.0
        else:
            self.sequence_repeat_count = 0

        self.last_step = current_step
        return float(reward)

    def get_serve_the_drink_goal(self):
        """Reward = 1.0 once per episode when correct drink is served."""
        if self._is_policy_loop():
            return 0.0
        if self.serve_reward_consumed:
            return 0.0
        # Use served latch (captured at place_glass time) rather than current glass.is_shaken
        if (
            self.glass_is_in_serving_position()
            and self.glass
            and self.correct_drink_served
            and not self.glass_in_left_hand
            and (self.client_likes_shake == self._served_shake_latch)
        ):
            self.serve_reward_consumed = True
            return 1.0
        return 0.0

    def get_return_the_glass_goal(self):
        """Reward = 1.0 once per episode when used glass returns to prep."""
        if self._is_policy_loop():
            return 0.0
        if self.return_reward_consumed:
            return 0.0
        if (
            self.glass_is_in_preparation_area() and
            self.glass["was_used"] and
            not self.glass_in_left_hand
        ):
            self.return_reward_consumed = True
            return 1.0
        return 0.0

    # ------------------------------------------------------------------ #
    # Policy execution
    # ------------------------------------------------------------------ #

    def execute_policy(self, policy_name):
        """Execute a policy by name and update tracking state."""
        method = getattr(self, policy_name + "_policy", None)
        if method and callable(method):
            self.prev_policy_executed = self.last_policy_executed
            self.last_policy_executed = policy_name
            self.policy_sequence.append(policy_name)
            if len(self.policy_sequence) > 10:
                self.policy_sequence.pop(0)
            method()
            return True
        return False


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
        self.change_reward_iterations = {}

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

        self.agent_bottle_subscription = self.create_subscription(
            Float32,
            "cognitive_node/world_model/last_bottle",
            self.agent_bottle_callback,
            1,
        )

    def agent_bottle_callback(self, msg):
        self.simulator.set_agent_bottle_choice(float(msg.data))

    # ------------------------------------------------------------------ #
    # Perception updates
    # ------------------------------------------------------------------ #

    def perceive_bottles(self):
        self.perceptions["bottles"].data = []
        for b in self.simulator.get_bottles_state():
            msg = self.base_messages["bottles"]()
            msg.distance = float(b["distance"])
            msg.angle = float(b["angle"])
            if hasattr(msg, "id"):
                msg.id = int(b["id"])
            self.perceptions["bottles"].data.append(msg)
        if not self.perceptions["bottles"].data:
            self.perceptions["bottles"].data.append(self.base_messages["bottles"]())

    def perceive_glass(self):
        self.perceptions["glass"].data = []
        msg = self.base_messages["glass"]()
        gs = self.simulator.get_glass_state()
        msg.distance = float(gs["distance"])
        msg.angle = float(gs["angle"])
        msg.state = bool(gs["state"])
        msg.drink_type = float(gs["drink_type"])
        msg.was_used = bool(gs["was_used"])
        msg.is_shaken = bool(gs["is_shaken"])
        self.perceptions["glass"].data.append(msg)

    def update_perceptions_from_simulator(self):
        self.perceive_bottles()
        self.perceive_glass()
        self.perceptions["robot_position"].data = float(self.simulator.robot_position)
        self.perceptions["glass_in_left_hand"].data = bool(self.simulator.glass_in_left_hand)
        self.perceptions["bottle_in_right_hand"].data = bool(self.simulator.bottle_in_right_hand)

        selected_bottle = self.simulator.get_selected_bottle_state()
        if "last_bottle" in self.perceptions:
            perception = self.perceptions["last_bottle"]
            if hasattr(perception, "id"):
                if selected_bottle is not None:
                    perception.id = int(selected_bottle["id"])
                    perception.distance = float(selected_bottle["distance"])
                    perception.angle = float(selected_bottle["angle"])
                    # Match the simulator's polar convention: angle = atan2(x, y)
                    perception.x = float(selected_bottle["distance"] * math.sin(selected_bottle["angle"]))
                    perception.y = float(selected_bottle["distance"] * math.cos(selected_bottle["angle"]))
                else:
                    perception.id = -1
                    perception.distance = -1.0
                    perception.angle = -1.0
                    perception.x = -1.0
                    perception.y = -1.0
            else:
                perception.data = int(selected_bottle["id"] if selected_bottle else -1)

        if "last_bottle_position" in self.perceptions:
            self.perceptions["last_bottle_position"].data = float(selected_bottle["distance"] if selected_bottle else -1.0)

        # --- Stage perception ---
        if "stage" in self.perceptions:
            # Determinar la etapa actual según el número de iteración
            current_stage = None
            if self.change_reward_iterations:
                # Asume que stages están ordenadas por iteración ascendente
                sorted_stages = sorted(self.change_reward_iterations.items(), key=lambda x: x[1])
                for stage_name, stage_iter in sorted_stages:
                    if self.simulator.iteration >= stage_iter:
                        current_stage = stage_name
                    else:
                        break
            # Codifica la etapa como un float (índice normalizado)
            if current_stage is not None:
                stage_names = list(self.change_reward_iterations.keys())
                idx = stage_names.index(current_stage)
                self.perceptions["stage"].data = float(idx) / max(1, len(stage_names)-1)
            else:
                self.perceptions["stage"].data = 0.0

        if "at_serv_with_glass" in self.perceptions:
            self.perceptions["at_serv_with_glass"].data = bool(
                self.simulator.is_at_serving_table() and self.simulator.glass_in_left_hand and not self.simulator.glass["was_used"] and self.simulator.glass["state"]
            )

        if "at_prep_with_used_glass" in self.perceptions:
            self.perceptions["at_prep_with_used_glass"].data = bool(
                self.simulator.is_at_preparation_table()
                and self.simulator.glass_in_left_hand
                and self.simulator.glass["was_used"]
            )

        if "glass_at_serving" in self.perceptions:
            self.perceptions["glass_at_serving"].data = bool(
                self.simulator.glass_is_in_serving_position()
            )

        # --- History features: help pnodes break perceptual symmetry ----------
        if "policy_pair" in self.perceptions:
            self.perceptions["policy_pair"].data = float(
                self.simulator.get_policy_pair_encoding()
            )

        # Backward-compatible legacy signals (deprecated): keep discrete mapping.
        if "last_policy" in self.perceptions:
            n_policies = len(BartenderSim.POLICIES)
            last_idx = BartenderSim.POLICY_TO_INDEX.get(
                self.simulator.last_policy_executed or "none", 0
            )
            self.perceptions["last_policy"].data = float(last_idx) / float(n_policies - 1)

        if "prev_policy" in self.perceptions:
            n_policies = len(BartenderSim.POLICIES)
            prev_idx = BartenderSim.POLICY_TO_INDEX.get(
                self.simulator.prev_policy_executed or "none", 0
            )
            self.perceptions["prev_policy"].data = float(prev_idx) / float(n_policies - 1)

        if "preference_known" in self.perceptions:
            self.perceptions["preference_known"].data = bool(
                self.simulator.is_client_preference_known()
            )

        if "drink_matches_preference" in self.perceptions:
            self.perceptions["drink_matches_preference"].data = bool(
                self.simulator._is_drink_matching_preference()
            )
        # -----------------------------------------------------------------------

        self.perceptions["client"].data = []
        client_msg = self.base_messages["client"]()
        client_msg.id = int(self.simulator.client_id)
        client_msg.preference = int(self.simulator.client_preference)
        client_msg.likes_shake = bool(self.simulator.client_likes_shake)
        self.perceptions["client"].data.append(client_msg)

    # ------------------------------------------------------------------ #
    # World control
    # ------------------------------------------------------------------ #

    def reset_world(self, data=None):
        self.get_logger().info("Resetting world...")
        self.simulator.reset_world()
        self.update_perceptions_from_simulator()
        self.update_reward_sensor()
        self.publish_perceptions()

    def _return_reward_enabled(self):
        """Enable return-the-glass reward from stage1 onwards."""
        stage_start = self.change_reward_iterations.get("stage1")
        if stage_start is None:
            stage_start = self.change_reward_iterations.get("stage0")
        if stage_start is None:
            # Backward compatible: if no stages are configured, keep reward enabled.
            return True
        return self.simulator.iteration >= int(stage_start)

    def _is_stage0(self):
        """Stage0 is the phase before return-the-glass reward is enabled."""
        return not self._return_reward_enabled()

    def update_reward_sensor(self):
        if "progress_goal" in self.perceptions:
            progress = self.simulator.get_progress_goal()
            self.perceptions["progress_goal"].data = progress
            self.get_logger().info(f"Progress reward: {progress}")
        if "serve_the_drink_goal" in self.perceptions:
            self.perceptions["serve_the_drink_goal"].data = self.simulator.get_serve_the_drink_goal()
        if "return_the_glass_goal" in self.perceptions:
            if self._return_reward_enabled():
                self.perceptions["return_the_glass_goal"].data = self.simulator.get_return_the_glass_goal()
            else:
                self.perceptions["return_the_glass_goal"].data = 0.0

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
    # ROS callbacks
    # ------------------------------------------------------------------ #

    def world_reset_service_callback(self, request, response):
        self.reset_world(request)
        response.success = True
        return response

    def new_command_callback(self, data):
        self.get_logger().debug(f"Command received... ITERATION: {data.iteration}")
        self.simulator.iteration = data.iteration
        self.update_reward_sensor()
        if data.command == "reset_world":
            self.reset_world(data)
        elif data.command == "end":
            self.get_logger().info("Ending simulator as requested by LTM...")
            rclpy.shutdown()

    def new_action_service_callback(self, request, response):
        self.get_logger().info(f"Executing policy {request.policy}")
        self.get_logger().info(f"ITERATION: {self.simulator.iteration}")

        self.update_perceptions_from_simulator()
        self.get_logger().info(f"PERCEPTIONS BEFORE: {self.perceptions}")

        self.simulator.execute_policy(request.policy)

        self.update_perceptions_from_simulator()
        self.get_logger().info(f"PERCEPTIONS AFTER: {self.perceptions}")

        self.update_reward_sensor()

        # In stage0, serving the drink ends the episode immediately.
        if (
            self._is_stage0()
            and "serve_the_drink_goal" in self.perceptions
            and float(self.perceptions["serve_the_drink_goal"].data) >= 1.0
        ):
            self.get_logger().info("Stage0: serve achieved, resetting world.")
            self.reset_world()
            response.success = True
            return response

        self.publish_perceptions()
        response.success = True
        return response

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

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

        # Si no está definida la percepción 'stage', la agrega como Float
        if "stage" not in self.perceptions:
            self.perceptions["stage"] = Float32()
            self.perceptions["stage"].data = 0.0
            self.get_logger().info("Publishing to: /stage (auto-added)")
            self.sim_publishers["stage"] = self.create_publisher(Float32, "/stage", 0)

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

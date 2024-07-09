from enum import Enum
import importlib
import os.path
import math
import yaml
import yamlloader
import numpy
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor
from core.service_client import ServiceClient

from core_interfaces.srv import LoadConfig
from simulators_interfaces.srv import CalculateClosestPosition, ObjectPickableWithTwoHands, ObjectTooFar
from core.utils import class_from_classname

class World(Enum):
    """Worlds to be simulated."""
    GRIPPER_AND_LOW_FRICTION = 1
    NO_GRIPPER_AND_HIGH_FRICTION = 2
    GRIPPER_AND_LOW_FRICTION_TWO_BOXES = 3
    KITCHEN = 4
    GRIPPER_AND_LOW_FRICTION_SHORT_ARM = 5
    GRIPPER_AND_LOW_FRICTION_DAMAGED_SERVO = 6
    GRIPPER_AND_LOW_FRICTION_OBSTACLE = 7

class Item(Enum):
    """Types of objects."""

    CYLINDER = 1
    BOX = 2
    SKILLET = 3
    GRAPES = 4
    APPLE = 5
    TOMATO = 6
    ORANGE = 7
    CARROT = 8
    LETTUCE = 9
    PINEAPPLE = 10
    WATERMELON = 11
    EGGPLANT = 12
    BANANA = 13

class LTMSim(Node):
    def __init__(self):
        """Init attributes when a new object is created."""
        super().__init__("LTMSim")
        self.rng = None
        self.ident = None
        self.last_reset_iteration = 0
        self.world = None
        self.base_messages = {}
        self.perceptions = {}
        self.sim_publishers = {}
        self.catched_object = None
        self.random_seed = self.declare_parameter('random_seed', value = 0).get_parameter_value().integer_value
        self.config_file = self.declare_parameter('config_file', descriptor=ParameterDescriptor(dynamic_typing=True)).get_parameter_value().string_value
        self.normal_inner = numpy.poly1d(
        numpy.polyfit([0.0, 0.3925, 0.785, 1.1775, 1.57], [0.45, 0.47, 0.525, 0.65, 0.9], 3)
        )
        self.normal_outer = numpy.poly1d(
            numpy.polyfit([0.0, 0.3925, 0.785, 1.1775, 1.57], [1.15, 1.25, 1.325, 1.375, 1.375], 3)
        )
        self.short_outer = numpy.poly1d(
            numpy.polyfit(
                [0.0, 0.3925, 0.785, 1.1775, 1.57],
                [1.15 * 0.8, 1.25 * 0.8, 1.325 * 0.8, 1.375 * 0.8, 1.375 * 0.8],
                3,
            )
        )

        self.cbgroup_server=MutuallyExclusiveCallbackGroup()
        self.cbgroup_client=MutuallyExclusiveCallbackGroup()

        self.calculate_closest_position_service = self.create_service(
            CalculateClosestPosition,
            "simulator/calculate_closest_position",
            self.calculate_closest_position_callback, 
            callback_group= self.cbgroup_server
        )

        self.object_pickable_with_two_hands_service = self.create_service(
            ObjectPickableWithTwoHands,
            "simulator/object_pickable_with_two_hands",
            self.object_pickable_with_two_hands_callback,
            callback_group= self.cbgroup_server
        )

        self.object_too_far_service = self.create_service(
            ObjectTooFar,
            "simulator/object_too_far",
            self.object_too_far_callback,
            callback_group= self.cbgroup_server
        )

        self.load_client=ServiceClient(LoadConfig, 'commander/load_experiment')

        

    def load_experiment_file_in_commander(self):
        loaded = self.load_client.send_request(file = self.config_file)
        return loaded

    def calculate_closest_position_callback(self, request, response):
        """
        The callback to calculate the closest feasible position with a given object angle

        :param request: The Request that includes the angle to the object
        :type request: simulators_interfaces.srv.CalculateClosestPosition_Request
        :param response: The Response that includes the closest distance and angle
        :type response: simulators_interfaces.srv.CalculateClosestPosition_Response
        :return: The Response that includes the closest distance and angle
        :rtype: simulators_interfaces.srv.CalculateClosestPosition_Response
        """
        ang = request.angle
        dist_near, ang_near = self.calculate_closest_position(ang)
        response.dist_near = dist_near
        response.ang_near = ang_near
        return response
    
    def calculate_closest_position(self, ang):
        """
        Calculate the closest feasible position for an object taking into account the angle.

        :param ang: The angle to the object
        :type ang: float
        :return: The closest distance and angle
        :rtype: float, float
        """
        dist = self.normal_inner(abs(ang))
        y_coord = dist * math.sin(ang)
        x_coord = dist * math.cos(ang)
        if y_coord < -0.38:
            y_coord = -0.38
            x_coord = 0.35
        elif y_coord > 0.38:
            y_coord = 0.38
            x_coord = 0.35
        new_dist = 0.01 + numpy.linalg.norm([y_coord, x_coord])
        new_ang = numpy.arctan2(y_coord, x_coord)
        return new_dist, new_ang
    
    def object_pickable_with_two_hands_callback(self, request, response):
        """
        The callback to calculate if an object can be pickable with two hands

        :param request: The Request that includes the distance and angle to the object
        :type request: simulators_interfaces.srv.ObjectPickableWithTwoHands_Request
        :param response: The Response that indicates if the object is pickable or not
        :type response: simulators_interfaces.srv.ObjectPickableWithTwoHands_Response
        :return: The Response that indicates if the object is pickable or not
        :rtype: simulators_interfaces.srv.ObjectPickableWithTwoHands_Response
        """
        dist = request.distance
        ang = request.angle
        pickable = self.object_pickable_with_two_hands(dist, ang)
        response.pickable = pickable
        return response

    @staticmethod
    def object_pickable_with_two_hands(dist, ang):
        """
        Return True if the object is in a place where it can be picked with two hands.

        :param dist: The distance to the object
        :type dist: float
        :param ang: The angle to the object
        :type ang: float
        :return: Value that indicates if the object is pickable or not
        :rtype: bool
        """
        return abs(ang) <= 0.3925 and 0.46 <= dist <= 0.75
    
    def object_too_far_callback(self, request, response):
        """
        The callback to calculate if and object is out of range of the robot

        :param request: The Request that includes the distance and angle to the object
        :type request: simulators_interfaces.srv.ObjectTooFar_Request
        :param response: The Response that indicates if the object if out of the range or not
        :type response: simulators_interfaces.srv.ObjectTooFar_Response
        :return: The Response that indicates if the object if out of the range or not
        :rtype: simulators_interfaces.srv.ObjectTooFar_Response
        """
        dist = request.distance
        ang = request.angle
        too_far = self.object_too_far(dist, ang)
        response.too_far = too_far
        return response
    
    
    def object_too_far(self, dist, ang):
        """
        Return True if the object is out of range of the robot.

        :param dist: The distance to the object
        :type dist: float
        :param ang: The angle to the object
        :type ang: float
        :return: Value that indicates if the object is out of range or not
        :rtype: bool
        """
        if self.world.name == World.GRIPPER_AND_LOW_FRICTION_SHORT_ARM.name:
            too_far = dist > self.short_outer(abs(ang))
        elif self.world.name == World.GRIPPER_AND_LOW_FRICTION_OBSTACLE.name:
            if abs(ang) < 0.17:
                too_far = dist > self.short_outer(abs(ang))
            else:
                too_far = dist > self.normal_outer(abs(ang))
        elif self.world.name == World.GRIPPER_AND_LOW_FRICTION_DAMAGED_SERVO.name:
            too_far = (dist > self.normal_outer(abs(ang))) or (
                ang > (0.8 * numpy.arctan2(1.07, 0.37))
            )
        else:
            too_far = dist > self.normal_outer(abs(ang))
        return bool(too_far)

    def object_too_close(self, dist, ang):
        """
        Return True if the object is too close to the robot to be caught.

        :param dist: The distance to the object
        :type dist: float
        :param ang: The angle to the object
        :type ang: float
        :return: Value that indicates if the object is too close or not
        :rtype: bool
        """
        return dist < self.normal_inner(abs(ang))
    
    @staticmethod
    def object_outside_table(dist, ang):
        """
        Return True if the object is outside the table. This is used in some scripts...

        :param dist: The distance to the object
        :type dist: float
        :param ang: The angle to the object
        :type ang: float
        :return: Value that indicates if the object is outside of the table or not
        :rtype: bool
        """
        object_y = numpy.sin(ang) * dist
        object_x = numpy.cos(ang) * dist
        return not (-1.07 <= object_y <= 1.07 and 0.37 <= object_x <= 1.27)
    
    @staticmethod
    def object_is_small(rad):
        """
        Return True if the ball is small, False if it is big. Right now, small is 0.03 and big 0.07.

        :param rad: Radius of the ball
        :type rad: float
        :return: The value that indicates if the ball is small or not
        :rtype: bool
        """
        return rad <= 0.05
    
    @staticmethod
    def send_object_twohandsreachable(dist):
        """
        Calculate the coordinates of the object when moving it to a place where it can be picked with two hands.

        :param dist: The distance to the object
        :type dist: float
        :return: The calculated coordinates
        :rtype: float, float
        """
        x_coord = 0
        y_coord = dist
        if y_coord < 0.46:
            y_coord = 0.46
        elif y_coord > 0.75:
            y_coord = 0.75
        new_dist = numpy.linalg.norm([y_coord, x_coord])
        return new_dist, 0.0
    
    def send_object_outofreach(self, ang):
        """
        Calculate the coordinates of the object when moving it out of reach.

        :param dist: The angle to the object
        :type dist: float
        :return: The calculated coordinates
        :rtype: float, float
        """
        dist = self.normal_outer(abs(ang))
        y_coord = dist * math.sin(ang)
        x_coord = dist * math.cos(ang)
        if y_coord < -1.07:
            y_coord = -1.07
            x_coord = 0.84
        elif y_coord > 1.07:
            y_coord = 1.07
            x_coord = 0.84
        new_dist = 0.01 + numpy.linalg.norm([y_coord, x_coord])
        new_ang = numpy.arctan2(y_coord, x_coord)
        return new_dist, new_ang
    
    def object_in_close_box(self, dist, ang):
        """
        Check if there is an object inside of a box.

        :param dist: The distance to the object
        :type dist: float
        :param ang: The angle to the object
        :type ang: float
        :return: The value that indicates if the object is inside of the box or not
        :rtype: bool
        """
        inside = False
        for box in self.perceptions["boxes"].data:
            if not self.object_too_far(box.distance, box.angle):
                inside = (abs(box.distance - dist) < 0.05) and (abs(box.angle - ang) < 0.05)
                if inside:
                    break
        return inside
    
    def object_in_far_box(self, dist, ang):
        """
        Check if there is an object inside of a box.

        :param dist: The distance to the object
        :type dist: float
        :param ang: The angle to the object
        :type ang: float
        :return: The value that indicates if the object is inside of the box or not
        :rtype: bool
        """
        inside = False
        for box in self.perceptions["boxes"].data:
            if self.object_too_far(box.distance, box.angle):
                inside = (abs(box.distance - dist) < 0.05) and (abs(box.angle - ang) < 0.05)
                if inside:
                    break
        return inside
    
    def object_with_robot(self, dist, ang):
        """
        Check if there is an object adjacent to the robot.

        :param dist: The distance to the object
        :type dist: float
        :param ang: The angle to the object
        :type ang: float
        :return: The value that indicates if the object is adjacent or not
        :rtype: bool
        """
        together = False
        if (not self.perceptions["ball_in_left_hand"].data) and (
            not self.perceptions["ball_in_right_hand"].data
        ):
            dist_near, ang_near = self.calculate_closest_position(ang)
            together = (abs(dist - dist_near) < 0.05) and (abs(ang - ang_near) < 0.05)
        return together
    
    def avoid_reward_by_chance(self, distance, angle):
        """
        Avoid a reward situation obtained by chance.

        :param distance: The distance to the object
        :type distance: float
        :param angle: The angle to the object
        :type angle: float
        :return: The distance value to avoid the reward
        :rtype: float
        """
        # This is necessary so sweep never puts the object close to the robot or colliding with a box.
        # This is not realistic, it needs to be improved.
        while (
            self.object_with_robot(distance, angle)
            or self.object_in_close_box(distance, angle)
            or self.object_in_far_box(distance, angle)
        ):
            if distance > 0.65:
                distance -= 0.10
            else:
                distance += 0.10
        return distance
    
    def reward_ball_in_box(self):
        """
        Reward for object in box goal.

        :return: True if there is reward or False if not
        :rtype: bool
        """
        self.perceptions["ball_in_box"].data = False
        for cylinder in self.perceptions["cylinders"].data:
            for box in self.perceptions["boxes"].data:
                if (cylinder.distance == box.distance) and (cylinder.angle == box.angle):
                    self.perceptions["ball_in_box"].data = True
                    return True
        return False
    

    def reward_ball_with_robot(self):
        """
        Reward for object with robot goal.

        :return: True if there is reward or False if not
        :rtype: bool
        """
        self.perceptions["ball_with_robot"].data = False
        for cylinder in self.perceptions["cylinders"].data:
            dist, ang = self.calculate_closest_position(cylinder.angle)
            if (cylinder.distance == dist) and (cylinder.angle == ang):
                self.perceptions["ball_with_robot"].data = True
                return True
        return False
    
    def reward_clean_area(self):
        """
        Reward for cleaning the table goal.

        :return: True if there is reward or False if not
        :rtype: bool
        """
        self.perceptions["clean_area"].data = False
        for cylinder in self.perceptions["cylinders"].data:
            for box in self.perceptions["boxes"].data:
                if (
                    (cylinder.distance == box.distance)
                    and (cylinder.angle == box.angle)
                    and self.object_too_far(box.distance, box.angle)
                ):
                    self.perceptions["clean_area"].data = True
                    return True
        return False

    def update_reward_sensor(self):
        """Update goal sensors' values."""
        for sensor in self.perceptions:
            reward_method = getattr(self, "reward_" + sensor, None)
            if callable(reward_method):
                reward_method()

    
    def random_position(self, in_valid=True, out_valid=True):
        """
        Return a random position in the table.

        :param in_valid: True if the position can be generated close to the robot
        :type in_valid: bool
        :param out_valid: True if the position can be generated away to the robot
        :type out_valid: bool
        :return: The distance and angle to the random position generated
        :rtype: float, float
        """
        #TODO: Solve the ugly hacks to test curriculum learning and check the Sphinx description
        valid = False
        while not valid:
            object_y = self.rng.uniform(low=-1.07, high=1.07)
            object_x = self.rng.uniform(low=0.37, high=1.27)
            distance = numpy.linalg.norm([object_y, object_x])
            angle = numpy.arctan2(object_y, object_x)
            valid = not self.object_too_close(distance, angle)
            # TODO: Ugly hacks to test curriculum learning
            if self.last_reset_iteration >= 6000 and self.last_reset_iteration < 8000:
                # if self.world == World.GRIPPER_AND_LOW_FRICTION_SHORT_ARM:
                #     max_distance = self.normal_outer(abs(angle))
                #     min_distance = self.short_outer(abs(angle))
                #     distance = self.rng.uniform(low=min_distance, high=max_distance)
                if self.world == World.GRIPPER_AND_LOW_FRICTION_DAMAGED_SERVO:
                    max_angle = numpy.arctan2(1.07, 0.37)
                    min_angle = 0.8 * numpy.arctan2(1.07, 0.37)
                    if angle < min_angle:
                        angle = self.rng.uniform(low=min_angle, high=max_angle)
                elif self.world == World.GRIPPER_AND_LOW_FRICTION_OBSTACLE:
                    angle = self.rng.uniform(low=-0.17, high=0.17)
                    max_distance = self.normal_outer(abs(angle))
                    min_distance = self.short_outer(abs(angle))
                    distance = self.rng.uniform(low=min_distance, high=max_distance)
            if not in_valid:
                valid = valid and self.object_too_far(distance, angle)
            if not out_valid:
                valid = valid and not self.object_too_far(distance, angle)
            if valid:
                for box in self.perceptions["boxes"].data:
                    if (abs(box.distance - distance) < 0.15) and (abs(box.angle - angle) < 0.15):
                        valid = False
                        break
            if valid:
                for cylinder in self.perceptions["cylinders"].data:
                    if (abs(cylinder.distance - distance) < 0.15) and (
                        abs(cylinder.angle - angle) < 0.15
                    ):
                        valid = False
                        break
        return distance, angle
    

    def random_perceptions(self):
        """Randomize the state of the environment."""
        # Objects
        self.catched_object = None
        if self.world in [
            World.GRIPPER_AND_LOW_FRICTION,
            World.NO_GRIPPER_AND_HIGH_FRICTION,
            World.GRIPPER_AND_LOW_FRICTION_SHORT_ARM,
            World.GRIPPER_AND_LOW_FRICTION_OBSTACLE,
            World.GRIPPER_AND_LOW_FRICTION_DAMAGED_SERVO,
        ]:
            self.perceptions["boxes"].data = []
            distance, angle = self.random_position(in_valid=True, out_valid=True)
            self.perceptions["boxes"].data.append(self.base_messages["boxes"]())
            self.perceptions["boxes"].data[0].distance = distance
            self.perceptions["boxes"].data[0].angle = angle
            self.perceptions["boxes"].data[0].diameter = 0.12
            # self.perceptions["boxes"].data[0].id = Item.box.value
            self.perceptions["cylinders"].data = []
            distance, angle = self.random_position(in_valid=True, out_valid=True)
            self.perceptions["cylinders"].data.append(self.base_messages["cylinders"]())
            self.perceptions["cylinders"].data[0].distance = distance
            self.perceptions["cylinders"].data[0].angle = angle
            if self.rng.uniform() > 0.5:
                self.perceptions["cylinders"].data[0].diameter = 0.03
            else:
                self.perceptions["cylinders"].data[0].diameter = 0.07
            # self.perceptions["cylinders"].data[0].id = Item.cylinder.value
            self.perceptions["ball_in_left_hand"].data = False
            self.perceptions["ball_in_right_hand"].data = False 
            object_distance = self.perceptions["cylinders"].data[0].distance
            object_angle = self.perceptions["cylinders"].data[0].angle
            if (
                (World.GRIPPER_AND_LOW_FRICTION.name in self.world.name)
                and self.object_is_small(object_distance)
                and (not self.object_too_far(object_distance, object_angle))
                and (self.rng.uniform() > 0.5)
            ):
                self.catched_object = self.perceptions["cylinders"].data[0]
                if object_angle > 0:
                    self.perceptions["ball_in_left_hand"].data = True
                    self.perceptions["ball_in_right_hand"].data = False
                else:
                    self.perceptions["ball_in_left_hand"].data = False
                    self.perceptions["ball_in_right_hand"].data = True
            if self.object_pickable_with_two_hands(distance, angle) and (self.rng.uniform() > 0.5):
                self.catched_object = self.perceptions["cylinders"].data[0]
                self.perceptions["ball_in_left_hand"].data = True
                self.perceptions["ball_in_right_hand"].data = True
        elif self.world == World.GRIPPER_AND_LOW_FRICTION_TWO_BOXES:
            self.perceptions["boxes"].data = []
            distance, angle = self.random_position(in_valid=True, out_valid=False)
            self.perceptions["boxes"].data.append(self.base_messages["boxes"]())
            self.perceptions["boxes"].data[0].distance = distance
            self.perceptions["boxes"].data[0].angle = angle
            self.perceptions["boxes"].data[0].diameter = 0.12
            # self.perceptions["boxes"].data[0].id = Item.box.value
            distance, angle = self.random_position(in_valid=False, out_valid=True)
            self.perceptions["boxes"].data.append(self.base_messages["boxes"]())
            self.perceptions["boxes"].data[1].distance = distance
            self.perceptions["boxes"].data[1].angle = angle
            self.perceptions["boxes"].data[1].diameter = 0.12
            # self.perceptions["boxes"].data[1].id = Item.box.value
            self.perceptions["cylinders"].data = []
            distance, angle = self.random_position(in_valid=True, out_valid=False)
            self.perceptions["cylinders"].data.append(self.base_messages["cylinders"]())
            self.perceptions["cylinders"].data[0].distance = distance
            self.perceptions["cylinders"].data[0].angle = angle
            if self.rng.uniform() > 0.5:
                self.perceptions["cylinders"].data[0].diameter = 0.03
            else:
                self.perceptions["cylinders"].data[0].diameter = 0.07
            # self.perceptions["cylinders"].data[0].id = Item.cylinder.value
            self.perceptions["ball_in_left_hand"].data = False
            self.perceptions["ball_in_right_hand"].data = False
        elif self.world == World.KITCHEN:
            self.perceptions["boxes"].data = []
            self.perceptions["boxes"].data.append(self.base_messages["boxes"]())
            self.perceptions["boxes"].data[0].distance = 0.605
            self.perceptions["boxes"].data[0].angle = 0.0
            self.perceptions["boxes"].data[0].diameter = 0.12
            # self.perceptions["boxes"].data[0].id = Item.skillet.value
            self.perceptions["cylinders"].data = []
            distance, angle = self.random_position(in_valid=True, out_valid=True)
            self.perceptions["cylinders"].data.append(self.base_messages["cylinders"]())
            self.perceptions["cylinders"].data[0].distance = distance
            self.perceptions["cylinders"].data[0].angle = angle
            self.perceptions["cylinders"].data[0].diameter = 0.03
            # self.perceptions["cylinders"].data[0].id = Item.carrot.value
            distance, angle = self.random_position(in_valid=True, out_valid=True)
            self.perceptions["cylinders"].data.append(self.base_messages["cylinders"]())
            self.perceptions["cylinders"].data[1].distance = distance
            self.perceptions["cylinders"].data[1].angle = angle
            self.perceptions["cylinders"].data[1].diameter = 0.03
            # self.perceptions["cylinders"].data[1].id = Item.eggplant.value
            distance, angle = self.random_position(in_valid=True, out_valid=True)
            self.perceptions["cylinders"].data.append(self.base_messages["cylinders"]())
            self.perceptions["cylinders"].data[2].distance = distance
            self.perceptions["cylinders"].data[2].angle = angle
            self.perceptions["cylinders"].data[2].diameter = 0.03
            # self.perceptions["cylinders"].data[2].id = Item.orange.value
            self.perceptions["ball_in_left_hand"].data = False
            self.perceptions["ball_in_right_hand"].data = False
        else:
            self.get_logger().info("Unknown world received in simulator!")
        # Goal sensors
        self.update_reward_sensor()

    def grasp_object_policy(self):
        """Grasp an object with a gripper."""
        if not self.catched_object:
            for cylinder in self.perceptions["cylinders"].data:
                if (
                    World.GRIPPER_AND_LOW_FRICTION.name in self.world.name
                    and (
                        not self.object_too_far(cylinder.distance, cylinder.angle)
                    )
                    and self.object_is_small(cylinder.diameter)
                ):
                    if cylinder.angle > 0.0:
                        self.perceptions["ball_in_left_hand"].data = True
                    else:
                        self.perceptions["ball_in_right_hand"].data = True
                    self.catched_object = cylinder
                    break

    def grasp_with_two_hands_policy(self):
        """Grasp an object using both arms."""
        if not self.catched_object:
            for cylinder in self.perceptions["cylinders"].data:
                if (self.object_pickable_with_two_hands(cylinder.distance, cylinder.angle)) and (
                    (World.NO_GRIPPER_AND_HIGH_FRICTION.name in self.world.name)
                    or (not self.object_is_small(cylinder.diameter))
                ):
                    self.perceptions["ball_in_left_hand"].data = True
                    self.perceptions["ball_in_right_hand"].data = True
                    self.catched_object = cylinder
                    break



    def change_hands_policy(self):
        """Exchange an object from one hand to the other one."""
        if self.perceptions["ball_in_left_hand"].data and (
            not self.perceptions["ball_in_right_hand"].data
        ):
            self.perceptions["ball_in_left_hand"].data = False
            self.perceptions["ball_in_right_hand"].data = True
            self.catched_object.angle = -self.catched_object.angle
            self.catched_object.distance = self.avoid_reward_by_chance(
                self.catched_object.distance, self.catched_object.angle
            )
        elif (not self.perceptions["ball_in_left_hand"].data) and self.perceptions[
            "ball_in_right_hand"
        ].data:
            self.perceptions["ball_in_left_hand"].data = True
            self.perceptions["ball_in_right_hand"].data = False
            self.catched_object.angle = -self.catched_object.angle
            self.catched_object.distance = self.avoid_reward_by_chance(
                self.catched_object.distance, self.catched_object.angle
            )

    def sweep_object_policy(self):
        """Sweep an object to the front of the robot."""
        if not self.catched_object:
            for cylinder in self.perceptions["cylinders"].data:
                if not self.object_too_far(cylinder.distance, cylinder.angle):
                    sign = numpy.sign(cylinder.angle)
                    (
                        cylinder.distance,
                        cylinder.angle,
                    ) = self.send_object_twohandsreachable(cylinder.distance)
                    if (
                        World.GRIPPER_AND_LOW_FRICTION.name in self.world.name
                    ) and self.object_is_small(cylinder.diameter):
                        cylinder.angle = -0.4 * sign
                    cylinder.distance = self.avoid_reward_by_chance(
                        cylinder.distance, cylinder.angle
                    )
                    break

    def put_object_in_box_policy(self):
        """Put an object into the box."""
        if self.catched_object:
            for box in self.perceptions["boxes"].data:
                if (not self.object_too_far(box.distance, box.angle)) and (
                    ((box.angle > 0.0) and self.perceptions["ball_in_left_hand"].data)
                    or ((box.angle <= 0.0) and self.perceptions["ball_in_right_hand"].data)
                ):
                    self.catched_object.distance = box.distance
                    self.catched_object.angle = box.angle
                    self.perceptions["ball_in_left_hand"].data = False
                    self.perceptions["ball_in_right_hand"].data = False
                    self.catched_object = None
                    break

    def put_object_with_robot_policy(self):
        """Put an object as close to the robot as possible."""
        if self.catched_object:
            (
                self.catched_object.distance,
                self.catched_object.angle,
            ) = self.calculate_closest_position(self.catched_object.angle)
            # Box and cylinder collide.
            # This is not realistic, it needs to be improved.
            if self.object_in_close_box(self.catched_object.distance, self.catched_object.angle):
                self.catched_object.angle += 0.10
            self.perceptions["ball_in_left_hand"].data = False
            self.perceptions["ball_in_right_hand"].data = False
            self.catched_object = None

    def throw_policy(self):
        """Throw an object."""
        if self.catched_object:
            for box in self.perceptions["boxes"].data:
                if self.object_too_far(box.distance, box.angle) and (
                    (box.angle > 0.0 and self.perceptions["ball_in_left_hand"].data)
                    or (box.angle <= 0.0 and self.perceptions["ball_in_right_hand"].data)
                ):
                    self.catched_object.distance = box.distance
                    self.catched_object.angle = box.angle
                    # else:
                    #     self.catched_object.distance, self.catched_object.angle = self.send_object_outofreach(
                    #         self.catched_object.angle, self.world.name
                    #     )
                    self.perceptions["ball_in_left_hand"].data = False
                    self.perceptions["ball_in_right_hand"].data = False
                    self.catched_object = None
                    break

    def ask_nicely_policy(self):
        """Ask someone to bring the object closer to us."""
        if not self.catched_object:
            for cylinder in self.perceptions["cylinders"].data:
                if self.object_too_far(cylinder.distance, cylinder.angle):
                    self.get_logger().info("Object too far in " + self.world.name)
                    if self.world == World.GRIPPER_AND_LOW_FRICTION_SHORT_ARM:
                        distance = self.short_outer(abs(cylinder.angle))
                    elif self.world == World.GRIPPER_AND_LOW_FRICTION_DAMAGED_SERVO:
                        max_allowed_angle = 0.8 * numpy.arctan2(1.07, 0.37)
                        if cylinder.angle > max_allowed_angle:
                            cylinder.angle = max_allowed_angle - 0.1
                        distance = self.normal_outer(abs(cylinder.angle))
                    elif self.world == World.GRIPPER_AND_LOW_FRICTION_OBSTACLE:
                        if abs(cylinder.angle) < 0.17:
                            distance = self.short_outer(abs(cylinder.angle))
                        else:
                            distance = self.normal_outer(abs(cylinder.angle))
                    else:
                        distance = self.normal_outer(abs(cylinder.angle))
                    cylinder.distance = self.avoid_reward_by_chance(distance - 0.1, cylinder.angle)
                    break

    
    def new_command_callback(self, data):
        """
        Process a command received

        :param data: The message that contais the command received
        :type data: ROS msg defined in setup_control_channel
        """
        self.get_logger().debug(f"Command received... ITERATION: {data.iteration}")
        if data.command == "reset_world":
            self.last_reset_iteration = data.iteration
            self.world = World[data.world]
            self.random_perceptions()
            self.publish_perceptions()
            if (not self.catched_object) and (
                self.perceptions["ball_in_left_hand"].data
                or self.perceptions["ball_in_right_hand"].data
            ):
                self.get_logger().error("Critical error: catched_object is empty and it should not!!!")
        elif data.command == "end":
            self.get_logger().info("Ending simulator as requested by LTM...")
            rclpy.shutdown()

    def publish_perceptions(self):
        for ident, publisher in self.sim_publishers.items():
            self.get_logger().debug("Publishing " + ident + " = " + str(self.perceptions[ident].data))
            publisher.publish(self.perceptions[ident])

    def new_action_callback(self, data):
        """
        Execute a policy and publish new perceptions.

        :param data: The message that contains the policy to execute
        :type data: ROS msg defined in setup_control_channel
        """
        self.get_logger().info("Executing policy " + str(data.data))
        getattr(self, data.data + "_policy")()
        self.update_reward_sensor()
        self.publish_perceptions()
        if (not self.catched_object) and (
            self.perceptions["ball_in_left_hand"].data
            or self.perceptions["ball_in_right_hand"].data
        ):
            self.get_logger().error("Critical error: catched_object is empty and it should not!!!")
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
        self.update_reward_sensor()
        self.publish_perceptions()
        if (not self.catched_object) and (
            self.perceptions["ball_in_left_hand"].data
            or self.perceptions["ball_in_right_hand"].data
        ):
            self.get_logger().error("Critical error: catched_object is empty and it should not!!!")
            rclpy.shutdown()
        response.success = True
        return response

    def setup_control_channel(self, simulation):
        """
        Configure the ROS topic where listen for commands to be executed.

        :param simulation: The params from the config file to setup the control channel
        :type simulation: dict
        """
        self.ident = simulation["id"]
        topic = simulation["control_topic"]
        classname = simulation["control_msg"]
        message = class_from_classname(classname)
        self.get_logger().info("Subscribing to... " + str(topic))
        self.create_subscription(message, topic, self.new_command_callback, 0)
        topic = simulation.get("executed_policy_topic")
        service = simulation.get("executed_policy_service")
        classname = simulation["executed_policy_msg"]
        message = class_from_classname(classname)
        if topic:
            self.get_logger().info("Subscribing to... " + str(topic))
            self.create_subscription(message, topic, self.new_action_callback, 0)
        if service:
            self.get_logger().info("Creating server... " + str(service))
            self.create_service(message, service, self.new_action_service_callback, callback_group=self.cbgroup_server)
            self.get_logger().info("Creating perception publisher timer... ")
            self.perceptions_timer = self.create_timer(0.01, self.publish_perceptions, callback_group=self.cbgroup_server)
            

    def setup_perceptions(self, perceptions):
        """
        Configure the ROS publishers where publish perception values.

        :param perceptions: The params from the config file to setup the perceptions
        :type perceptions: dict
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
            else:
                self.perceptions[sid].data = False
            self.get_logger().info("I will publish to... " + str(topic))
            self.sim_publishers[sid] = self.create_publisher(message, topic, 0) #TODO: ¿latch in ROS2?

    def load_configuration(self):
        """
        Load configuration from a file.

        :param random_seed: The seed for the random numbers generation
        :type random_seed: int
        :param config_file: The file with the params to configurate the simulation
        :type config_file: yaml file
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
                self.setup_perceptions(config["SimulatedBaxter"]["Perceptions"])
                # Be ware, we can not subscribe to control channel before creating all sensor publishers.
                self.setup_control_channel(config["Control"])
        if self.random_seed:
            self.rng = numpy.random.default_rng(self.random_seed)
            self.get_logger().info(f"Setting random number generator with seed {self.random_seed}")
        else:
            self.rng = numpy.random.default_rng()
        
        self.load_experiment_file_in_commander()

def main(args=None):
    rclpy.init(args=args)
    sim = LTMSim()
    sim.load_configuration()

    try:
        rclpy.spin(sim)
    except KeyboardInterrupt:
        print('Keyboard Interrupt Detected: Shutting down simulator...')
    finally:
        sim.destroy_node()

if __name__ == '__main__':
    main()       
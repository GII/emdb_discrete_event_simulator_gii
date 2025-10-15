import math
import numpy as np
from enum import Enum
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.axes import Axes
from scipy.spatial import distance
from numpy.random import Generator
from rclpy.impl.rcutils_logger import RcutilsLogger

import numbers
import time
import random

class EntityType(Enum):
    """
    Class that defines the different types of entities in the simulator.
    """
    DEFAULT = 0
    ROBOT = 1
    BOX = 2
    BALL = 3



class Entity:
    """
    Class that implements an Entity in the simulator.
    """
    def __init__(self, name, x=0.0, y=0.0, angle=0.0, type=EntityType.DEFAULT) -> None:
        """
        Create an Entity with a name, position (x, y), angle and type.

        :param name: Name of the entity.
        :type name: str
        :param x: x position of the entity.
        :type x: float
        :param y: y position of the entity.
        :type y: float
        :param angle: Angle of the entity in degrees.
        :type angle: float
        :param type: Type of the entity.
        :type type: simulators.scenarios_2D.EntityType
        """
        self.name=name
        self.visual=[]
        self.x=x
        self.y=y
        self.angle=angle
        self.type=type

    def get_angle(self):
        """
        Get the angle of the entity.

        :return: Angle of the entity in degrees.
        :rtype: float
        """
        return self.angle
    
    def get_pos(self):
        """
        Get the position of the entity.

        :return: Tuple with the x and y position of the entity.
        :rtype: tuple
        """
        return (self.x, self.y)
    
    def set_pos(self, x=None, y=None):
        """
        Set the position of the entity.
        If x or y is not provided, it will not be changed.

        :param x: x position of the entity.
        :type x: float
        :param y: y position of the entity.
        :type y: float
        :raises TypeError: If x or y is not a number.
        """
        if x:
            if isinstance(x, numbers.Number):
                self.x=x
            else:
                raise TypeError("X Position must be of numeric type")
        if y:
            if isinstance(x, numbers.Number):
                self.y=y
            else:
                raise TypeError("Y Position must be of numeric type")
        self.update_visual()
    
    def set_angle(self, angle):
        """
        Set the angle of the entity.
        The angle is bounded in the range [-180, 180] degrees.

        :param angle: Angle of the entity in degrees.
        :type angle: float
        """
        self.angle=((angle+180)%360)-180 #Bounded in range [-180, 180]
        self.update_visual()

    def set_color(self, color):
        """
        Set the color of the entity.
        This method should be implemented in the subclasses.

        :param color: Color of the entity.
        :type color: str
        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError
    
    def set_alpha(self, alpha):
        """
        Set the transparency of the entity.
        This method should be implemented in the subclasses.

        :param alpha: Transparency of the entity, between 0 (transparent) and 1 (opaque).
        :type alpha: float
        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError
    
    def register_visual(self, ax:Axes):
        """
        Register the visual representation of the entity in the given Axes.
        This method should be implemented in the subclasses.

        :param ax: Axes where the visual representation will be registered.
        :type ax: matplotlib.axes.Axes
        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError
    
    def update_visual(self):
        """
        Update the visual representation of the entity.
        This method should be implemented in the subclasses.

        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError
    

class Robot(Entity):
    """
    Class that implements a Robot entity in the simulator.
    """
    def __init__(self, name, x=0.0, y=0.0, angle=0.0) -> None:
        """
        Create the robot entity.

        :param name: Name of the robot.
        :type name: str
        :param x: x position of the robot.
        :type x: float
        :param y: y position of the robot.
        :type y: float
        :param angle: Angle of the robot in degrees.
        :type angle: float
        """
        super().__init__(name, x, y, angle, type=EntityType.ROBOT)
        self.visual.append(patches.Rectangle((0, 0), 75, 60, angle=0.0, fc=(0.8, 0, 0.2), label=f"{name}_body"))
        self.visual.append(patches.Rectangle((0, 0), 20, 60, angle=0.0, fc='black', label=f'{name}_act'))
        self.catched_object=None
        self.gripper_state=False #True if closed
        self.update_visual()
    
    def set_gripper(self, value):
        """
        Set the gripper state of the robot.

        :param value: True if the gripper is closed, False otherwise.
        :type value: bool
        """
        self.gripper_state=value
        self.update_visual()
    
    def register_visual(self, ax:Axes):
        """
        Register the visual representation of the robot in the given Axes.

        :param ax: Axes where the visual representation will be registered.
        :type ax: matplotlib.axes.Axes
        """
        for patch in self.visual:
            ax.add_patch(patch)
    
    def update_visual(self):
        """
        Update the visual representation of the robot.
        """
        #Set body position
        w = self.visual[0].get_width()
        h = self.visual[0].get_height()
        # New x, y position
        new_x_body = self.x - w / 2 * math.cos(self.angle * math.pi / 180) + h / 2 * math.sin(
            self.angle * math.pi / 180)
        new_y_body = self.y - w / 2 * math.sin(self.angle * math.pi / 180) - h / 2 * math.cos(
            self.angle * math.pi / 180)
        
        #Actuator position relative to body
        new_x_act = new_x_body + w * math.cos(self.angle * math.pi / 180)
        new_y_act = new_y_body + w * math.sin(self.angle * math.pi / 180)

        self.visual[0].xy = (new_x_body, new_y_body)
        self.visual[1].xy = (new_x_act, new_y_act)

        #Set visual angle
        self.visual[0].angle = self.angle
        self.visual[1].angle = self.angle

        if self.gripper_state:
            self.visual[1].set_facecolor('yellow')
            self.visual[1].set_edgecolor("black")
        else:
            self.visual[1].set_facecolor('black')
            self.visual[1].set_edgecolor("black")
    
    def set_color(self, color):
        """
        Set the color of the robot.

        :param color: Color of the robot.
        :type color: str
        """
        self.visual[0].set_color(color)
    
    def set_alpha(self, alpha):
        """
        Set the transparency of the robot entity in the visual representation.

        :param alpha: Transparency of the robot, between 0 (transparent) and 1 (opaque).
        :type alpha: float
        """
        self.visual[0].set_alpha(alpha)
        self.visual[1].set_alpha(alpha)

class Ball(Entity):
    """
    Class that implements a Ball entity in the simulator.
    """
    def __init__(self, name, x=0.0, y=0.0, angle=0.0, radius=40.0, color="red") -> None:
        """
        Create the ball entity.

        :param name: Name of the ball.
        :type name: str
        :param x: x position of the ball.
        :type x: float
        :param y: y position of the ball.
        :type y: float
        :param angle: Angle of the ball in degrees.
        :type angle: float
        :param radius: Radius of the ball.
        :type radius: float
        :param color: Color of the ball.
        :type color: str
        """
        super().__init__(name, x, y, angle, type=EntityType.BALL)
        self.visual=patches.Circle((0, 0), radius, fc=color, label=name)
        self.catched_by=None
        self.update_visual()

    def register_visual(self, ax:Axes):
        """
        Register the visual representation of the ball in the given Axes.

        :param ax: Axes where the visual representation will be registered.
        :type ax: matplotlib.axes.Axes
        """
        ax.add_patch(self.visual)

    def update_visual(self):
        """
        Update the visual representation of the ball.
        """
        self.visual.set_center((self.x, self.y))
    
    def set_color(self, color):
        """
        Set the color of the ball.

        :param color: Color of the ball.
        :type color: str    
        """
        self.visual.set_color(color)

    def set_alpha(self, alpha):
        """
        Set the transparency of the ball entity in the visual representation.

        :param alpha: Transparency of the ball, between 0 (transparent) and 1 (opaque).
        :type alpha: float
        """
        self.visual.set_alpha(alpha)


class Box(Entity):
    """
    Class that implements a Box entity in the simulator.
    """
    def __init__(self, name, x=0.0, y=0.0, angle=0.0, color="blue", w=100.0, h=100.0) -> None:
        """
        Create the box entity.

        :param name: Name of the box.
        :type name: str
        :param x: x position of the box.
        :type x: float
        :param y: y position of the box.
        :type y: float
        :param angle: Angle of the box in degrees.
        :type angle: float
        :param color: Color of the box.
        :type color: str
        :param w: Width of the box.
        :type w: float
        :param h: Height of the box.
        :type h: float
        """
        super().__init__(name, x, y, angle, type=EntityType.BOX)
        self.visual=patches.Rectangle((0, 0), w, h, angle=0.0, fc=color, label=f"{name}")
        self.contents=[]
        self.update_visual()
    
    def register_visual(self, ax: Axes):
        """
        Register the visual representation of the box in the given Axes.

        :param ax: Axes where the visual representation will be registered.
        :type ax: matplotlib.axes.Axes    
        """
        ax.add_patch(self.visual)
    
    def update_visual(self):
        """
        Update the visual representation of the box.
        """
        #Set body position
        w = self.visual.get_width()
        h = self.visual.get_height()
        # New x, y position
        new_x = self.x - w / 2 * math.cos(self.angle * math.pi / 180) + h / 2 * math.sin(
            self.angle * math.pi / 180)
        new_y = self.y - w / 2 * math.sin(self.angle * math.pi / 180) - h / 2 * math.cos(
            self.angle * math.pi / 180)
        self.visual.xy = (new_x, new_y)
        #Set visual angle
        self.visual.angle = self.angle
    
    def set_color(self, color):
        """
        Set the color of the box.

        :param color: Color of the box.
        :type color: str
        """
        self.visual[0].set_color(color)
    
    def set_alpha(self, alpha):
        """
        Set the transparency of the box entity in the visual representation.

        :param alpha: Transparency of the box, between 0 (transparent) and 1 (opaque).
        :type alpha: float
        """
        self.visual[0].set_alpha(alpha)
        self.visual[1].set_alpha(alpha)


        

class Sim(object):
    """ Class that implements a Simulator
    
    This simulator makes possible to make/test different experiments in a virtual scenario.
    It contains the two arms of the Baxter robot, the Robobo! robot, some boxes and a
    ball.
    
    The implemented methods allow the user to move both robots throw the scenario (with
    and without the ball), get distances and relative angles between the different objects 
    and get/set the position of all of them.
    """

    def __init__(self, x_size=(0, 2500), y_size=(0, 1000), x_bounds=(100, 2400), y_bounds=(50, 800), visualize=True, verbose=False):
        """
        Create the different objects present in the Simulator and place them in it.

        :param x_size: Size of the plot in the x axis.
        :type x_size: tuple
        :param y_size: Size of the plot in the y axis.
        :type y_size: tuple
        :param x_bounds: Movement limits in the x axis.
        :type x_bounds: tuple
        :param y_bounds: Movement limits in the y axis.
        :type y_bounds: tuple
        :param visualize: If True, the simulator will visualize the objects in a plot.
        :type visualize: bool
        :param verbose: If True, the simulator will print debug information.
        :type verbose: bool
        """
        self.verbose=verbose

        #Plot and experimental area bounds
        self.x_plt_bounds=x_size
        self.y_plt_bounds=y_size
        self.x_bounds=x_bounds
        self.y_bounds=y_bounds

        # Enable simulation visualization (see the objects moving)
        self.visualize = visualize

        #Object list
        self.entities=[]

        # Generate Plots
        if self.visualize:
            self.fig = plt.figure()
            self.fig.canvas.set_window_title('Simulator')
            self.ax = plt.axes(xlim=self.x_plt_bounds, ylim=self.y_plt_bounds)
            self.ax.axes.get_xaxis().set_visible(False)
            self.ax.axes.get_yaxis().set_visible(False)
            self.ax.axes.set_aspect("equal")

            ##Subtitle
            self.ball_active_goal = plt.Circle((500, 900), 40, fc='white', alpha=1.0, label='ball_active_goal')
            self.ball_active_subgoal = plt.Circle((1200, 900), 40, fc='white', alpha=1.0, label='ball_active_subgoal')
            self.ball_active_context = plt.Circle((2000, 900), 40, fc='white', alpha=1.0, label='ball_context')

            xy1 = (self.ball_active_context.center[0], self.ball_active_context.center[1]+1.2*self.ball_active_context.radius)
            xy2 = (self.ball_active_goal.center[0], self.ball_active_goal.center[1]+1.2*self.ball_active_goal.radius)
            xy3 = (self.ball_active_subgoal.center[0], self.ball_active_subgoal.center[1] + 1.2 * self.ball_active_subgoal.radius)

            # Draw Movement boundaries
            plt.axhline(y=self.y_bounds[1], xmin=self.x_bounds[0]/self.x_plt_bounds[1], xmax=self.x_bounds[1]/self.x_plt_bounds[1], linestyle='--', color='grey')
            plt.axhline(y=self.y_bounds[0], xmin=self.x_bounds[0]/self.x_plt_bounds[1], xmax=self.x_bounds[1]/self.x_plt_bounds[1], linestyle='--', color='grey')
            plt.axvline(x=self.x_bounds[1], ymin=self.y_bounds[0]/self.y_plt_bounds[1], ymax=self.y_bounds[1]/self.y_plt_bounds[1], linestyle='--', color='grey')
            plt.axvline(x=self.x_bounds[0], ymin=self.y_bounds[0]/self.y_plt_bounds[1], ymax=self.y_bounds[1]/self.y_plt_bounds[1], linestyle='--', color='grey')

    def plot_entities(self):
        """
        Plot the generated entities.
        """
        for entity in reversed(self.entities):
            entity.register_visual(self.ax)

    def get_close_entities(self, entity:Entity, threshold):
        """
        Get a list of entities that are close to the given entity within a certain threshold.

        :param entity: Entity to which we want to find close entities.
        :type entity: simulators.scenarios_2D.Entity
        :param threshold: Distance threshold to consider an entity as close.
        :type threshold: float
        :return: List of entities that are close to the given entity.
        :rtype: list of simulators.scenarios_2D.Entity
        """    
        close_list=[]
        ent_list= [ent for ent in self.entities if ent!=entity]
        for ent in ent_list:
            dist = distance.euclidean(ent.get_pos(), entity.get_pos())
            if dist <= threshold:
                close_list.append(ent)
        return close_list

    def filter_entities(self, entities, type):
        """
        Filter entities by type.

        :param entities: List of entities to filter.
        :type entities: list of simulators.scenarios_2D.Entity
        :param type: Type of entity to filter by.
        :type type: simulators.scenarios_2D.EntityType
        :return: List of entities of the specified type.
        :rtype: list of simulators.scenarios_2D.Entity
        """
        return [entity for entity in entities if entity.type == type]

    def apply_action(self, **params):
        """
        Placeholder method: Implement the action logic according to the desired scenario.
        """
        self.world_rules()
        if self.visualize:
            self.fig.canvas.draw()
            plt.pause(0.01)

    def world_rules(self):
        """
        Placeholder method: Implement the desired world logic according to the desired scenario.

        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    def enforce_limits(self, entity: Entity, limits):
        """
        Check if the next position of one of the entities is inside its movement limits.

        :param entity: Entity to check the limits.
        :type entity: simulators.scenarios_2D.Entity
        :param limits: Movement limits of the entity, in the form ((x_min, x_max), (y_min, y_max)).
        :type limits: tuple of tuples
        """
        (x, y) = entity.get_pos()
        (x_min, x_max) = limits[0]
        (y_min, y_max) = limits[1]      
        changed=False
        # Movement boundaries
        if x > x_max:
            x=x_max
            changed=True
        elif x<x_min:
            x=x_min
            changed=True
        if y > y_max:
            y=y_max
            changed=True
        elif y<y_min:
            y=y_min
            changed=True
        if changed:
            entity.set_pos(x, y)

    def get_sensorization(self):
        """
        Return a sensorization vector with the distance between the object in which the robot is focused and its
        actuator, the color of this object...

        :raises NotImplementedError: If the method is not implemented in the subclass.
         """
        raise NotImplementedError

    def get_scenario_data(self):
        """
        Scenario data needed to predict future states using the world model.

        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError


    def restart_scenario(self):
        """
        Set the scenario to the desired initial conditions.

        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    def restart_scenario_partially(self):
        """
        Partially reinitialize the scenario.

        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    @staticmethod
    def normalize_value(value, max_value, min_value=0.0):
        """
        Normalize a value between 0 and 1, given a maximum and minimum value.

        :param value: Value to normalize.
        :type value: float
        :param max_value: Maximum value for normalization.
        :type max_value: float
        :param min_value: Minimum value for normalization.
        :type min_value: float
        :return: Normalized value.
        :rtype: float
        """
        return (value - min_value) / (max_value - min_value)

    @staticmethod
    def get_relative_angle(x1_y1, x2_y2):
        """
        Return the relative angle between two points

        :param x1_y1: Tuple with the coordinates of the first point (x1, y1).
        :type x1_y1: tuple
        :param x2_y2: Tuple with the coordinates of the second point (x2, y2).
        :type x2_y2: tuple
        :return: Relative angle in degrees between the two points.
        :rtype: float
        """
        (x1, y1) = x1_y1
        (x2, y2) = x2_y2
        return math.atan2(y2 - y1, x2 - x1) * 180 / math.pi





class Baxter2Arms(Sim):
    """
    Class that simulates the Baxter robot.
    """
    def __init__(self, x_size=(0, 2500), y_size=(0, 1000), x_bounds=(100, 2400), y_bounds=(50, 800), visualize=True, verbose=False):
        """
        Create the Baxter robot with two arms.

        :param x_size: Size of the plot in the x axis.
        :type x_size: tuple
        :param y_size: Size of the plot in the y axis.
        :type y_size: tuple
        :param x_bounds: Movement limits in the x axis.
        :type x_bounds: tuple
        :param y_bounds: Movement limits in the y axis.
        :type y_bounds: tuple
        :param visualize: If True, the simulator will visualize the objects in a plot.
        :type visualize: bool
        :param verbose: If True, the simulator will print debug information.
        :type verbose: bool
        """
        super().__init__(x_size, y_size, x_bounds, y_bounds, visualize, verbose)
        #Define objects in simulation
        ## Baxter
        self.baxter_left=Robot("baxter_left", 700, 300, 90)
        self.baxter_right=Robot("baxter_right", 1800, 300, 90)
        self.robots=[self.baxter_left, self.baxter_right]
        self.baxter_left_limits=((self.x_bounds[0], self.x_bounds[1]/2),self.y_bounds)
        self.baxter_right_limits=((self.x_bounds[1]/2, self.x_bounds[1]),self.y_bounds)
        self.entities.extend(self.robots) #Include robots in entities list

    def move_robot_arm(self, arm:Robot, vel):
        """
        Move robot arm wih a specific velocity.

        :param arm: Robot entity to move.
        :type arm: simulators.scenarios_2D.Robot
        :param vel: Velocity to move the arm.
        :type vel: float
        """
        x, y = arm.get_pos()
        arm.set_pos(x + vel * math.cos(arm.angle * math.pi / 180),
                                  y + vel * math.sin(arm.angle * math.pi / 180))

    def robot_arm_action(self, arm: Robot, relative_angle, vel, step_world=True):
        """
        Move robot arm with a specific angle and velocity.

        :param arm: Robot entity to move.
        :type arm: simulators.scenarios_2D.Robot
        :param relative_angle: Relative angle to move the arm.
        :type relative_angle: float
        :param vel: Velocity to move the arm.
        :type vel: float
        """
        angle = arm.angle + relative_angle
        arm.set_angle(angle)
        self.move_robot_arm(arm, vel)
        if step_world:
            self.world_rules()

    def apply_action(self, rel_angle_l=0.0, rel_angle_r=0.0, vel_left=0.0, vel_right=0, gripper_left=False, gripper_right=False):
        """
        Move arms with a specific angle and velocity (default 0).

        :param rel_angle_l: Relative angle to move the left arm.
        :type rel_angle_l: float
        :param rel_angle_r: Relative angle to move the right arm.
        :type rel_angle_r: float
        :param vel_left: Velocity to move the left arm.
        :type vel_left: float
        :param vel_right: Velocity to move the right arm.
        :type vel_right: float
        :param gripper_left: Gripper state of the left arm (True if closed, False otherwise).
        :type gripper_left: bool
        :param gripper_right: Gripper state of the right arm (True if closed, False otherwise).
        :type gripper_right: bool
        """
        self.robot_arm_action(self.baxter_left, rel_angle_l, vel_left, step_world=False)
        self.robot_arm_action(self.baxter_right, rel_angle_r, vel_right, step_world=False)
        self.baxter_left.set_gripper(gripper_left)
        self.baxter_right.set_gripper(gripper_right)
        self.world_rules()
        if self.visualize:
            self.fig.canvas.draw()
            plt.pause(0.001)




class ComplexScenario(Baxter2Arms):
    """
    Class that implements a complex scenario with a Baxter robot and multiple balls and boxes.
    """
    def __init__(self, x_size=(0, 2500), y_size=(0, 1000), x_bounds=(100, 2400), y_bounds=(50, 800), visualize=True):
        """
        Create the complex scenario with a Baxter robot, multiple balls and boxes.

        :param x_size: Size of the plot in the x axis.
        :type x_size: tuple
        :param y_size: Size of the plot in the y axis.
        :type y_size: tuple
        :param x_bounds: Movement limits in the x axis.
        :type x_bounds: tuple
        :param y_bounds: Movement limits in the y axis.
        :type y_bounds: tuple
        :param visualize: If True, the simulator will visualize the objects in a plot.
        :type visualize: bool
        """
        super().__init__(x_size, y_size, x_bounds, y_bounds, visualize)
        ##Objects
        self.objects=[]
        self.objects.append(Ball("ball_1", 300, 650, color="red"))
        self.objects.append(Ball("ball_2", 700, 650, color="green"))
        self.objects.append(Ball("ball_3", 1100, 700, color="blue"))
        self.objects.append(Ball("ball_4", 1500, 700, color="orange"))
        self.objects.append(Ball("ball_5", 1900, 650, color="purple"))
        self.objects.append(Ball("ball_6", 2300, 600, color="cyan"))
        self.entities.extend(self.objects)

        ##Boxes
        self.box1 = Box("box_1", 900, 400, w=150, h=150)
        self.box2 = Box("box_2", 1600, 400, w=150, h=150, color="blue")
        self.boxes=[self.box1, self.box2]
        self.entities.extend(self.boxes)
    
        
        # Show figure and patches
        self.plot_entities()

    def world_rules(self):
        """
        Establish the ball position in the scenario.
        """

        #Enforce position bounds
        self.enforce_limits(self.baxter_left, self.baxter_left_limits)
        self.enforce_limits(self.baxter_right, self.baxter_right_limits)
        for object in self.objects:
            self.enforce_limits(object, (self.x_bounds, self.y_bounds))
        #Check for grasps
        for robot in self.robots:
            #Catch close objects
            if robot.gripper_state and not robot.catched_object:
                close_object=self.get_close_object(robot)
                if close_object:
                    if not close_object.catched_by:
                        robot.catched_object=close_object
                        robot.catched_object.catched_by=robot
            if not robot.gripper_state and robot.catched_object:
                robot.catched_object.catched_by=None
                robot.catched_object=None
        #Update position of grasped objects
        for object in self.objects:
            if object.catched_by:
                pose=object.catched_by.get_pos()
                object.set_pos(pose[0], pose[1])


class SimpleScenario(Baxter2Arms):
    """
    Class that implements a simple scenario with a Baxter robot and a ball.
    """
    def __init__(self, x_size=(0, 2800), y_size=(0, 1550), x_bounds=(100, 2700), y_bounds=(50, 1350), visualize=True, logger=None):
        """
        Create the simple scenario with a Baxter robot and a ball.

        :param x_size: Size of the plot in the x axis.
        :type x_size: tuple
        :param y_size: Size of the plot in the y axis.
        :type y_size: tuple
        :param x_bounds: Movement limits in the x axis.
        :type x_bounds: tuple
        :param y_bounds: Movement limits in the y axis.
        :type y_bounds: tuple
        :param visualize: If True, the simulator will visualize the objects in a plot.
        :type visualize: bool
        :param logger: Logger to log debug information.
        :type logger: rclpy.impl.rcutils_logger.RcutilsLogger
        """
        super().__init__(x_size, y_size, x_bounds, y_bounds, visualize)
        self.logger:RcutilsLogger=logger
        ##Objects
        self.objects=[]
        self.objects.append(Ball("ball_1", 300, 650, color="red"))
        self.entities.extend(self.objects)

        ##Boxes
        self.box1 = Box("box_1", 900, 400, w=150, h=150)
        self.entities.append(self.box1)
        
        # Show figure and patches
        if self.visualize:
            self.plot_entities()
            plt.pause(0.001)

    def world_rules(self):
        """
        Establish the ball position in the scenario
        """

        #Enforce position bounds
        self.enforce_limits(self.baxter_left, self.baxter_left_limits)
        self.enforce_limits(self.baxter_right, self.baxter_right_limits)
        for object in self.objects:
            self.enforce_limits(object, (self.x_bounds, self.y_bounds))
        #Check for grasps
        for robot in self.robots:
            #Catch close objects
            if robot.gripper_state and not robot.catched_object:
                close_object=self.filter_entities(self.get_close_entities(robot, threshold=50), EntityType.BALL)
                if close_object:
                    if not close_object[0].catched_by:
                        robot.catched_object=close_object[0]
                        robot.catched_object.catched_by=robot
            if not robot.gripper_state and robot.catched_object:
                robot.catched_object.catched_by=None
                robot.catched_object=None

            #Check if something is in the box
            objs_close=self.filter_entities(self.get_close_entities(self.box1, threshold=50), EntityType.BALL)
            self.box1.contents=[]
            for obj in objs_close:
                if not obj.catched_by:
                    self.box1.contents.append(obj)

        #Update position of grasped objects
        for object in self.objects:
            if object.catched_by:
                pose=object.catched_by.get_pos()
                object.set_pos(pose[0], pose[1])
        
        if self.logger:
            self.logger.info(f"DEBUG --------------- Updated World ---------------")
            for robot in self.robots:
                self.logger.info(f"{robot.name} gripper state: {robot.gripper_state}")
                if robot.catched_object:
                    self.logger.info(f"{robot.name} has catched {robot.catched_object.name}")    
                else:
                    self.logger.info(f"{robot.name} has not catched any object")
            for obj in self.entities:
                self.logger.info(f"{obj.name} is at position {obj.get_pos()}")

    def restart_scenario(self, rng:Generator):
        """
        Set the scenario to random initial conditions.

        :param rng: Random number generator.
        :type rng: numpy.random.Generator
        """
        self.baxter_left.set_pos(rng.uniform(self.baxter_left_limits[0][0], self.baxter_left_limits[0][1]), rng.uniform(self.baxter_left_limits[1][0], self.baxter_left_limits[1][1]))
        self.baxter_right.set_pos(rng.uniform(self.baxter_right_limits[0][0], self.baxter_right_limits[0][1]), rng.uniform(self.baxter_right_limits[1][0], self.baxter_right_limits[1][1])) 
        self.baxter_left.set_gripper(False)
        self.baxter_right.set_gripper(False)
        
        self.box1.contents=[]
        #TODO: Reset grippers

        first_shuffle=True
        while any([distance.euclidean(self.box1.get_pos(), obj.get_pos()) < 50 for obj in self.objects]) or first_shuffle:
            self.box1.set_pos(rng.uniform(self.x_bounds[0], self.x_bounds[1]), rng.uniform(self.y_bounds[0], self.y_bounds[1]))
            for object in self.objects:
                object.set_pos(rng.uniform(self.x_bounds[0], self.x_bounds[1]), rng.uniform(self.y_bounds[0], self.y_bounds[1]))
                object.catched_by=None
            first_shuffle=False


if __name__ == '__main__':
    """Simulator Demo"""
    a = ComplexScenario()

    while True:
        vel_l=random.uniform(0, 60)
        ang_l=random.uniform(-90, 90)
        vel_r=random.uniform(0, 60)
        ang_r=random.uniform(-90, 90)
        grp_l=bool(random.getrandbits(1))
        grp_r=bool(random.getrandbits(1))
        a.apply_action(ang_l, ang_r, vel_l, vel_r, grp_l, grp_r)
        plt.pause(0.01)



import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.axes import Axes
from scipy.spatial import distance
import numbers

import time
import random


class Entity:
    def __init__(self, name, x=0, y=0, angle=0) -> None:
        self.name=name
        self.visual=[]
        self.x=x
        self.y=y
        self.angle=angle

    def get_angle(self):
        return self.angle
    
    def get_pos(self):
        return (self.x, self.y)
    
    def set_pos(self, x=None, y=None):
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
        self.angle=angle
        self.update_visual()

    def set_color(self, color):
        raise NotImplementedError
    
    def set_alpha(self, alpha):
        raise NotImplementedError
    
    def register_visual(self, ax:Axes):
        raise NotImplementedError
    
    def update_visual(self):
        raise NotImplementedError
    

class Robot(Entity):
    def __init__(self, name, x=0, y=0, angle=0) -> None:
        super().__init__(name, x, y, angle)
        self.visual.append(patches.Rectangle((0, 0), 75, 60, angle=0.0, fc=(0.8, 0, 0.2), label=f"{name}_body"))
        self.visual.append(patches.Rectangle((0, 0), 20, 60, angle=0.0, fc='black', label=f'{name}_act'))
        self.catched_object=None
        self.gripper_state=False #True if closed
        self.update_visual()
    
    def set_gripper(self, value):
        self.gripper_state=value
        self.update_visual()
    
    def register_visual(self, ax:Axes):
        for patch in self.visual:
            ax.add_patch(patch)
    
    def update_visual(self):
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
        self.visual[0].set_color(color)
    
    def set_alpha(self, alpha):
        self.visual[0].set_alpha(alpha)
        self.visual[1].set_alpha(alpha)

class Ball(Entity):
    def __init__(self, name, x=0, y=0, angle=0, radius=40, color="red") -> None:
        super().__init__(name, x, y, angle)
        self.visual=patches.Circle((0, 0), radius, fc=color, label=name)
        self.catched_by=None
        self.update_visual()

    def register_visual(self, ax:Axes):
        ax.add_patch(self.visual)

    def update_visual(self):
        self.visual.set_center((self.x, self.y))
    
    def set_color(self, color):
        self.visual.set_color(color)

    def set_alpha(self, alpha):
        self.visual.set_alpha(alpha)


class Box(Entity):
    def __init__(self, name, x=0, y=0, angle=0, color="blue", w=100, h=100) -> None:
        super().__init__(name, x, y, angle)
        self.visual=patches.Rectangle((0, 0), w, h, angle=0.0, fc=color, label=f"{name}")
        self.contents=[]
        self.update_visual()
    
    def register_visual(self, ax: Axes):
        ax.add_patch(self.visual)
    
    def update_visual(self):
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
        self.visual[0].set_color(color)
    
    def set_alpha(self, alpha):
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

    def __init__(self, x_size=(0, 2500), y_size=(0, 1000), x_bounds=(100, 2400), y_bounds=(50, 800), visualize=True):
        """Create the different objects present in the Simulator and place them in it"""

        #Plot and experimental area bounds
        self.x_plt_bounds=x_size
        self.y_plt_bounds=y_size
        self.x_bounds=x_bounds
        self.y_bounds=y_bounds

        # Enable simulation visualization (see the objects moving)
        self.visualize = True

        #Object list
        self.entities=[]

        # Generate Plots
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
        self.ax.annotate("CONTEXT", xy=xy1, fontsize=10, ha="center")
        xy2 = (self.ball_active_goal.center[0], self.ball_active_goal.center[1]+1.2*self.ball_active_goal.radius)
        self.ax.annotate("GOAL", xy=xy2, fontsize=10, ha="center")
        xy3 = (self.ball_active_subgoal.center[0], self.ball_active_subgoal.center[1] + 1.2 * self.ball_active_subgoal.radius)
        self.ax.annotate("SUB-GOAL", xy=xy3, fontsize=10, ha="center")

        # Draw Movement boundaries
        plt.axhline(y=self.y_bounds[1], xmin=self.x_bounds[0]/self.x_plt_bounds[1], xmax=self.x_bounds[1]/self.x_plt_bounds[1], linestyle='--', color='grey')
        plt.axhline(y=self.y_bounds[0], xmin=self.x_bounds[0]/self.x_plt_bounds[1], xmax=self.x_bounds[1]/self.x_plt_bounds[1], linestyle='--', color='grey')
        plt.axvline(x=self.x_bounds[1], ymin=self.y_bounds[0]/self.y_plt_bounds[1], ymax=self.y_bounds[1]/self.y_plt_bounds[1], linestyle='--', color='grey')
        plt.axvline(x=self.x_bounds[0], ymin=self.y_bounds[0]/self.y_plt_bounds[1], ymax=self.y_bounds[1]/self.y_plt_bounds[1], linestyle='--', color='grey')

    def plot_entities(self):
        for entity in reversed(self.entities):
            entity.register_visual(self.ax)


    def apply_action(self, **params):
        """Placeholder method: Implement the action logic according to the desired scenario"""
        self.world_rules()
        if self.visualize:
            self.fig.canvas.draw()

    def world_rules(self):
        """Placeholder method: Implement the desired world logic according to the desired scenario"""
        raise NotImplementedError

    def enforce_limits(self, entity: Entity, limits):
        """Check if the next position of one of the entities is inside its movement limits"""
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
        """Return a sensorization vector with the distance between the object in which the robot is focused and its
         actuator, the color of this object..."""
        raise NotImplementedError

    def get_scenario_data(self):
        """Scenario data needed to predict future states using the world model"""
        raise NotImplementedError


    def restart_scenario(self):
        """Set the scenario to the desired initial conditions"""
        raise NotImplementedError

    def restart_scenario_partially(self):
        """Partially reinitialize the scenario"""
        raise NotImplementedError

    @staticmethod
    def normalize_value(value, max_value, min_value=0.0):
        return (value - min_value) / (max_value - min_value)

    @staticmethod
    def get_relative_angle(x1_y1, x2_y2):
        """Return the relative angle between two points"""
        (x1, y1) = x1_y1
        (x2, y2) = x2_y2
        return math.atan2(y2 - y1, x2 - x1) * 180 / math.pi





class Baxter2Arms(Sim):
    def __init__(self, x_size=(0, 2500), y_size=(0, 1000), x_bounds=(100, 2400), y_bounds=(50, 800), visualize=True):
        super().__init__(x_size, y_size, x_bounds, y_bounds, visualize)
        #Define objects in simulation
        ## Baxter
        self.baxter_left=Robot("baxter_left", 700, 300, 90)
        self.baxter_right=Robot("baxter_right", 1800, 300, 90)
        self.robots=[self.baxter_left, self.baxter_right]
        self.baxter_left_limits=((100, 1250),self.y_bounds)
        self.baxter_right_limts=((1050, 2400),self.y_bounds)
        self.entities.extend(self.robots) #Include robots in entities list

    def move_robot_arm(self, arm:Robot, vel=10):
        """Move robot arm wih a specific velocity (default 10)"""
        x, y = arm.get_pos()
        arm.set_pos(x + vel * math.cos(arm.angle * math.pi / 180),
                                  y + vel * math.sin(arm.angle * math.pi / 180))

    def robot_arm_action(self, arm: Robot, relative_angle, vel=10):
        """Move robot arm with a specific angle and velocity (default 10)"""
        angle = arm.angle + relative_angle
        arm.set_angle(angle)
        self.move_robot_arm(arm, vel)
        self.world_rules()

    def apply_action(self, rel_angle_l=0, rel_angle_r=0, vel_left=20, vel_right=20, gripper_left=False, gripper_right=False):
        """Move arms with a specific angle and velocity (default 20)"""
        self.robot_arm_action(self.baxter_left, rel_angle_l, vel_left)
        self.robot_arm_action(self.baxter_right, rel_angle_r, vel_right)
        self.baxter_left.set_gripper(gripper_left)
        self.baxter_right.set_gripper(gripper_right)
        self.world_rules()
        if self.visualize:
            self.fig.canvas.draw()
    
    def get_close_object(self, robot:Robot, threshold=20):        
        for object in self.objects:
            dist=distance.euclidean(robot.get_pos(), object.get_pos())
            if dist<=threshold:
                return object
        return None



class ComplexScenario(Baxter2Arms):
    def __init__(self, x_size=(0, 2500), y_size=(0, 1000), x_bounds=(100, 2400), y_bounds=(50, 800), visualize=True):
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
        """Establish the ball position in the scenario"""

        #Enforce position bounds
        self.enforce_limits(self.baxter_left, self.baxter_left_limits)
        self.enforce_limits(self.baxter_right, self.baxter_right_limts)
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
    def __init__(self, x_size=(0, 2500), y_size=(0, 1000), x_bounds=(100, 2400), y_bounds=(50, 800), visualize=True):
        super().__init__(x_size, y_size, x_bounds, y_bounds, visualize)
        ##Objects
        self.objects=[]
        self.objects.append(Ball("ball_1", 300, 650, color="red"))
        self.entities.extend(self.objects)

        ##Boxes
        self.box1 = Box("box_1", 900, 400, w=150, h=150)
        self.entities.append(self.box1)
        
        # Show figure and patches
        self.plot_entities()

    def world_rules(self):
        """Establish the ball position in the scenario"""

        #Enforce position bounds
        self.enforce_limits(self.baxter_left, self.baxter_left_limits)
        self.enforce_limits(self.baxter_right, self.baxter_right_limts)
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
        plt.pause(0.1)



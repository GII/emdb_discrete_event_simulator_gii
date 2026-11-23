# e-MDB discrete event simulator to be used in simple experiments

This repository contains the packages that implement the discrete event simulators used to test concepts / algorithms in the cognitive architecture. They have no physics simulation at all.

These are the available simulators, with their ROS executables:

- **Put object in box** (`simulator_discrete`): A Baxter robot has to put a cylinder into a box.
- **Fruit shop** (`fruit_shop_simulator`): A TIAGo++ robot has to classify fruits by their weight.
- **Fruit serve** (`fruit_serve_simulator`): A TIAGo++ robot has to serve fruits.
- **Pump panel** (`pump_panel_simulator`): Operation of a panel to start a pumping plant. It is a work in progress.
- **Bartender** (`bartender_simulator`, `bartender_rl_simulator`): A robot has to serve drinks to its clients.

The package also includes the 2D scenarios (`simulators.scenarios_2D`) used by the 2D simulator in [emdb_experiments_gii](https://github.com/GII/emdb_experiments_gii).

The `physics_simulators` folder holds simulators that do have physics:

- **UR5e in MuJoCo** (package `ur5_mujoco`): A UR5e arm with a magnet, a Robotiq 2F-85 or an OnRobot 2FG7 gripper has to grasp a ball and lift it. The package holds the simulation (`ur5_mujoco.scenarios`) and the robot models; its e-MDB node is the *ur5_mujoco_emdb* package in [emdb_experiments_gii](https://github.com/GII/emdb_experiments_gii). See its [README](physics_simulators/ur5_mujoco/README.md).

There are three ROS 2 packages in this repository:

- **simulators:** Implementation of the discrete event simulators.
- **simulators_interfaces:** Services and messages definitions.
- **ur5_mujoco** (in `physics_simulators`): The MuJoCo UR5e simulator (scenarios and robot models).

You can find information and tutorials about this repository and the others that are part of the e-MDB cognitive architecture software in the [PILLAR Robots official documentation](https://docs.pillar-robots.eu/en/latest/)


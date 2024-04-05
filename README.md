# e-MDB discrete event simulator to be used in simple experiments

Note:

***WORK IN PROGRESS***

The original repository has been split in 5 and we are refactoring everything, please, be patient while we move and rename everything.

These are the cognitive architecture repositories for PILLAR and their content:

- _emdb_core_. Essential elements of the cognitive architecture. These are necessary to run an experiment using the cognitive architecture.
- _emdb_cognitive_nodes_gii_. Reference implementation for the main cognitive nodes.
- _emdb_cognitive_processes_gii_. Reference implementation for the main cognitive processes.
- _emdb_discrete_event_simulator_gii_. Implementation of a discrete event simulator used in many experiments.
- _emdb_experiments_gii_. Configuration files for experiments.

This repository includes a couple of ROS packages, one with a discrete event simulator and another one with a corresponding messages package. This simulator has no physics simulation at all. It is just a simulator that assumes a perfect world where nothing fails that it is used to test concepts / algorithms in the cognitive architecture. There will be more packages to use Gazebo...

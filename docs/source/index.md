This is part of the e-MDB architecture documentation. Main page [here.](https://docs.pillar-robots.eu/en/latest/)

# e-MDB discrete event simulator for GII's experiments

This [repository](https://github.com/pillar-robots/emdb_discrete_event_simulator_gii) contains packages which implement different discrete event simulators which are used in several experiments to test the components of the e-MDB cognitive architecture. These simulators have no physics simulation at all. They are just simulators that assume a perfect world where nothing fails and they are used to test concepts / algorithms.

There are two ROS 2 packages in this repository:

- **simulators:** Implementation of the discrete event simulators.
- **simulators_interfaces:** Services and messages definitions.

In this page you can only find the [API documentation](simulators/api_documentation.rst) of the simulators implemented. If you can know the details of the experiments, visit the [page](https://docs.pillar-robots.eu/projects/emdb_experiments_gii/en/latest/experiments/default_experiment.html) dedicated to them. 


```{toctree}
:caption: e-MDB discrete event simulator
:hidden:
:glob:

simulators/*
```
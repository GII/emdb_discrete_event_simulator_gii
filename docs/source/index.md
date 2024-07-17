This is part of the e-MDB architecture documentation. Main page [here.](https://docs.pillar-robots.eu/en/latest/)

# e-MDB discrete event simulator for GII's experiments

This [repository](https://github.com/pillar-robots/emdb_discrete_event_simulator_gii) contains the packages that implements a discrete event simulator of the Baxter robot. This simulator has no physics simulation at all. It is just a simulator that assumes a perfect world where nothing fails that it is used to test concepts / algorithms in the cognitive architecture.

We can found the documentation of an experiment using this simulator [here](https://docs.pillar-robots.eu/projects/emdb_experiments_gii/en/latest/experiments/default_experiment.html)

There are two ROS 2 packages in this repository:

- **simulators:** Implementation of discrete event simulator.
- **simulators_interfaces:** Services and messages definitions.

Here we can find the [API documentation](simulators/api_documentation.rst) of these packages.


```{toctree}
:caption: e-MDB discrete event simulator
:hidden:
:glob:

simulators/*
```
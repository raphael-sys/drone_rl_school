# Drone RL (Reinforcement Learning) School

In this project the drone will learn how to fly and achieve multiple user set goals.
And in the process there will be a chance to see multiple key concepts of Reinforcement Learning and 
modern software development.

## Environment
It all starts with a very simple drone 3D simulation that contains a drones position, velocity, acceleration
and a target position. 
In a first step the drone can only accelerate into one direction per step, so its available actions are
+a_x, -a_x, +a_y, -a_y, +a_z, -a_z
and only one action can be performed per step.

Gravity is currently not considered, as this constant offset can be offset easily and is not compatible with the
drone currently only being allowed one acceleration per step.

## Tasks and Control Algorithms
The first task for the drone is to fly to and hold the target position.

### Benchmark: Classic PID control
As a benchmark we have a deterministic PID controller from classic controls theory.
For an empty environment with the only task being to fly towards the goal this benchmark
is actually hard to beat if the controller is tuned well.

### Basic method with discrete state perception: Q-Learning
As a first basic method of RL 

### Continous perception using deep learning: DQN



## Toolchain
Besides testing out various highly interesting RL methods, the target of this project
is also to use modern engineering tools and toolchains.


### Versioning: git and gitHub
Basic versioning is performed in git right from the start of the project. 
For entirely new features we use feature branches, general improvements that will
not break code functionality will also be commited to master, as I am currently
working on this project alone.
The code is hosted on gitHub.

### Automatic Software Testing: PyTest
To handle the various agents, environments and project resources it is essential
to implement automatic software testing early on. Using PyTest we set up a system
where key functionality and general workflows are tested using this powerfull
and convenient library.
While currently test driven design is out of scope for this research project, 
the testing will be an important foundation for any further toolchain additions.

# Continous Integration of new features: CI/CD
A next step for this project will be the implementation of a CI toolchain.
This toolchain will help checking, formatting and testing code fully automatic on each push to the repository.

# Keywords
Reinforchement Learning, RL, Deep Learning, Deep Reinforcement Learning, Q-table, Q-Learning, DQN, Deep Q-Network, ...
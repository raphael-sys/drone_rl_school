# TODOs

## Upcoming tasks

### (Integrate a CI/CD setup)
- involve steps like linter, formatter, testing and so on (see also https://github.com/DLR-RM/stable-baselines3/blob/master/CONTRIBUTING.md and GPT conversation 'why use pytest')
- maybe use tox for this

### Integrate other RL pipeline components
- integrate the StableBaselines3 library as a selectable agent
- implement automatic parameter sweeps and run some grid searches

### Make world more realistic and prepare for new features
- allow for accelerations/actions in all directions simultaniously (not just one direction per step)
- allow for continous actions (instead of just discrete on/off actions)

### Improve difficulty for the agent
- introduce gravity
- introduce perception <-> introduce obstacles
    - decide perception type
    - implement basic perception
    - introduce perception into decision logic
    - introduce obstacles into the environment
    - implement a way of creating obstacles
    - adapt the evaluation function to end a simulation and negative reward on collision
    - the floor could be an obstacle

### Test out more RL techniques
- try out new and more advanced techniques

## Done tasks

### Improve existing RL methods
- the LR adaption currently has a local and global version
- adaption of LR and Epsilon are currently not used but instead decrease really slowly (really just a constant value)
    - fix and implement cleanly
    - evaluate effect of this change

### Improve the reward function for dqn
- reward for dqn currently does not work for values really close to the target
    - fix and evaluate
    - try to beat or compare to pid agent
    
### Introduce pytest as a testing framework
- install and set up pytest
- write a few tests for basic components and run them

![Drag Racing](http://github.com/rodSiry/bagarreIO/miniature.jpg)

# bagarre IO

_Solve violence first, then solve anything else with violence._

**bagarreIO** is a library that provides carefully designed environments to train and compare humanoid AI agents on the challenging task of **1v1 close combat**.

## Installation

### Dependencies
1. gym
2. mujoco

## Environment Description & Fighting rules

Two 17-dof simulated humanoids must rekt each other according to the following principles : 

1. Both agents can see the enemy state in addition of their own
2. Rewards are distributed depending on 1) opponent body part : head typically gives the highest score and hands the lowest and 2) the contact force measured during impact.
3. Note that we may add or modify rules progressively to avoid boring solutions

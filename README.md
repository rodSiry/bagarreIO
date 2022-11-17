# bagarre IO

_Solve violence first, then solve anything else with violence._

**bagarreIO** is a library that provides carefully designed environments to train and compare humanoid AI agents on the challenging task of **1v1 close combat**.

## Installation

### Dependencies
gym,
mujoco

## Environment Description & Fighting rules

Two 17-dof simulated humanoids must rekt each other : 
Rewards are distributed depending on 1) opponent body part : head gives the highest score and hands the lowest and 2) the contact force measured during impact.

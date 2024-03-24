# Imitation Learning for Superhuman Performance

<p align="center">
  <img src="/Assets/great.gif" width="30%" alt="Superhuman Imitation Learning - Great Performance" />
  <img src="/Assets/no_fall.gif" width="30%" alt="Superhuman Imitation Learning - No Fall" />
  <img src="/Assets/no_energy_waste.gif" width="30%" alt="Superhuman Imitation Learning - Efficient Energy Usage" />
</p>

This project attempts to implement and extend the methodologies discussed in the paper ["Towards Uniformly Superhuman Autonomy via Subdominance Minimization"](https://proceedings.mlr.press/v162/ziebart22a/ziebart22a.pdf), specifically applying the concepts to the OpenAI Gym game called Bipedal Walker instead of applying it on a mouse cursor task as the paper did.

## Installation

To set up the necessary environment for this project, follow these steps:

**Install OpenAI Gym:**

   First, you need to install the OpenAI Gym package. You can do this using pip:

pip install gym

**Replace Gym Files:**

After installing the OpenAI Gym, you'll need to replace some of its files with the modified versions provided in this repository. Specifically, replace the following files in your Gym installation with the ones from this repo:

- `bipedal_walker.py`
- `evaluation.py`
- `on_policy_algorithm.py`
- `ppo.py`

**Prepare the Models:**

Ensure the `Training/Saved Models` directory contains a collection of saved models. Each model should excel at one specific feature or aspect of the game, as these will be the basis for the imitation learning process.

## Running the Project

To run the project and start the training process, execute the following command:

python superhuman_imitation.py




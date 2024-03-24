import gym
import os 
import wandb
from gym.wrappers import GrayScaleObservation
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecVideoRecorder, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import numpy as np
import torch
import pdb
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import TD3





env_name = "BipedalWalker-v3"
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps":300000,
    "env_name": env_name,
}

run = wandb.init(
    project="intro_to_gym",
    config=config,
    sync_tensorboard=True,  
    monitor_gym=True,  
    save_code=True,  
)
def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env) 
    return env


env = DummyVecEnv([make_env])

#Write down all the paths
rootdir = "/Users/ammarpl/Documents/superhuman/Training/Saved Models"
paths = []
for root, dirs, files in os.walk(rootdir):
 for f in files:
   paths.append(os.path.join('Training', 'Saved Models', f.split(".zip")[0]))

print(paths)
n_samples = len(paths)
n_trajectories = 3

# model = PPO.load(path)
models = []
for i in range(n_samples):
    print("# N: ", i)
    models.append(PPO.load(paths[i]))
    # model2 = PPO.load(PPO_path2)

# Set the initial alpha values
alpha = [1.33 , 1.33, 1.33]


# Number of iterations
n_iterations = 100

# Learning rate
eta_t = 0.01


# Regularization parameter
lambda_reg = 0.1
rewards = [0 for i in range(n_samples*n_trajectories)]
falls = [0 for i in range(n_samples*n_trajectories)]
motors = [0 for i in range(n_samples*n_trajectories)]
positions = [0 for i in range(n_samples*n_trajectories)]
for i in range(n_samples):
    print("# N: ", i)
    for j in range(n_trajectories):
        rewards[i*n_trajectories+j],_ , falls[i*n_trajectories+j], motors[i*n_trajectories+j]= evaluate_policy(models[i], env, n_eval_episodes=1)
        positions[i*n_trajectories+j] = rewards[i*n_trajectories+j] - falls[i*n_trajectories+j] - motors[i*n_trajectories+j]
        # reward2,rewardstd, fall2,motor2= evaluate_policy(model2, env, n_eval_episodes=100)


print("rewards:", rewards)
print("falls:", falls)
print("motors:", motors)
print("positions:", positions)

# reward,rewardstd, fall,motor = evaluate_policy(model, env, n_xeval_episodes=4, render=True)
print(rewards,falls,motors,positions)

w= np.random.uniform(low= 0.1, high=2.0, size=(3,))
w = np.array([1.0, 1.0, 1.0])
print("initial weight vector:",w)

super_i = 0
history = []
history_fall = []
history_motor = []
history_position = []

for iteration in range(n_iterations):
        


    model = TD3(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")

    # Learn from the environment
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2
        ),
        log_interval=25000 // 2048,  # print the training info every 25000 timesteps
        weights=w
    )
    reward,rewardstd, fall,motor = evaluate_policy(model, env, n_eval_episodes=25)
    position = reward-fall-motor
    history.append(reward)
    history_fall.append(fall)
    history_motor.append(motor)
    history_position.append(position)
    
    print("Iteration",iteration+1)
    print("reward",reward,"fall",fall,"motor",motor, "position",position,"different_reward",)
    
    sup_vecs = [[],[],[]]
    
    for i in range(n_samples*n_trajectories):
        #k=0
        if falls[i]>= (alpha[0]/(alpha[0]-1)) * fall and len(sup_vecs[0])< int(0.1*n_samples*n_trajectories):
             sup_vecs[0].append(i)
             continue
        #k=1
        if motors[i]>= ((alpha[1])/(alpha[1]-1)) * motor and len(sup_vecs[1])< int(0.1*n_samples*n_trajectories):
             sup_vecs[1].append(i)
             continue
        #k=2
        if positions[i]>= ((alpha[2]-1)/(alpha[2])) * position and len(sup_vecs[2])< int(0.1*n_samples*n_trajectories):
             sup_vecs[2].append(i)
             continue
        
    print(sup_vecs)
    if sup_vecs[1] == []:
         print("The alpha value for falls or motors might be bad")
    
    #K = 0
    for j in range(len(sup_vecs[0])):
        f_k_xi_tilde = falls[sup_vecs[0][j]]
        f_k_xi_star = fall
        # Update alpha_k
        alpha[0] *= np.exp(eta_t * (f_k_xi_tilde - f_k_xi_star - lambda_reg * alpha[0]))

        # Update w_k
        w[0] *= np.exp(eta_t * alpha[0])
        w[0] = min(100, w[0])
    

    #K = 1
    for j in range(len(sup_vecs[1])):
        f_k_xi_tilde = motors[sup_vecs[1][j]]
        f_k_xi_star = motor

        # Update alpha_k
        alpha[1] *= np.exp(eta_t * (f_k_xi_tilde - f_k_xi_star - lambda_reg * alpha[1]))

        # Update w_k
        w[1] *= np.exp(eta_t * alpha[1])
    
    #K = 2
    for j in range(len(sup_vecs[2])):
        f_k_xi_tilde = positions[sup_vecs[2][j]]
        f_k_xi_star = position

        # Update alpha_k
        alpha[2] *= np.exp(eta_t * (f_k_xi_tilde - f_k_xi_star - lambda_reg * alpha[2]))

        # Update w_k
        w[2] *= np.exp(eta_t * alpha[2])

    print("weights after iteration " + str(iteration)+ " =",w)
    
    if iteration > 10 and history[-2] < [-3] and history[-2] < history[-1]:
        print("model converged","support vector:",f_k_xi_tilde,"our model",f_k_xi_star)
        break
        

PPO_path = os.path.join('Training', 'Saved Models', f'Superhuman_BipedalWalker_{timesteps}_iteration_{super_i}')
model.save(PPO_path)

with open(f'QSuperhuman_BipedalWalker_{config["total_timesteps"]}_iteration_{super_i}_history.txt', "w") as fp:
    fp.writelines(str(item) + "\n" for item in history)
with open(f'QSuperhuman_BipedalWalker_{config["total_timesteps"]}_iteration_{super_i}_history_motor.txt', "w") as fp:
    fp.writelines(str(item) + "\n" for item in history_motor)
with open(f'QSuperhuman_BipedalWalker_{config["total_timesteps"]}_iteration_{super_i}_history_fall.txt', "w") as fp:
    fp.writelines(str(item) + "\n" for item in history_fall)
with open(f'QSuperhuman_BipedalWalker_{config["total_timesteps"]}_iteration_{super_i}_history_position.txt', "w") as fp:
    fp.writelines(str(item) + "\n" for item in history_position)

evaluate_policy(model, env, n_eval_episodes=1, render=True)
run.finish()

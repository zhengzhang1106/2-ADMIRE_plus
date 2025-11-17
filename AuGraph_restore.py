import numpy as np
import random
import torch
import ray
from ray.rllib.agents.ppo import PPOTrainer
from AuGraph_env_restore import AuGraphEnvRestore
from ray.rllib.models.catalog import ModelCatalog
from AuGraph_model import AuGraphModel
import Restore_path

## 设置随机种子
seed_num = 1
np.random.seed(seed_num)
random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)

## 运行ray
ray.shutdown()
ray.init()
ModelCatalog.register_custom_model('augraph_model', AuGraphModel)  # 使用自定义模型

config_re = {
    'env': AuGraphEnvRestore,
    'framework': 'torch',
    'seed': seed_num,
    # 'num_gpus': int(os.environ.get("RLLIB_NUM_GPUS", "0")),  # GPU
    'num_gpus': 0,  # GPU，需要<1

    # Should use a critic as a baseline (otherwise don't use value baseline;
    # required for using GAE).
    "use_critic": True,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # The GAE (lambda) parameter.
    "lambda": 0.9,
    # Initial coefficient for KL divergence.
    "kl_coeff": 0.2,
    # Size of batches collected from each worker.
    "rollout_fragment_length": 256,
    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    "train_batch_size": 2048,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 128,
    # Whether to shuffle sequences in the batch when training (recommended).
    "shuffle_sequences": True,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 15,
    # Stepsize of SGD.
    "lr": 1e-4,  # tune.grid_search([1e-4, 5e-5]),
    # Learning rate schedule.
    "lr_schedule": None,
    # Coefficient of the value function loss. IMPORTANT: you must tune this if
    # you set vf_share_layers=True inside your model's config.
    "vf_loss_coeff": 1.0,
    # Coefficient of the entropy regularizer.
    "entropy_coeff": 0.01,
    # Decay schedule for the entropy regularizer.
    "entropy_coeff_schedule": None,
    # PPO clip parameter.
    "clip_param": 0.2,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 10,
    # If specified, clip the global norm of gradients by this amount.
    "grad_clip": 0.5,
    # Target value for KL divergence.
    "kl_target": 0.01,
    # Whether to rollout "complete_episodes" or "truncate_episodes".
    "batch_mode": "truncate_episodes",
    # Which observation filter to apply to the observation.
    "observation_filter": "NoFilter",

    # Deprecated keys:
    # Share layers for value function. If you set this to True, it's important
    # to tune vf_loss_coeff.
    # Use config.model.vf_share_layers instead.
    # "vf_share_layers": DEPRECATED_VALUE,

    # ========= Model ============
    # 在进入actor和critic的隐藏层之前，会先运行'model'里的参数
    # "use_state_preprocessor": True,     # 可以使用自定义model
    # 自定义模型
    'model': {
        'custom_model': 'augraph_model',
        "post_fcnet_hiddens": [256, 128],
        "post_fcnet_activation": 'relu',  # tune.grid_search(['relu','tanh'])
    },

    'gamma': 0.98,      # 奖励衰减
    # 'timesteps_per_iteration': 100,    # 每次迭代100个step

    # === Exploration Settings ===
    "exploration_config": {
        "type": "ICM",  # <- Use the Curiosity module for exploring.
        "eta": 0.1,   # Weight for intrinsic rewards before being added to extrinsic ones.
        "lr": 1e-4,  # Learning rate of the curiosity (ICM) module.
        "feature_dim": 128,   # Dimensionality of the generated feature vectors.
        # Setup of the feature net (used to encode observations into feature (latent) vectors).
        "feature_net_config": {
            "fcnet_hiddens": [256, 128],
            "fcnet_activation": "relu",
        },
        "inverse_net_hiddens": [128, 128],
        "inverse_net_activation": "relu",  # Activation of the "inverse" model.
        "forward_net_hiddens": [128, 128],  # Hidden layers of the "forward" model.
        "forward_net_activation": "relu",  # Activation of the "forward" model.
        "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
        # Specify, which exploration sub-type to use (usually, the algo's "default"
        # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
        "sub_exploration": {
            "type": "StochasticSampling",
        },
        # "type": "StochasticSampling",
    },
    # Switch to greedy actions in evaluation workers.
    "evaluation_config": {
        "explore": False,
    },

    "num_workers": 0,
    # Prevent iterations from going lower than this time span
    "min_iter_time_s": 1,
}

path = Restore_path.path

# 恢复经过训练的agent
agent = PPOTrainer(config=config_re, env=AuGraphEnvRestore)
agent.restore(path)
env = AuGraphEnvRestore({})
episode_reward = 0
done = False
obs = env.reset()

count = 0  # 可以在测试的时候也多跑几次，取一个最好的，把explore开启之后是能选到训练时最好的路径
reward_max = -10000
reward_list = []    # 奖励集合
while count < 20:
    virtual_hop_cumulate = 0  # 累计虚拟拓扑跳数
    virtual_hop_cumulate_list = []  # 统计结果
    while not done:
        action = agent.compute_action(obs)
        # action = agent.compute_action(obs,explore=False)
        obs, reward, done, info = env.step(action)
        # print("Action:", action*1000)
        # print("State:", obs)
        # print("Reward:", reward)
        print('------')
        episode_reward += reward
        print("Total Reward:", episode_reward)

    reward_list.append(episode_reward/50)
    count += 1
    episode_reward = 0
    obs = env.reset()
    done = False

print("奖励列表", reward_list)

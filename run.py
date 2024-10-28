import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib import animation
from env_zj2 import ZJEnv
from tqdm import tqdm
import json


def save_frames_as_gif(frames, path, filename):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=300)
    anim.save(path + filename, writer='imagemagick')


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1024
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: int = int(1)
    """the environment id of the task"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 1e4
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def make_env(env_id, seed):
    # 将一个函数的计算结果存储起来，以便在后续调用时直接使用，而不需要重新计算
    env = ZJEnv(env_id)
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    convergent = 0

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    print("-------Starting RL------")
    print(f"Exploration:{args.learning_starts}, Total:{args.total_timesteps}")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    # SummaryWriter 用于将训练过程中的数据以可视化的方式记录下来，通常用于TensorBoard可视化工具中
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # 设计随机数
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # 使用cuda
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # 设置环境
    # 用于创建一个同步化的向量化环境
    envs = make_env(args.env_id, args.seed)
    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    # 策略网络
    actor = Actor(envs).to(device)
    # 创建两个Q函数网络  Critic  用于估计在给定状态下采取某动作后获得的期望累积奖励（Q值）
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    # 创建两个目标Q函数网络（稳定训练-仅偶尔与主网络（qf1和qf2）进行参数更新）
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    # 将主Q函数网络的参数加载到目标Q函数网络
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # 目标Q函数网络与主Q函数网络具有相同的初始参数。在后续训练过程中，目标网络的参数将按一定策略（如软更新策略）逐步与主网络同步，以保持一定程度的滞后性。
    # 定义了优化器（optimizer）来更新所创建的神经网络的参数
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    # 基于动作熵 来鼓励策略探索更多状态-动作空间
    if args.autotune:
        # 计算目标熵
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        # 创建对数动作熵系数
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        # 计算当前的动作熵系数
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # envs.single_observation_space.dtype = np.float32

    # 定义经验池
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # 存储路径
    save_gif = './resultgif/' + envs.name + '/'

    if not os.path.exists(save_gif):
        # 如果不存在则创建文件夹
        os.mkdir(save_gif)

    # 开始游戏
    state = envs.reset()
    for global_step in tqdm(range(args.total_timesteps)):
        # ALGO LOGIC: put action logic here

        # 先探索 到达 learning_starts 步后采用神经网络开始学习
        if global_step < args.learning_starts:
            # 探索 在环境里随机走
            actions = np.array(envs.single_action_space.sample())
        else:
            # 拿着障碍去神经网络里学习
            actions, _, _ = actor.get_action(torch.Tensor(state).unsqueeze(0).to(device))
            actions = actions[0].detach().cpu().numpy()

        # TRY NOT TO MODIFY: 执行游戏并记录
        next_state, reward, done, _ = envs.step(actions, False)

        # TRY NOT TO MODIFY: 记录奖励以用于绘图
        # if "final_info" in infos:
        #     for info in infos["final_info"]:
        #         print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        #         # 日志写入器对象
        #         writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        #         writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        #         break

        # TRY NOT TO MODIFY: 将数据保存到 to reply buffer; 处理 `final_observation`
        real_next_state = next_state.copy()
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(state, real_next_state, actions, reward, done, _)
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        # if reward != -100:
        if not done:
            state = next_state

        # ALGO LOGIC: training.
        # learning_starts 步之后开始学习
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                        args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # 更新 the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # if global_step % 100 == 0:
            #     writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            #     writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            #     writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            #     writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            #     writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            #     writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            #     writer.add_scalar("losses/alpha", alpha, global_step)
            #     # print("SPS:", int(global_step / (time.time() - start_time)))
            #     writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            #
            #     if args.autotune:
            #         writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

            # 验证

            if global_step % 500 == 0:
                print(f"-----------global_step【{global_step}】测试-----------")
                envs.init_pygame()
                state = envs.reset()
                frames = []
                path = []
                first = True
                done = False
                while not done:
                    path.append((state[1], state[0]))
                    _, _, actions = actor.get_action(torch.Tensor(state).unsqueeze(0).to(device))
                    actions = actions.detach().cpu().numpy()
                    next_state, reward, done, _ = envs.step(actions[0], True)
                    print(f"State: \t{state}, actions: \t{actions},Next State: \t{next_state}, info: \t{_}, Reward: \t{reward}, Done: \t{done}")
                    if first:
                        frames.append(envs.render())
                        first = False
                    else:
                        com = envs.render()
                        if np.array_equal(frames[len(frames) - 1], com):
                            pass
                        else:
                            frames.append(com)
                    state = next_state
                if _ == "Get!":
                    print(f"________________Get! in {global_step}_________________")
                    # 如果到达的次数达到 convergent 次 大概说明收敛 以这个结果作为规划的结果
                    convergent += 1
                    if convergent > 2:
                        print(f"------------Number:{args.env_id} is finish-------------")
                        path.append((envs.destination[1], envs.destination[0]))

                        item = {}
                        item['index'] = args.env_id
                        item['coordinate'] = path

                        with open(f'pathResult\\{args.env_id}.json', 'w') as json_file:
                            json.dump(item, json_file, indent=4)

                        # 然后就结束程序
                        exit(0)
                else:
                    convergent = 0
                save_frames_as_gif(frames, save_gif, str(global_step) + 'epoch_______'+_+'______.gif')
                print("global_step:", int(global_step), " gif...")

                # 验证完回到起点
                state = envs.reset()

    envs.close()
    writer.close()

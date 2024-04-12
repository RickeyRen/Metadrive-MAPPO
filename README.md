# Config.py 参数解释

## 准备参数
- `--algorithm_name` : 指定要使用的算法,可选值为"rmappo"(递归多智能体近端策略优化)或"mappo"(多智能体近端策略优化),默认为"mappo"。
- `--experiment_name` : 用于区分不同实验的标识符,默认为"check"。
- `--seed` : 用于设置numpy和torch的随机种子,默认为1。
- `--cuda` : 布尔型,默认为True,使用GPU进行训练;否则使用CPU。
- `--cuda_deterministic` : 布尔型,默认为True,确保随机种子有效;若设置为False,则会绕过该功能。
- `--n_training_threads` : 用于训练的torch线程数,默认为5。
- `--n_rollout_threads` : 用于训练rollout的并行环境数,默认为1。
- `--n_eval_rollout_threads` : 用于评估rollout的并行环境数,默认为1。
- `--n_render_rollout_threads` : 用于渲染rollout的并行环境数,默认为1。
- `--num_env_steps` : 训练的环境步数,默认为1000万步。
- `--user_name` : 用于wandb使用,指定用户名以简单收集训练数据,默认为"marl"。

## 环境参数
- `--env_name` : 指定环境名称,默认为"COMP0124"。
- `--use_obs_instead_of_state` : 布尔型,默认为False,使用全局状态;若设置为True,将使用连接的观测值。

## 回放缓冲区参数 
- `--episode_length` : 任何回合的最大长度,默认为1000。

## 网络参数
- `--share_policy` : 布尔型,默认为False,控制是否所有智能体共享同一策略。
- `--use_centralized_V` : 布尔型,默认为True,使用集中式值函数估计。
- `--stacked_frames` : 用于actor/critic网络隐藏层的维度,默认为1。  
- `--use_stacked_frames` : 布尔型,默认为False,控制是否使用堆叠帧。
- `--hidden_size` : actor/critic网络隐藏层的维度,默认为256。
- `--layer_N` : actor/critic网络的层数,默认为3。
- `--use_ReLU` : 布尔型,默认为False,控制是否使用ReLU激活函数。
- `--use_popart` : 布尔型,默认为False,控制是否使用PopArt对奖励进行归一化。
- `--use_valuenorm` : 布尔型,默认为True,控制是否使用运行均值和标准差对奖励进行归一化。
- `--use_feature_normalization` : 布尔型,默认为True,控制是否对输入应用层归一化。
- `--use_orthogonal` : 布尔型,默认为True,控制是否使用正交权重初始化和偏置为0的初始化。
- `--gain` : 最后一个动作层的增益,默认为0.01。

## 循环策略参数
- `--use_naive_recurrent_policy` : 布尔型,默认为False,控制是否使用一个简单的循环策略。
- `--use_recurrent_policy` : 布尔型,默认为False,控制是否使用循环策略。
- `--recurrent_N` : 循环层的数量,默认为1。
- `--data_chunk_length` : 用于训练循环策略的数据块长度,默认为10。

## 优化器参数
- `--lr` : 学习率,默认为5e-4。
- `--critic_lr` : critic网络的学习率,默认为5e-4。
- `--opti_eps` : RMSprop优化器的epsilon值,默认为1e-5。
- `--weight_decay` : 权重衰减系数,默认为0。

## PPO参数
- `--ppo_epoch` : PPO的循环次数,默认为15。
- `--use_clipped_value_loss` : 布尔型,默认为True,裁剪值损失;若设置,则不裁剪值损失。
- `--clip_param` : PPO裁剪参数,默认为0.2。
- `--num_mini_batch` : PPO的小批次数,默认为1。
- `--entropy_coef` : 熵系数,默认为0.01。
- `--value_loss_coef` : 值损失系数,默认为1。
- `--use_max_grad_norm` : 布尔型,默认为True,使用梯度的最大范数;若设置,则不使用。
- `--max_grad_norm` : 梯度的最大范数,默认为10.0。
- `--use_gae` : 布尔型,默认为True,使用广义优势估计。
- `--gamma` : 奖励的折现因子,默认为0.99。
- `--gae_lambda` : GAE的lambda参数,默认为0.95。
- `--use_proper_time_limits` : 布尔型,默认为False,计算回报时考虑时间限制。
- `--use_huber_loss` : 布尔型,默认为True,使用Huber损失;若设置,则不使用Huber损失。
- `--use_value_active_masks` : 布尔型,默认为True,控制是否在值损失中屏蔽无用数据。
- `--use_policy_active_masks` : 布尔型,默认为True,控制是否在策略损失中屏蔽无用数据。
- `--huber_delta` : Huber损失的系数,默认为10.0。

## 运行参数
- `--use_linear_lr_decay` : 布尔型,默认为False,控制是否对学习率使用线性衰减策略。

## 保存参数
- `--save_interval` : 连续两次保存模型之间的时间间隔,默认为1。  

## 日志参数
- `--log_interval` : 连续两次打印日志之间的时间间隔,默认为5。

## 评估参数 
- `--use_eval` : 布尔型,默认为True,控制是否在训练过程中启动评估。
- `--eval_interval` : 连续两次评估进度之间的时间间隔,默认为100。
- `--eval_episodes` : 单次评估的回合数,默认为10。

## 渲染参数
- `--save_gifs` : 布尔型,默认为False,控制是否保存渲染视频。
- `--use_render` : 布尔型,默认为False,控制是否在训练期间渲染环境。
- `--use_render_eval` : 布尔型,默认为True,控制是否在评估期间渲染环境。
- `--render_episodes` : 渲染给定环境的回合数,默认为2。
- `--ifi` : 保存视频中每个渲染图像的播放间隔,默认为0.1。

## 预训练参数
- `--model_dir` : 设置预训练模型的路径,默认为None。

## 环境选择参数
- `--env` : 指定环境,可选值为envs字典的键,默认为"roundabout"。
- `--top_down` : 布尔型,默认为True,使用上视角。  
- `--num_agents` : 智能体数量,默认为2。
- `--random_traffic` : 布尔型,默认为True,在道路上随机放置其他车辆。
- `--human_vehicle` : 布尔型,默认为True,道路上有其他车辆。
- `--traffic_density` : 字典类型,默认为density变量,指定交通密度。
- `--obs_num_others` : 智能体观测其他智能体的数量,默认为4。
- `--show_navi` : 布尔型,默认为True,显示导航标记。
- `--show_dest` : 布尔型,默认为True,显示目的地标记。

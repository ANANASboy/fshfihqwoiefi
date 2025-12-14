import os
import copy
import argparse
import yaml
import shutil
import numpy as np
import torch as th
import datetime
from tensorboardX import SummaryWriter

from models.model_registry import Model, Strategy
from environments.var_voltage_control.voltage_control_env import VoltageControl
from utilities.util import convert, dict2str, translate_action
from utilities.trainer import PGTrainer
from utilities.vec_env import SubprocVecEnv


def make_env(env_config, rank):
    """Factory to create env with rank-dependent seed."""
    def _thunk():
        cfg = copy.deepcopy(env_config)
        base_seed = cfg.get("seed", 0)
        cfg["seed"] = base_seed + rank * 1000
        return VoltageControl(cfg)
    return _thunk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train rl agent.")
    parser.add_argument("--save-path", type=str, nargs="?", default="./results", help="Please enter the directory of saving model.")
    parser.add_argument("--alg", type=str, nargs="?", default="matd3", help="Please enter the alg name.")
    parser.add_argument("--env", type=str, nargs="?", default="var_voltage_control", help="Please enter the env name.")
    parser.add_argument("--alias", type=str, nargs="?", default="train_run_1", help="Please enter the alias for exp control.")
    parser.add_argument("--mode", type=str, nargs="?", default="distributed", help="Please enter the mode: distributed or decentralised.")
    parser.add_argument("--scenario", type=str, nargs="?", default="case33_3min_final", help="Please input the valid name of an environment scenario.")
    parser.add_argument("--voltage-barrier-type", type=str, nargs="?", default="bowl", help="Please input the valid voltage barrier type: l1, courant_beltrami, l2, bowl or bump.")
    # Learning rate & decay (with lower bound)
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate (if not overridden by config).")
    parser.add_argument("--lr-decay-step", type=int, default=1000, help="StepLR step size in episodes.")
    parser.add_argument("--lr-decay-gamma", type=float, default=0.95, help="StepLR decay factor.")
    parser.add_argument("--lr-min", type=float, default=1e-5, help="Lower bound for learning rate.")
    parser.add_argument("--save-top-k", type=int, default=20, help="Keep top-K models by mean_test_reward.")
    # Decoupled exploration noise (PV / ESS) with decay
    parser.add_argument("--pv-noise-start", type=float, default=0.5, help="Initial std for PV exploration noise.")
    parser.add_argument("--pv-noise-end", type=float, default=0.01, help="Minimum std for PV exploration noise.")
    parser.add_argument("--pv-noise-decay", type=float, default=0.9995, help="Per-step decay factor for PV noise.")
    parser.add_argument("--ess-noise-start", type=float, default=0.3, help="Initial std for ESS exploration noise.")
    parser.add_argument("--ess-noise-end", type=float, default=0.01, help="Minimum std for ESS exploration noise.")
    parser.add_argument("--ess-noise-decay", type=float, default=0.9998, help="Per-step decay factor for ESS noise.")
    # Multiprocessing / data params
    parser.add_argument("--n-rollout-threads", type=int, default=16, help="Number of parallel env processes.")
    parser.add_argument("--batch-size", type=int, default=256, help="Replay batch size.")
    parser.add_argument("--updates-per-step", type=int, default=4, help="Number of updates per data collection step.")
    # [新增] 用于指定加载的模型路径，默认不加载
    parser.add_argument("--load-model", type=str, nargs="?", default=None, help="Path to the model.pt to resume training.")
    parser.add_argument("--load-log", type=str, nargs="?", default=None, help="Path to the OLD tensorboard dir to copy history from.")
    parser.add_argument("--start-episode", type=int, default=0, help="Force start episode number.")
    argv = parser.parse_args()

    print(f"CUDA Available: {th.cuda.is_available()}")
    if th.cuda.is_available():
        print(f"GPU Name: {th.cuda.get_device_name(0)}")

    # load env args
    with open("./args/env_args/" + argv.env + ".yaml", "r") as f:
        env_config_dict = yaml.safe_load(f)["env_args"]
    data_path = env_config_dict["data_path"].split("/")
    data_path[-1] = argv.scenario
    env_config_dict["data_path"] = "/".join(data_path)
    net_topology = argv.scenario

    # set the action range
    assert net_topology in ['case33_3min_final', 'case141_3min_final', 'case322_3min_final'], f'{net_topology} is not a valid scenario.'
    if argv.scenario == 'case33_3min_final':
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.8
    elif argv.scenario == 'case141_3min_final':
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.6
    elif argv.scenario == 'case322_3min_final':
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.8

    assert argv.mode in ['distributed', 'decentralised'], "Please input the correct mode, e.g. distributed or decentralised."
    env_config_dict["mode"] = argv.mode
    env_config_dict["voltage_barrier_type"] = argv.voltage_barrier_type

    # load default args
    with open("./args/default.yaml", "r") as f:
        default_config_dict = yaml.safe_load(f)

    # load alg args
    with open("./args/alg_args/" + argv.alg + ".yaml", "r") as f:
        alg_config_dict = yaml.safe_load(f)["alg_args"]
        alg_config_dict["action_scale"] = env_config_dict["action_scale"]
    alg_config_dict["action_bias"] = env_config_dict["action_bias"]

    log_name = "-".join([argv.env, net_topology, argv.mode, argv.alg, argv.voltage_barrier_type, argv.alias])
    alg_config_dict = {**default_config_dict, **alg_config_dict}
    # Inject decoupled noise hyperparameters
    alg_config_dict["pv_noise_start"] = argv.pv_noise_start
    alg_config_dict["pv_noise_end"] = argv.pv_noise_end
    alg_config_dict["pv_noise_decay"] = argv.pv_noise_decay
    alg_config_dict["ess_noise_start"] = argv.ess_noise_start
    alg_config_dict["ess_noise_end"] = argv.ess_noise_end
    alg_config_dict["ess_noise_decay"] = argv.ess_noise_decay
    # Learning rate decay parameters (with lower bound)
    alg_config_dict["lr"] = argv.lr
    alg_config_dict["lr_decay_step"] = argv.lr_decay_step
    alg_config_dict["lr_decay_gamma"] = argv.lr_decay_gamma
    alg_config_dict["lr_min"] = argv.lr_min
    # Multiprocessing / data params
    alg_config_dict["n_rollout_threads"] = argv.n_rollout_threads
    alg_config_dict["batch_size"] = argv.batch_size
    alg_config_dict["updates_per_step"] = argv.updates_per_step

    # define a single env for shape info & trainer
    env_single = VoltageControl(copy.deepcopy(env_config_dict))

    alg_config_dict["agent_num"] = env_single.get_num_of_agents()
    alg_config_dict["obs_size"] = env_single.get_obs_size()
    alg_config_dict["action_dim"] = env_single.get_total_actions()
    args = convert(alg_config_dict)

    # define the save path
    if argv.save_path[-1] == "/":
        save_path = argv.save_path
    else:
        save_path = argv.save_path + "/"

    # create dirs
    for d in ["model_save", "tensorboard"]:
        if d not in os.listdir(save_path):
            os.mkdir(save_path + d)

    # TensorBoard dir
    tb_save_path = save_path + "tensorboard/" + log_name
    if log_name not in os.listdir(save_path + "tensorboard/"):
        os.mkdir(tb_save_path)
        print(f"📁 Created new TensorBoard dir: {tb_save_path}")
        if argv.load_log is not None and os.path.exists(argv.load_log):
            print(f"📃 Copying history logs from: {argv.load_log} ...")
            for file_name in os.listdir(argv.load_log):
                if "tfevents" in file_name:
                    src = os.path.join(argv.load_log, file_name)
                    dst = os.path.join(tb_save_path, file_name)
                    shutil.copy2(src, dst)
            print("✅ History logs copied! TensorBoard curves will be continuous.")
    else:
        print(f"⚠️ Warning: Log dir {log_name} exists. Cleaning up...")
        for f in os.listdir(tb_save_path):
            os.remove(os.path.join(tb_save_path, f))

    # model save dir
    model_save_path = save_path + "model_save/" + log_name
    if log_name not in os.listdir(save_path + "model_save/"):
        os.mkdir(model_save_path)

    # Logger
    logger = SummaryWriter(tb_save_path)

    model = Model[argv.alg]
    strategy = Strategy[argv.alg]

    print(f"{args}\n")

    # Top-K checkpoint tracking: (reward, episode, file_path)
    top_k_checkpoints = []

    if strategy == "pg":
        train = PGTrainer(args, model, env_single, logger)
    elif strategy == "q":
        raise NotImplementedError("This needs to be implemented.")
    else:
        raise RuntimeError("Please input the correct strategy, e.g. pg or q.")

    # Resume logic
    start_episode = 0
    if argv.load_model is not None and os.path.exists(argv.load_model):
        print(f"🔌 Loading checkpoint from: {argv.load_model} ...")
        checkpoint = th.load(argv.load_model, map_location=train.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            train.behaviour_net.load_state_dict(checkpoint["model_state_dict"])
        else:
            train.behaviour_net.load_state_dict(checkpoint)
        print("✅ Model weights loaded successfully!")

        if isinstance(checkpoint, dict):
            try:
                if "policy_optimizer_state_dict" in checkpoint:
                    train.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
                if "value_optimizer_state_dict" in checkpoint:
                    train.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
                if "mixer_optimizer_state_dict" in checkpoint and hasattr(train, 'mixer_optimizer'):
                    train.mixer_optimizer.load_state_dict(checkpoint["mixer_optimizer_state_dict"])
                print("✅ Optimizer state loaded (Smooth Resume).")
            except Exception as e:
                print(f"⚠️ Warning: Could not load optimizer state ({e}). Training will restart momentum.")

            if "episode" in checkpoint:
                start_episode = checkpoint["episode"] + 1
                print(f"⏭ Resuming training from Episode {start_episode}...")

            if argv.start_episode > 0:
                start_episode = argv.start_episode
                print(f"⚠️ Manually set start episode to: {start_episode}")

            train.episodes = start_episode
            train.steps = start_episode * args.max_steps

        print("Resume training setup complete.\n")
    else:
        if argv.load_model:
            print(f"❌ Error: Model path '{argv.load_model}' does not exist!")

    with open(save_path + "tensorboard/" + log_name + "/log.txt", "w+") as file:
        alg_args2str = dict2str(alg_config_dict, 'alg_params')
        env_args2str = dict2str(env_config_dict, 'env_params')
        file.write(alg_args2str + "\n")
        file.write(env_args2str + "\n")

    # --- Multiprocessing env creation ---
    env_fns = [make_env(env_config_dict, rank) for rank in range(args.n_rollout_threads)]
    vec_env = SubprocVecEnv(env_fns)
    obs_batch, state_batch = vec_env.reset()
    actions_avail = th.tensor(env_single.get_avail_actions(), device=train.device).float()
    actions_avail = actions_avail.repeat(args.n_rollout_threads, 1, 1)

    # Init hidden states
    # init_hidden returns shape (1, n_agents, hid_size)
    last_hid = train.behaviour_net.policy_dicts[0].init_hidden()

    # [Fix] Repeat for all parallel environments (n_rollout_threads)
    # Target shape: (n_rollout_threads, n_agents, hid_size)
    last_hid = last_hid.repeat(args.n_rollout_threads, 1, 1)

    # Main loop (parallel rollout + updates)
    # ==============================================================================
    # [最终完整版] 主训练循环
    # 包含了：
    # 1. 修复重复采样 & 步数计数
    # 2. 修复 LR 频率 & RNN 重置
    # 3. 【本次修复】补回了 r_soc_limit 等详细指标的显示！
    # ==============================================================================

    # 初始化累加器
    ep_rewards = np.zeros(args.n_rollout_threads)
    ep_steps = np.zeros(args.n_rollout_threads)
    # [新增] 用于累加详细指标 (如 soc_limit, voltage_reward 等)
    ep_metrics = {}

    print(f"🚀 Training Started! Target: {args.train_episodes_num} Episodes.")

    while train.episodes < args.train_episodes_num:
        stat = {}

        # 1. 获取动作
        obs_tensor = th.tensor(obs_batch, device=train.device).float()
        actions, action_pol, log_prob_a, _, hid = train.behaviour_net.get_actions(
            obs_tensor,
            status='train',
            exploration=True,
            actions_avail=actions_avail,
            target=False,
            last_hid=last_hid
        )
        value = train.behaviour_net.value(obs_tensor, action_pol)

        # 2. 环境执行 (单次调用)
        _, actual = translate_action(args, actions, env_single)
        next_obs_batch, next_state_batch, reward_batch, done_batch, info_batch = vec_env.step(actual)

        # 3. 数据累加 (核心修复点！)
        ep_rewards += reward_batch
        ep_steps += 1

        # [新增] 累加 info 里的详细指标
        if info_batch:
            # 如果是第一次，初始化字典 keys
            if not ep_metrics:
                for k, v in info_batch[0].items():
                    # 只记录数值类型的指标 (排除字符串等)
                    if isinstance(v, (int, float, np.number)):
                        ep_metrics[k] = np.zeros(args.n_rollout_threads)

            # 累加每一项
            for k in ep_metrics:
                # 提取 16 个环境的值，如果没有该 key 则补 0
                values = np.array([info.get(k, 0.0) for info in info_batch])
                ep_metrics[k] += values

        # 4. 下一步预测
        next_obs_tensor = th.tensor(next_obs_batch, device=train.device).float()
        next_actions, next_action_pol, _, _, next_hid = train.behaviour_net.get_actions(
            next_obs_tensor,
            status='train',
            exploration=True,
            actions_avail=actions_avail,
            target=False,
            last_hid=hid  # 这里传当前的 hid
        )
        next_value = train.behaviour_net.value(next_obs_tensor, next_action_pol)

        # 5. RNN 隐状态重置 (防止记忆泄漏)
        if np.any(done_batch):
            dones_indices = np.where(done_batch)[0]
            for idx in dones_indices:
                next_hid[idx] = 0.0

                # 6. 存入 Buffer
        if hasattr(train, "replay_buffer"):
            for env_idx in range(args.n_rollout_threads):
                obs_i = obs_batch[env_idx:env_idx + 1]
                next_obs_i = next_obs_batch[env_idx:env_idx + 1]
                action_i = action_pol[env_idx:env_idx + 1].detach().cpu().numpy()
                logp_i = log_prob_a[env_idx:env_idx + 1].detach().cpu().numpy()
                value_i = value[env_idx:env_idx + 1].detach().cpu().numpy()
                next_value_i = next_value[env_idx:env_idx + 1].detach().cpu().numpy()

                reward_i = np.array([[reward_batch[env_idx]] * env_single.get_num_of_agents()], dtype=np.float32)
                avail_i = env_single.get_avail_actions()[None, ...]

                last_hid_i = last_hid[env_idx:env_idx + 1].detach().cpu().numpy()
                next_hid_i = next_hid[env_idx:env_idx + 1].detach().cpu().numpy()

                trans = train.behaviour_net.Transition(
                    obs_i, action_i, logp_i, value_i, next_value_i, reward_i, next_obs_i,
                    done_batch[env_idx], done_batch[env_idx], avail_i, last_hid_i, next_hid_i
                )
                train.replay_buffer.add_experience(trans)
        # 9. 训练网络 (Training Updates)
        # ----------------------------------------------------------------------
        if hasattr(train, "replay_buffer") and len(train.replay_buffer.buffer) >= args.batch_size:
            for _ in range(args.updates_per_step):
                train.value_replay_process(stat)
                train.policy_replay_process(stat)
                if hasattr(train, 'mixer_optimizer') and train.args.mixer:
                    train.mixer_replay_process(stat)


        # 7. 处理 Done 结算与日志
        dones_idx = np.where(done_batch)[0]
        for env_idx in dones_idx:
            # 结算总奖励
            current_steps = max(1, ep_steps[env_idx])
            mean_reward = ep_rewards[env_idx] / current_steps
            stat['mean_train_reward'] = float(mean_reward)

            # [新增] 结算其他详细指标 (r_soc_limit, r_voltage 等)
            for k in ep_metrics:
                avg_val = ep_metrics[k][env_idx] / current_steps
                stat['mean_train_' + k] = float(avg_val)
                # 清零该指标的累加器
                ep_metrics[k][env_idx] = 0.0

            # 记录日志
            train.logging(stat)
            train.episodes += 1

            # 衰减学习率 (Correct Location)
            train.lr_step()

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] ✅ Episode {train.episodes} Finished. Mean Reward: {mean_reward:.2f}")

            # 清零基本累加器
            ep_rewards[env_idx] = 0.0
            ep_steps[env_idx] = 0.0

            if train.episodes >= args.train_episodes_num:
                break

        if train.episodes >= args.train_episodes_num:
            break

        # 8. 更新状态
        train.steps += args.n_rollout_threads
        obs_batch, state_batch = next_obs_batch, next_state_batch
        last_hid = next_hid.detach()




        # ----------------------------------------------------------------------
        # 10. 定期评估 (Evaluation)
        # ----------------------------------------------------------------------
        # [注意] 使用 >= 判断，防止多进程一次跳过多集导致错过评估点
        if train.episodes > 0 and train.episodes % args.eval_freq == 0:
            train.behaviour_net.evaluation(stat, train)

        # ----------------------------------------------------------------------
        # 11. 定期保存模型 (Save Model)
        # ----------------------------------------------------------------------
        if train.episodes > 0 and train.episodes % args.save_model_freq == 0:
            train.print_info(stat)
            save_dict = {
                "episode": train.episodes,
                "model_state_dict": train.behaviour_net.state_dict(),
                "policy_optimizer_state_dict": train.policy_optimizer.state_dict(),
                "value_optimizer_state_dict": train.value_optimizer.state_dict()
            }
            if hasattr(train, 'mixer_optimizer') and train.mixer_optimizer is not None:
                save_dict["mixer_optimizer_state_dict"] = train.mixer_optimizer.state_dict()

            th.save(save_dict, save_path + "model_save/" + log_name + "/model.pt")
            print(f"💾 Model saved at Episode {train.episodes}!\n")
        # Top-K save
        if "mean_test_reward" in stat:
            current_reward = stat["mean_test_reward"]
            should_save = (len(top_k_checkpoints) < argv.save_top_k) or (current_reward > top_k_checkpoints[-1][0] if top_k_checkpoints else True)
            if should_save:
                ckpt_name = f"model_ep{train.episodes}_rew{current_reward:.4f}.pt"
                full_path = os.path.join(save_path, "model_save", log_name, ckpt_name)
                save_dict = {
                    "episode": train.episodes,
                    "model_state_dict": train.behaviour_net.state_dict(),
                    "policy_optimizer_state_dict": train.policy_optimizer.state_dict(),
                    "value_optimizer_state_dict": train.value_optimizer.state_dict()
                }
                if hasattr(train, 'mixer_optimizer') and train.mixer_optimizer is not None:
                    save_dict["mixer_optimizer_state_dict"] = train.mixer_optimizer.state_dict()
                th.save(save_dict, full_path)
                top_k_checkpoints.append((current_reward, train.episodes, full_path))
                top_k_checkpoints = sorted(top_k_checkpoints, key=lambda x: x[0], reverse=True)
                if len(top_k_checkpoints) > argv.save_top_k:
                    worst_reward, worst_ep, worst_path = top_k_checkpoints.pop()
                    if os.path.exists(worst_path):
                        os.remove(worst_path)

        # periodic save based on episodes
        if (train.episodes > 0) and (train.episodes % args.save_model_freq == 0):
            train.print_info(stat)
            save_dict = {
                "episode": train.episodes,
                "model_state_dict": train.behaviour_net.state_dict(),
                "policy_optimizer_state_dict": train.policy_optimizer.state_dict(),
                "value_optimizer_state_dict": train.value_optimizer.state_dict()
            }
            if hasattr(train, 'mixer_optimizer') and train.mixer_optimizer is not None:
                save_dict["mixer_optimizer_state_dict"] = train.mixer_optimizer.state_dict()
            th.save(save_dict, save_path + "model_save/" + log_name + "/model.pt")
            print("Model (with optimizer state) saved!\n")

    vec_env.close()
    logger.close()

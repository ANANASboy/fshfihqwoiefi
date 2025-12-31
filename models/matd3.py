import torch as th
import torch.nn as nn
import numpy as np
from utilities.util import select_action
from models.model import Model
from critics.mlp_critic import MLPCritic

# 引入 Agent
from agents.mlp_agent import MLPAgent


class MATD3(Model):
    def __init__(self, args, target_net=None):
        super(MATD3, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.batchnorm = nn.BatchNorm1d(self.args.agent_num).to(self.device)

    def construct_value_net(self):
        # [修复] 真正的 TD3 需要两个完全独立的 Critic 网络列表
        if self.args.agent_id:
            input_shape = (self.obs_dim + self.act_dim) * self.n_ + 1 + self.n_
        else:
            input_shape = (self.obs_dim + self.act_dim) * self.n_ + 1

        output_shape = 1

        # 构建 Critic 1
        if self.args.shared_params:
            self.critic1_dicts = nn.ModuleList([MLPCritic(input_shape, output_shape, self.args)])
            self.critic2_dicts = nn.ModuleList([MLPCritic(input_shape, output_shape, self.args)])
        else:
            self.critic1_dicts = nn.ModuleList(
                [MLPCritic(input_shape, output_shape, self.args) for _ in range(self.n_)])
            self.critic2_dicts = nn.ModuleList(
                [MLPCritic(input_shape, output_shape, self.args) for _ in range(self.n_)])

    def construct_policy_net(self):
        # 保持你之前的 Actor 逻辑 (PV/ESS 分流)
        self.pv_input_dim = 3
        self.ess_input_dim = self.obs_dim

        if self.args.agent_type == 'mlp':
            if self.args.gaussian_policy:
                from agents.mlp_agent_gaussian import MLPAgent
            else:
                from agents.mlp_agent import MLPAgent
            Agent = MLPAgent
        elif self.args.agent_type == 'rnn':
            if self.args.gaussian_policy:
                from agents.rnn_agent_gaussian import RNNAgent
            else:
                from agents.rnn_agent import RNNAgent
            Agent = RNNAgent
        else:
            raise NotImplementedError()

        self.actor_pv = Agent(self.pv_input_dim, self.args)
        self.actor_ess = Agent(self.ess_input_dim, self.args)
        self.policy_dicts = nn.ModuleList([self.actor_pv, self.actor_ess])

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def value(self, obs, act):
        # [修复] 分别通过两个网络计算 Q1 和 Q2
        batch_size = obs.size(0)

        # --- 数据预处理 (保持原逻辑) ---
        obs_repeat = obs.unsqueeze(1).repeat(1, self.n_, 1, 1)
        obs_reshape = obs_repeat.contiguous().view(batch_size, self.n_, -1)

        agent_ids = th.eye(self.n_).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        if self.args.agent_id:
            obs_reshape = th.cat((obs_reshape, agent_ids), dim=-1)

        act_repeat = act.unsqueeze(1).repeat(1, self.n_, 1, 1)
        act_mask_others = agent_ids.unsqueeze(-1)
        act_mask_i = 1. - act_mask_others
        act_i = act_repeat * act_mask_others
        act_others = act_repeat * act_mask_i
        act_repeat = act_others.detach() + act_i

        if self.args.shared_params:
            obs_reshape = obs_reshape.contiguous().view(batch_size * self.n_, -1)
            act_reshape = act_repeat.contiguous().view(batch_size * self.n_, -1)
        else:
            obs_reshape = obs_reshape.contiguous().view(batch_size, self.n_, -1)
            act_reshape = act_repeat.contiguous().view(batch_size, self.n_, -1)

        inputs = th.cat((obs_reshape, act_reshape), dim=-1)

        # --- [核心修改] 这里的 inputs 不需要再拼 0 或 1 了 ---
        # 直接送入两个独立的网络

        if self.args.shared_params:
            # Critic 1
            q1, _ = self.critic1_dicts[0](inputs, None)
            q1 = q1.contiguous().view(batch_size, self.n_, 1)

            # Critic 2
            q2, _ = self.critic2_dicts[0](inputs, None)
            q2 = q2.contiguous().view(batch_size, self.n_, 1)
        else:
            # 兼容非共享参数 (虽然你可能不用)
            q1_list, q2_list = [], []
            for i in range(self.n_):
                q1_i, _ = self.critic1_dicts[i](inputs[:, i, :], None)
                q2_i, _ = self.critic2_dicts[i](inputs[:, i, :], None)
                q1_list.append(q1_i)
                q2_list.append(q2_i)
            q1 = th.stack(q1_list, dim=1)
            q2 = th.stack(q2_list, dim=1)

        return th.cat([q1, q2], dim=0)  # 返回拼接后的结果，为了兼容 get_loss 的切片逻辑

    def policy(self, obs, last_hid=None):
        # ... (保持你之前的 Policy 代码不变) ...
        # 直接复制之前我帮你改好的 policy 函数内容
        batch_size = obs.shape[0]
        obs_pv = obs[:, :6, :self.pv_input_dim]
        obs_ess = obs[:, 6:7, :]

        N = self.n_
        if last_hid is not None:
            if last_hid.dim() == 2:
                if last_hid.shape[0] % N == 0:
                    pass
                else:
                    # fallback specific fix logic if needed
                    pass

        hid_pv, hid_ess = None, None
        if last_hid is not None:
            # 安全检查 reshape
            if last_hid.shape[0] == batch_size:
                # (B, N, H)
                hid_pv = last_hid[:, :6, :]
                hid_ess = last_hid[:, 6:7, :]
            elif last_hid.shape[0] == batch_size * N:
                # (B*N, H) -> (B, N, H)
                last_hid_reshaped = last_hid.view(batch_size, N, -1)
                hid_pv = last_hid_reshaped[:, :6, :]
                hid_ess = last_hid_reshaped[:, 6:7, :]

        obs_pv_flat = obs_pv.reshape(-1, self.pv_input_dim)
        hid_pv_flat = hid_pv.reshape(-1, self.args.hid_size) if hid_pv is not None else None
        means_pv_flat, log_std_pv_flat, h_pv_flat = self.actor_pv(obs_pv_flat, hid_pv_flat)
        means_pv = means_pv_flat.view(batch_size, 6, -1)
        h_pv = h_pv_flat.view(batch_size, 6, -1) if h_pv_flat is not None else None
        if log_std_pv_flat is not None:
            log_std_pv = log_std_pv_flat.view(batch_size, 6, -1)
        else:
            log_std_pv = None

        obs_ess_flat = obs_ess.reshape(-1, self.ess_input_dim)
        hid_ess_flat = hid_ess.reshape(-1, self.args.hid_size) if hid_ess is not None else None
        means_ess_flat, log_std_ess_flat, h_ess_flat = self.actor_ess(obs_ess_flat, hid_ess_flat)
        means_ess = means_ess_flat.view(batch_size, 1, -1)
        h_ess = h_ess_flat.view(batch_size, 1, -1) if h_ess_flat is not None else None
        if log_std_ess_flat is not None:
            log_std_ess = log_std_ess_flat.view(batch_size, 1, -1)
        else:
            log_std_ess = None

        means = th.cat([means_pv, means_ess], dim=1)
        if self.args.gaussian_policy:
            log_stds = th.cat([log_std_pv, log_std_ess], dim=1)
        else:
            pv_std = getattr(self, "pv_noise", None).current_std if hasattr(self, "pv_noise") else 0.05
            ess_std = getattr(self, "ess_noise", None).current_std if hasattr(self, "ess_noise") else 0.05
            stds_pv = th.ones_like(means_pv).to(self.device) * pv_std
            stds_ess = th.ones_like(means_ess).to(self.device) * ess_std
            stds = th.cat([stds_pv, stds_ess], dim=1)
            log_stds = th.log(stds)

        if h_pv is not None and h_ess is not None:
            hiddens = th.cat([h_pv, h_ess], dim=1)
        else:
            hiddens = None

        return means, log_stds, hiddens

    def get_actions(self, obs, status, exploration, actions_avail, target=False, last_hid=None, clip=False):
        # 保持不变
        target_policy = self.target_net.policy if self.args.target else self.policy
        if self.args.continuous:
            means, log_stds, hiddens = self.policy(obs, last_hid=last_hid) if not target else target_policy(obs,
                                                                                                            last_hid=last_hid)
            means[actions_avail == 0] = 0.0
            if log_stds is not None: log_stds[actions_avail == 0] = 0.0
            actions, log_prob_a = select_action(self.args, means, status=status, exploration=exploration,
                                                info={'clip': clip, 'log_std': log_stds})
            restore_mask = 1. - (actions_avail == 0).to(self.device).float()
            restore_actions = restore_mask * actions
            action_out = (means, log_stds)
        else:
            logits, _, hiddens = self.policy(obs, last_hid=last_hid) if not target else target_policy(obs,
                                                                                                      last_hid=last_hid)
            logits[actions_avail == 0] = -9999999
            actions, log_prob_a = select_action(self.args, logits, status=status, exploration=exploration)
            restore_actions = actions
            action_out = logits
        return actions, restore_actions, log_prob_a, action_out, hiddens

    def get_loss(self, batch):
        batch_size = len(batch.state)
        state, actions, old_log_prob_a, old_values, old_next_values, rewards, next_state, done, last_step, actions_avail, last_hids, hids = self.unpack_data(
            batch)

        # 维度修复
        last_hids = last_hids.view(batch_size, self.n_, -1)
        hids = hids.view(batch_size, self.n_, -1)

        # 1. 计算当前的 Actor 动作 (用于 Policy Loss)
        _, actions_pol, log_prob_a, action_out, _ = self.get_actions(state, status='train', exploration=False,
                                                                     actions_avail=actions_avail, target=False,
                                                                     last_hid=last_hids)

        # 2. 计算 Target Q (用于 Value Loss)
        if self.args.double_q:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=True,
                                                        actions_avail=actions_avail, target=False, last_hid=hids,
                                                        clip=True)
        else:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=True,
                                                        actions_avail=actions_avail, target=True, last_hid=hids,
                                                        clip=True)

        # 3. Value 更新：两个网络都要算
        # 当前 Q (训练 Critic 用)
        q_current = self.value(state, actions)  # 返回 [Batch*2, N, 1] (因为上面 cat 了 q1, q2)
        q1_curr, q2_curr = q_current[:batch_size], q_current[batch_size:]

        # Target Q (计算目标用)
        q_target = self.target_net.value(next_state, next_actions.detach())
        q1_target, q2_target = q_target[:batch_size], q_target[batch_size:]

        # 构造 TD Target (Standard TD3: min(q1_t, q2_t))
        next_values = th.min(q1_target, q2_target)
        returns = rewards + self.args.gamma * (1 - done) * next_values.detach()

        # 4. Policy 更新：只用 Q1 (Standard TD3)
        # 重新算一次 Q1，输入是当前 Policy 的动作 actions_pol
        # 注意：这里我们得单独调一下 critic1，或者调 value() 然后切片
        q_pol_all = self.value(state, actions_pol)
        q1_pol = q_pol_all[:batch_size]

        if self.args.normalize_advantages:
            q1_pol = self.batchnorm(q1_pol)

        policy_loss = -q1_pol.mean()

        # 5. Critic Loss
        value_loss = 0.5 * ((returns - q1_curr).pow(2).mean() + (returns - q2_curr).pow(2).mean())

        return policy_loss, value_loss, action_out
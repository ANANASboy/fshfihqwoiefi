import torch as th
import torch.nn as nn
import numpy as np
from utilities.util import select_action
from models.model import Model
from critics.mlp_critic import MLPCritic

# 引入 Agent，注意根据配置选择类型
from agents.mlp_agent import MLPAgent
# 如果你用的是 Gaussian Policy，可能需要 from agents.mlp_agent_gaussian import MLPAgent
# 但通常 standard TD3 用的是 deterministic policy (mlp_agent.py) + noise

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
        # Critic 保持不变，它看全局信息，不需要切片
        if self.args.agent_id:
            input_shape = (self.obs_dim + self.act_dim) * self.n_ + 1 + self.n_
        else:
            input_shape = (self.obs_dim + self.act_dim) * self.n_ + 1
        output_shape = 1
        if self.args.shared_params:
            self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args) ] )
        else:
            self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args) for _ in range(self.n_) ] )

    def construct_policy_net(self):
        # --- [Phase 4 Modification: Grouped Parameter Sharing] ---
        # 不再使用统一的循环创建，而是手动创建两类 Actor
        
        # 1. 定义两类 Agent 的输入维度
        # PV: 仅使用局部有效特征 [V, theta, P, Q_last] (Phase 2 get_obs 逻辑)
        self.pv_input_dim = 3
        # ESS: 使用完整的 Padding 后特征 (Phase 2 get_obs 逻辑)
        self.ess_input_dim = self.obs_dim 

        # 2. 根据配置选择 Agent 类 (MLP 或 RNN)
        if self.args.agent_type == 'mlp':
            if self.args.gaussian_policy:
                from agents.mlp_agent_gaussian import MLPAgent
            else:
                from agents.mlp_agent import MLPAgent
            Agent = MLPAgent
        elif self.args.agent_type == 'rnn':
            # 暂时假设用 MLP，RNN 需要更复杂的 hidden state 处理
            if self.args.gaussian_policy:
                from agents.rnn_agent_gaussian import RNNAgent
            else:
                from agents.rnn_agent import RNNAgent
            Agent = RNNAgent
        else:
            raise NotImplementedError()
            
        # 3. 初始化两个独立的 Actor 网络
        # actor_pv: 供前 6 个 PV 智能体共享
        self.actor_pv = Agent(self.pv_input_dim, self.args)
        # actor_ess: 供第 7 个 ESS 智能体独享
        self.actor_ess = Agent(self.ess_input_dim, self.args)

        # 4. 注册到 ModuleList 以便 Optimizer 识别参数
        self.policy_dicts = nn.ModuleList([self.actor_pv, self.actor_ess])

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def value(self, obs, act):
        # Value 网络保持原样，因为它处理的是 Critic 逻辑
        # ... (原有代码保持不变) ...
        # obs_shape = (b, n, o)
        # act_shape = (b, n, a)
        batch_size = obs.size(0)

        obs_repeat = obs.unsqueeze(1).repeat(1, self.n_, 1, 1) # shape = (b, n, n, o)
        obs_reshape = obs_repeat.contiguous().view(batch_size, self.n_, -1) # shape = (b, n, n*o)

        # add agent id
        agent_ids = th.eye(self.n_).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device) # shape = (b, n, n)
        if self.args.agent_id:
            obs_reshape = th.cat( (obs_reshape, agent_ids), dim=-1 ) # shape = (b, n, n*o+n)

        act_repeat = act.unsqueeze(1).repeat(1, self.n_, 1, 1) # shape = (b, n, n, a)
        act_mask_others = agent_ids.unsqueeze(-1) # shape = (b, n, n, 1)
        act_mask_i = 1. - act_mask_others
        act_i = act_repeat * act_mask_others
        act_others = act_repeat * act_mask_i

        # detach other agents' actions
        act_repeat = act_others.detach() + act_i # shape = (b, n, n, a)
        
        if self.args.shared_params:
            obs_reshape = obs_reshape.contiguous().view( batch_size*self.n_, -1 ) # shape = (b*n, n*o+n/n*o)
            act_reshape = act_repeat.contiguous().view( batch_size*self.n_, -1 ) # shape = (b*n, n*a)
        else:
            obs_reshape = obs_reshape.contiguous().view( batch_size, self.n_, -1 ) # shape = (b, n, n*o+n/n*o)
            act_reshape = act_repeat.contiguous().view( batch_size, self.n_, -1 ) # shape = (b, n, n*a)

        inputs = th.cat( (obs_reshape, act_reshape), dim=-1 )
        ones = th.ones( inputs.size()[:-1] + (1,), dtype=th.float ).to(self.device)
        zeros = th.zeros( inputs.size()[:-1] + (1,), dtype=th.float ).to(self.device)
        inputs1 = th.cat( (inputs, zeros), dim=-1 )
        inputs2 = th.cat( (inputs, ones), dim=-1 )

        if self.args.shared_params:
            agent_value = self.value_dicts[0]
            values1, _ = agent_value(inputs1, None)
            values2, _ = agent_value(inputs2, None)
            values1 = values1.contiguous().view(batch_size, self.n_, 1)
            values2 = values2.contiguous().view(batch_size, self.n_, 1)
        else:
            values1, values2 = [], []
            for i, agent_value in enumerate(self.value_dicts):
                values_1, _ = agent_value(inputs1[:, i, :], None)
                values_2, _ = agent_value(inputs2[:, i, :], None)
                values1.append(values_1)
                values2.append(values_2)
            values1 = th.stack(values1, dim=1)
            values2 = th.stack(values2, dim=1)

        return th.cat([values1, values2], dim=0)

    def policy(self, obs, last_hid=None):
        """
        前向传播：根据异构 Agent 分流处理，并处理 RNN 的维度折叠
        obs shape: (batch, 7, 71)
        """
        batch_size = obs.shape[0]

        # --- 1. 拆分输入数据 ---
        # PV: 前 6 个 Agent，切片取前 4 个特征
        obs_pv = obs[:, :6, :self.pv_input_dim] # (batch, 6, 4)
        
        # ESS: 第 7 个 Agent (索引 6)，取所有特征
        obs_ess = obs[:, 6:7, :] # (batch, 1, 71)

        # === 【插入修复代码 START】 ===
        # # 修复 ReplayBuffer 采样导致的维度展平问题
        # if last_hid is not None:
        #     # last_hid.shape[1] = Total_Features (7 * Hidden_Size)
        #     # last_hid.shape[0] = Batch_Size
        #
        #     # 【新修复代码 START】
        #     # 这里的 7 是 agent_num
        #     # last_hid.shape[1] // 7 应该等于 Hidden_Size (例如 256)
        #     hid_size = last_hid.shape[1] // 7
        #
        #     # 将 [Batch, Agent * Hidden] 变为 [Batch, Agent, Hidden]
        #     last_hid = last_hid.view(last_hid.shape[0], 7, hid_size)
        #     # 【新修复代码 END】
        N = self.n_  # = args.agent_num
        if last_hid is not None:
            if last_hid.dim() == 2:
                # last_hid: (B*N, hid)  -> (B, N, hid)
                assert last_hid.shape[0] % N == 0, f"last_hid.shape[0]={last_hid.shape[0]} not divisible by N={N}"
                B = last_hid.shape[0] // N
                # shared_params=True 时：last_hid 应保持 (batch*n_agents, hid_dim)
                # 如果你某处把它弄成了 3D，就压回 2D
                if last_hid is not None and last_hid.dim() == 3:
                    last_hid = last_hid.contiguous().view(-1, last_hid.size(-1))
            elif last_hid.dim() == 3:
                # already (B, N, hid)
                pass
            else:
                raise RuntimeError(f"Unexpected last_hid.dim()={last_hid.dim()}, shape={tuple(last_hid.shape)}")

        # 处理 Hidden State
        hid_pv, hid_ess = None, None
        if last_hid is not None:
            hid_pv = last_hid[:, :6, :] # (batch, 6, hid)
            hid_ess = last_hid[:, 6:7, :] # (batch, 1, hid)

        # --- 2. PV Actor (Shared) 前向传播 ---
        # 关键修复：GRUCell 不支持 3D 输入，必须将 (Batch, Agents) 维度合并
        # Reshape input: (batch, 6, 4) -> (batch*6, 4)
        obs_pv_flat = obs_pv.reshape(-1, self.pv_input_dim)
        
        # Reshape hidden: (batch, 6, hid) -> (batch*6, hid)
        hid_pv_flat = None
        if hid_pv is not None:
            hid_pv_flat = hid_pv.reshape(-1, self.args.hid_size)

        # Forward Pass
        means_pv_flat, log_std_pv_flat, h_pv_flat = self.actor_pv(obs_pv_flat, hid_pv_flat)
        
        # Restore Shape: (batch*6, dim) -> (batch, 6, dim)
        means_pv = means_pv_flat.view(batch_size, 6, -1)
        if log_std_pv_flat is not None:
            log_std_pv = log_std_pv_flat.view(batch_size, 6, -1)
        else:
            log_std_pv = None
            
        if h_pv_flat is not None:
            h_pv = h_pv_flat.view(batch_size, 6, -1)
        else:
            h_pv = None

        # --- 3. ESS Actor (Independent) 前向传播 ---
        # 同样做 Flatten 处理以防万一 (虽然维度是 1)
        obs_ess_flat = obs_ess.reshape(-1, self.ess_input_dim)
        
        hid_ess_flat = None
        if hid_ess is not None:
            hid_ess_flat = hid_ess.reshape(-1, self.args.hid_size)
            
        means_ess_flat, log_std_ess_flat, h_ess_flat = self.actor_ess(obs_ess_flat, hid_ess_flat)

        # Restore Shape: (batch*1, dim) -> (batch, 1, dim)
        means_ess = means_ess_flat.view(batch_size, 1, -1)
        if log_std_ess_flat is not None:
            log_std_ess = log_std_ess_flat.view(batch_size, 1, -1)
        else:
            log_std_ess = None
            
        if h_ess_flat is not None:
            h_ess = h_ess_flat.view(batch_size, 1, -1)
        else:
            h_ess = None

        # --- 4. 拼接输出 (Merge) ---
        means = th.cat([means_pv, means_ess], dim=1) # (batch, 7, act_dim)
        
        # # 处理 Log Std (针对 Gaussian Policy)
        # if self.args.gaussian_policy:
        #     log_stds = th.cat([log_std_pv, log_std_ess], dim=1)
        # else:
        #     # 兼容非高斯策略: 构造固定 std
        #     stds = th.ones_like(means).to(self.device) * self.args.fixed_policy_std
        #     log_stds = th.log(stds)

        if self.args.gaussian_policy:
            log_stds = th.cat([log_std_pv, log_std_ess], dim=1)
        else:
            # ==============================================================
            # [V6 修改] 异构噪声策略 (Heterogeneous Noise Strategy)
            # 改为使用可衰减的噪声生成器（PV / ESS 独立）
            # ==============================================================
            pv_std = getattr(self, "pv_noise", None).current_std if hasattr(self, "pv_noise") else 0.0
            ess_std = getattr(self, "ess_noise", None).current_std if hasattr(self, "ess_noise") else 0.0
            stds_pv = th.ones_like(means_pv).to(self.device) * pv_std
            stds_ess = th.ones_like(means_ess).to(self.device) * ess_std

            stds = th.cat([stds_pv, stds_ess], dim=1)
            log_stds = th.log(stds)

            

        # 处理 Hidden State (拼接回 batch, 7, hid)
        if h_pv is not None and h_ess is not None:
            hiddens = th.cat([h_pv, h_ess], dim=1)
        else:
            hiddens = None

        return means, log_stds, hiddens

    def get_actions(self, obs, status, exploration, actions_avail, target=False, last_hid=None, clip=False):
        target_policy = self.target_net.policy if self.args.target else self.policy
        if self.args.continuous:
            # 调用重写后的 policy 方法
            means, log_stds, hiddens = self.policy(obs, last_hid=last_hid) if not target else target_policy(obs, last_hid=last_hid)
            
            # Masking
            means[actions_avail == 0] = 0.0
            log_stds[actions_avail == 0] = 0.0
            
            # --- [修复后的逻辑：直接使用，不求和] ---
            # 直接传入 select_action，它会处理 batch 和 agent 维度
            actions, log_prob_a = select_action(self.args, means, status=status, exploration=exploration, info={'clip': clip, 'log_std': log_stds})
            
            restore_mask = 1. - (actions_avail == 0).to(self.device).float()
            restore_actions = restore_mask * actions
            action_out = (means, log_stds)
        else:
            # Discrete action logic (if needed)
            logits, _, hiddens = self.policy(obs, last_hid=last_hid) if not target else target_policy(obs, last_hid=last_hid)
            logits[actions_avail == 0] = -9999999
            actions, log_prob_a = select_action(self.args, logits, status=status, exploration=exploration)
            restore_actions = actions
            action_out = logits
            
        return actions, restore_actions, log_prob_a, action_out, hiddens

    def get_loss(self, batch):
        # Loss 计算逻辑保持原样，它依赖 get_actions 和 value，这两个我们都兼容好了
        batch_size = len(batch.state)
        state, actions, old_log_prob_a, old_values, old_next_values, rewards, next_state, done, last_step, actions_avail, last_hids, hids = self.unpack_data(batch)
        # ================= [修复代码开始] =================
        # 错误原因：unpack_data 将数据展平为 (Batch*Agents, Hidden)，导致 policy 函数切片时维度不足
        # 修复方法：强制将 Hidden State 恢复为 (Batch, Agents, Hidden)
        last_hids = last_hids.view(batch_size, self.n_, -1)
        hids = hids.view(batch_size, self.n_, -1)
        # ================= [修复代码结束] =================

        _, actions_pol, log_prob_a, action_out, _ = self.get_actions(state, status='train', exploration=False, 
            actions_avail=actions_avail, target=False, last_hid=last_hids)
        
        if self.args.double_q:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=True, 
                actions_avail=actions_avail, target=False, last_hid=hids, clip=True)
        else:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=True, 
                actions_avail=actions_avail, target=True, last_hid=hids, clip=True)
        
        compose_pol = self.value(state, actions_pol)
        values_pol = compose_pol[:batch_size, :]
        values_pol = values_pol.contiguous().view(-1, self.n_)
        
        compose = self.value(state, actions)
        values1, values2 = compose[:batch_size, :], compose[batch_size:, :]
        values1 = values1.contiguous().view(-1, self.n_)
        values2 = values2.contiguous().view(-1, self.n_)
        
        next_compose = self.target_net.value(next_state, next_actions.detach())
        next_values1, next_values2 = next_compose[:batch_size, :], next_compose[batch_size:, :]
        next_values1 = next_values1.contiguous().view(-1, self.n_)
        next_values2 = next_values2.contiguous().view(-1, self.n_)
        
        returns = th.zeros((batch_size, self.n_), dtype=th.float, device=self.device)
        assert values_pol.size() == next_values1.size() == next_values2.size()
        assert returns.size() == values1.size() == values2.size()
        
        # update twin values by the minimized target q
        done = done.to(self.device)
        next_values = th.stack([next_values1, next_values2], -1)
        returns = rewards + self.args.gamma * (1 - done) * th.min(next_values.detach(), -1)[0]
        
        deltas1 = returns - values1
        deltas2 = returns - values2
        advantages = values_pol
        
        if self.args.normalize_advantages:
            advantages = self.batchnorm(advantages)
            
        policy_loss = - advantages
        policy_loss = policy_loss.mean()
        value_loss = 0.5 * ( deltas1.pow(2).mean() + deltas2.pow(2).mean() )
        return policy_loss, value_loss, action_out

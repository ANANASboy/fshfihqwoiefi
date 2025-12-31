from ..multiagentenv import MultiAgentEnv
import numpy as np
import  pandapower as pp
from pandapower import ppException
import pandas as pd
import copy
import os
from collections import namedtuple
from .pf_res_plot import pf_res_plotly
from .voltage_barrier.voltage_barrier_backend import VoltageBarrier



def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


class ActionSpace(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high


class VoltageControl(MultiAgentEnv):
    """this class is for the environment of distributed active voltage control

        it is easy to interact with the environment, e.g.,

        state, global_state = env.reset()
        for t in range(240):
            actions = agents.get_actions(state) # a vector involving all agents' actions
            reward, done, info = env.step(actions)
            next_state = env.get_obs()
            state = next_state
    """
    def __init__(self, kwargs):
        """initialisation
        """
        # unpack args
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args

        # set the data path
        self.data_path = args.data_path

        # set the random seed
        np.random.seed(args.seed)
        
        # load the model of power network
        self.base_powergrid = self._load_network()
        
        # load data
        self.pv_data = self._load_pv_data()
        self.active_demand_data = self._load_active_demand_data()
        self.reactive_demand_data = self._load_reactive_demand_data()

        # define episode and rewards
        self.episode_limit = args.episode_limit
        self.voltage_barrier_type = getattr(args, "voltage_barrier_type", "l1")
        self.voltage_weight = getattr(args, "voltage_weight", 1.0)
        self.q_weight = getattr(args, "q_weight", 0.1)
        self.line_weight = getattr(args, "line_weight", None)
        self.dv_dq_weight = getattr(args, "dq_dv_weight", None)

        # define constraints and uncertainty
        self.v_upper = getattr(args, "v_upper", 1.05)
        self.v_lower = getattr(args, "v_lower", 0.95)
        self.active_demand_std = self.active_demand_data.values.std(axis=0) / 100.0
        self.reactive_demand_std = self.reactive_demand_data.values.std(axis=0) / 100.0
        self.pv_std = self.pv_data.values.std(axis=0) / 100.0
        self._set_reactive_power_boundary()

        # define action space and observation space for heterogeneous agents
        self.n_agents = 7  # 6 PVs + 1 ESS
        self.n_actions = 2 # Action is [P, Q]. For PVs, P is ignored.
        self.action_space = ActionSpace(low=np.array([-1.0]*self.n_actions), high=np.array([1.0]*self.n_actions))
        
        self.history = getattr(args, "history", 1)
        self.state_space = getattr(args, "state_space", ["pv", "demand", "reactive", "vm_pu", "va_degree"])
        
        # --- [修改开始] ---
        # 1. 先初始化 last_actions，因为 reset() -> get_obs() 会用到它
        self.last_actions = np.zeros((self.n_agents, self.n_actions))
        
        # 2. 然后再调用 reset()
        agents_obs, state = self.reset()
        # --- [修改结束] ---

        self.obs_size = agents_obs[0].shape[0]
        self.state_size = state.shape[0]
        
        # Storing last step's values
        self.last_v = self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)

        # initialise voltage barrier function
        self.voltage_barrier = VoltageBarrier(self.voltage_barrier_type)
        self._rendering_initialized = False

    def reset(self, reset_time=True):
        """reset the env
        """
        # reset the time step, cumulative rewards and obs history
        self.steps = 1
        self.sum_rewards = 0
        self.soc = 0.5
        self.ess_capacity = 2.0
        if self.history > 1:
            self.obs_history = {i: [] for i in range(self.n_agents)}

        # reset the power grid
        self.powergrid = copy.deepcopy(self.base_powergrid)
        solvable = False
        while not solvable:
            # reset the time stamp
            if reset_time:
                self._episode_start_hour = self._select_start_hour()
                self._episode_start_day = self._select_start_day()
                self._episode_start_interval = self._select_start_interval()
            # get one episode of data
            self.pv_histories = self._get_episode_pv_history()
            self.active_demand_histories = self._get_episode_active_demand_history()
            self.reactive_demand_histories = self._get_episode_reactive_demand_history()
            self._set_demand_and_pv()
            
            # --- 修改开始：适配新的动作维度 ---
            if self.args.reset_action:
                # 1. 获取所有智能体的随机动作 (7, 2)
                rand_actions = self.get_action()
                
                # 2. 只提取 PV 的 Q 动作 (前 6 个智能体，第 2 列)
                # shape 变为 (6,)，正好对应 sgen 的数量
                pv_q_actions = rand_actions[:self.n_agents-1, 1]
                
                # 3. 赋值并执行物理截断
                self.powergrid.sgen["q_mvar"] = pv_q_actions
                self.powergrid.sgen["q_mvar"] = self._clip_reactive_power(self.powergrid.sgen["q_mvar"], self.powergrid.sgen["p_mw"])
            # --- 修改结束 ---

            try:    
                pp.runpp(self.powergrid)
                solvable = True
            except ppException:
                print ("The power flow for the initialisation of demand and PV cannot be solved.")
                print (f"This is the pv: \n{self.powergrid.sgen['p_mw']}")
                print (f"This is the q: \n{self.powergrid.sgen['q_mvar']}")
                print (f"This is the active demand: \n{self.powergrid.load['p_mw']}")
                print (f"This is the reactive demand: \n{self.powergrid.load['q_mvar']}")
                print (f"This is the res_bus: \n{self.powergrid.res_bus}")
                solvable = False

        return self.get_obs(), self.get_state()
    
    def manual_reset(self, day, hour, interval):
        """manual reset the initial date
        """
        # reset the time step, cumulative rewards and obs history
        self.steps = 1
        self.sum_rewards = 0
        self.soc = 0.5
        self.ess_capacity = 2.0
        if self.history > 1:
            self.obs_history = {i: [] for i in range(self.n_agents)}

        # reset the power grid
        self.powergrid = copy.deepcopy(self.base_powergrid)

        # reset the time stamp
        self._episode_start_hour = hour
        self._episode_start_day = day
        self._episode_start_interval = interval
        solvable = False
        while not solvable:
            # get one episode of data
            self.pv_histories = self._get_episode_pv_history()
            self.active_demand_histories = self._get_episode_active_demand_history()
            self.reactive_demand_histories = self._get_episode_reactive_demand_history()
            self._set_demand_and_pv(add_noise=False)
            # random initialise action
            # if self.args.reset_action:
            #     self.powergrid.sgen["q_mvar"] = self.get_action()
            #     self.powergrid.sgen["q_mvar"] = self._clip_reactive_power(self.powergrid.sgen["q_mvar"], self.powergrid.sgen["p_mw"])
            # random initialise action
            if self.args.reset_action:
                # 1. 获取所有智能体的随机动作 (7, 2)
                rand_actions = self.get_action()

                # 2. 只提取 PV 的 Q 动作 (前 6 个智能体，第 2 列)
                # 这样 shape 变为 (6,)，正好对应 sgen 的数量
                pv_q_actions = rand_actions[:self.n_agents - 1, 1]

                # 3. 赋值并执行物理截断
                self.powergrid.sgen["q_mvar"] = pv_q_actions
                self.powergrid.sgen["q_mvar"] = self._clip_reactive_power(self.powergrid.sgen["q_mvar"],
                                                                          self.powergrid.sgen["p_mw"])
            try:    
                pp.runpp(self.powergrid)
                solvable = True
            except ppException:
                print ("The power flow for the initialisation of demand and PV cannot be solved.")
                print (f"This is the pv: \n{self.powergrid.sgen['p_mw']}")
                print (f"This is the q: \n{self.powergrid.sgen['q_mvar']}")
                print (f"This is the active demand: \n{self.powergrid.load['p_mw']}")
                print (f"This is the reactive demand: \n{self.powergrid.load['q_mvar']}")
                print (f"This is the res_bus: \n{self.powergrid.res_bus}")
                solvable = False

        return self.get_obs(), self.get_state()

    def step(self, actions, add_noise=True):
        """function for the interaction between agent and the env each time step
        """
        last_powergrid = copy.deepcopy(self.powergrid)

        # check whether the power balance is unsolvable
        solvable = self._take_action(actions)
        if solvable:
            # get the reward of current actions
            reward, info = self._calc_reward()
        else:
            q_loss = np.mean( np.abs(self.powergrid.sgen["q_mvar"]) )
            self.powergrid = last_powergrid
            reward, info = self._calc_reward()
            reward -= 200.
            # keep q_loss
            info["destroy"] = 1.
            info["totally_controllable_ratio"] = 0.
            info["q_loss"] = q_loss

        # set the pv and demand for the next time step
        self._set_demand_and_pv(add_noise=add_noise)

        # terminate if episode_limit is reached
        self.steps += 1
        self.sum_rewards += reward
        if self.steps >= self.episode_limit or not solvable:
            terminated = True
        else:
            terminated = False
        if terminated:
            print (f"Episode terminated at time: {self.steps} with return: {self.sum_rewards:2.4f}.")

            # ==========================================
            # 【在这里添加】全局缩放，防止梯度爆炸
            # ==========================================
        reward = reward / 100.0

        return reward, terminated, info

    def get_state(self):
        """return the global state for the power system
           the default state: voltage, active power of generators, bus state, load active power, load reactive power
        """
        state = []
        if "demand" in self.state_space:
            state += list(self.powergrid.res_bus["p_mw"].sort_index().to_numpy(copy=True))
            state += list(self.powergrid.res_bus["q_mvar"].sort_index().to_numpy(copy=True))
        if "pv" in self.state_space:
            state += list(self.powergrid.sgen["p_mw"].sort_index().to_numpy(copy=True))
        if "reactive" in self.state_space:
            state += list(self.powergrid.sgen["q_mvar"].sort_index().to_numpy(copy=True))
        if "vm_pu" in self.state_space:
            state += list(self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True))
        if "va_degree" in self.state_space:
            state += list(self.powergrid.res_bus["va_degree"].sort_index().to_numpy(copy=True))
        state = np.array(state)
        return state
    
    def get_obs(self):
        """
        Returns observations for heterogeneous agents (PVs and ESS).
        PVs get local observations.
        ESS gets a local observation.
        All observations are padded to the same length.
        """
        obs_list = []
        
        # Agent 0-5: PV Generators (local observation)
        for agent_id in range(self.n_agents - 1):
            sgen_bus = self.powergrid.sgen["bus"][agent_id]
            
            # Local voltage
            vm = self.powergrid.res_bus["vm_pu"][sgen_bus]
            
            # Local PV active power
            p_pv = self.powergrid.sgen["p_mw"][agent_id]
            
            # Last action (only Q is relevant for PV)
            last_q = self.last_actions[agent_id, 1]
            
            # Observation: [v_{t,i}, P_{t,i}^{gen}, q_{t-1,i}]
            obs_pv = np.array([vm, p_pv, last_q])
            obs_list.append(obs_pv)

        # Agent 6: ESS (local observation)
        ess_agent_id = self.n_agents - 1

        # ESS bus index
        ess_bus = 32
        
        # 1. Local voltage at ESS bus
        v_local = self.powergrid.res_bus["vm_pu"][ess_bus]

        # 2. SoC
        soc = self.soc

        # 3 & 4. Last action for ESS: [last_p, last_q]
        last_action = self.last_actions[ess_agent_id]

        # 5 & 6. Time features: [sin_t, cos_t]
        time_features = self._get_time_features()

        # Observation: [v_local, soc, last_p, last_q, sin_t, cos_t]
        obs_ess = np.concatenate(([v_local, soc], last_action, time_features))
        obs_list.append(obs_ess)
        
        # Padding
        max_len = max(len(o) for o in obs_list)
        agents_obs = [np.pad(o, (0, max_len - len(o)), 'constant') for o in obs_list]
        
        if self.history > 1:
            # This history logic is from the original implementation
            agents_obs_ = []
            for i, obs in enumerate(agents_obs):
                if len(self.obs_history[i]) >= self.history - 1:
                    obs_ = np.concatenate(self.obs_history[i][-(self.history-1):] + [obs])
                else:
                    zeros = [np.zeros_like(obs)] * (self.history - len(self.obs_history[i]) - 1)
                    obs_ = self.obs_history[i] + [obs]
                    obs_ = zeros + obs_
                    obs_ = np.concatenate(obs_)
                agents_obs_.append(copy.deepcopy(obs_))
                self.obs_history[i].append(copy.deepcopy(obs))
            agents_obs = agents_obs_

        return agents_obs

    def get_obs_agent(self, agent_id):
        """return observation for agent_id 
        """
        agents_obs = self.get_obs()
        return agents_obs[agent_id]
    
    def get_obs_size(self):
        """return the observation size
        """
        return self.obs_size

    def get_state_size(self):
        """return the state size
        """
        return self.state_size

    def get_action(self):
        """return the action according to a uniform distribution over [action_lower, action_upper)
        """
        # 修改：不再使用 sgen 的 shape，而是使用 (n_agents, n_actions) 即 (7, 2)
        rand_action = np.random.uniform(
            low=self.action_space.low, 
            high=self.action_space.high, 
            size=(self.n_agents, self.n_actions)
        )
        return rand_action

    def get_total_actions(self):
        """return the total number of actions an agent could ever take 
        """
        return self.n_actions

    def get_avail_actions(self):
        """return available actions for all agents
        """
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return np.expand_dims(np.array(avail_actions), axis=0)

    def get_avail_agent_actions(self, agent_id):
        """ return the available actions for agent_id 
        """
        if self.args.mode == "distributed":
            # 修改：返回长度为 n_actions (即 2) 的列表 [1, 1]
            return [1] * self.n_actions
        elif self.args.mode == "decentralised":
            avail_actions = np.zeros(self.n_actions)
            zone_sgens = self.base_powergrid.sgen.loc[self.base_powergrid.sgen["name"] == f"zone{agent_id+1}"]
            avail_actions[zone_sgens.index] = 1
            return avail_actions

    def get_num_of_agents(self):
        """return the number of agents
        """
        return self.n_agents

    def _get_time_features(self):
        """Returns sinusoidal time features for the current step."""
        T = 24 * (60 // self.time_delta)  # Total intervals in a day
        current_interval_of_day = (self._episode_start_interval + self.steps) % T
        sin_time = np.sin(2 * np.pi * current_interval_of_day / T)
        cos_time = np.cos(2 * np.pi * current_interval_of_day / T)
        return np.array([sin_time, cos_time])

    def _get_voltage(self):
        return self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)

    def _create_basenet(self, base_net):
        """initilization of power grid
        set the pandapower net to use
        """
        if base_net is None:
            raise Exception("Please provide a base_net configured as pandapower format.")
        else:
            return base_net

    def _select_start_hour(self):
        """select start hour for an episode
        """
        return np.random.choice(24)
    
    def _select_start_interval(self):
        """select start interval for an episode
        """
        return np.random.choice( 60 // self.time_delta )

    def _select_start_day(self):
        """select start day (date) for an episode
        """
        pv_data = self.pv_data
        pv_days = (pv_data.index[-1] - pv_data.index[0]).days
        self.time_delta = (pv_data.index[1] - pv_data.index[0]).seconds // 60
        episode_days = ( self.episode_limit // (24 * (60 // self.time_delta) ) ) + 1  # margin
        return np.random.choice(pv_days - episode_days)

    # def _load_network(self):
    #     """load network
    #     """
    #     network_path = os.path.join(self.data_path, 'model.p')
    #     base_net = pp.from_pickle(network_path)
    #     pp.create_storage(base_net, bus=32, p_mw=0.0, max_e_mwh=2.0, sn_mva=1.0, name="ESS", index=0)
    #     return self._create_basenet(base_net)
    def _load_network(self):
        """load network
        """
        network_path = os.path.join(self.data_path, 'model.p')
        base_net = pp.from_pickle(network_path)

        # 1. 创建储能
        pp.create_storage(base_net, bus=32, p_mw=0.0, max_e_mwh=2.0, sn_mva=1.0, name="ESS", index=0)

        # 2. [新增] 强制补全 sn_mva 列 (修复 NaN Bug) -- 请加上这一段！
        if "sn_mva" not in base_net.storage.columns:
            base_net.storage["sn_mva"] = 1.0
        else:
            base_net.storage["sn_mva"] = base_net.storage["sn_mva"].fillna(1.0)

        return self._create_basenet(base_net)

    def _load_pv_data(self):
        """load pv data
        the sensor frequency is set to 3 or 15 mins as default
        """
        pv_path = os.path.join(self.data_path, 'pv_active.csv')
        pv = pd.read_csv(pv_path, index_col=None)
        pv.index = pd.to_datetime(pv.iloc[:, 0])
        pv.index.name = 'time'
        pv = pv.iloc[::1, 1:] * self.args.pv_scale
        return pv

    def _load_active_demand_data(self):
        """load active demand data
        the sensor frequency is set to 3 or 15 mins as default
        """
        demand_path = os.path.join(self.data_path, 'load_active.csv')
        demand = pd.read_csv(demand_path, index_col=None)
        demand.index = pd.to_datetime(demand.iloc[:, 0])
        demand.index.name = 'time'
        demand = demand.iloc[::1, 1:] * self.args.demand_scale
        return demand
    
    def _load_reactive_demand_data(self):
        """load reactive demand data
        the sensor frequency is set to 3 min as default
        """
        demand_path = os.path.join(self.data_path, 'load_reactive.csv')
        demand = pd.read_csv(demand_path, index_col=None)
        demand.index = pd.to_datetime(demand.iloc[:, 0])
        demand.index.name = 'time'
        demand = demand.iloc[::1, 1:] * self.args.demand_scale
        return demand

    def _get_episode_pv_history(self):
        """return the pv history in an episode
        """
        episode_length = self.episode_limit
        history = self.history
        start = self._episode_start_interval + self._episode_start_hour * (60 // self.time_delta) + self._episode_start_day * 24 * (60 // self.time_delta)
        nr_intervals = episode_length + history + 1  # margin of 1
        episode_pv_history = self.pv_data[start:start + nr_intervals].values
        return episode_pv_history
    
    def _get_episode_active_demand_history(self):
        """return the active power histories for all loads in an episode
        """
        episode_length = self.episode_limit
        history = self.history
        start = self._episode_start_interval + self._episode_start_hour * (60 // self.time_delta) + self._episode_start_day * 24 * (60 // self.time_delta)
        nr_intervals = episode_length + history + 1  # margin of 1
        episode_demand_history = self.active_demand_data[start:start + nr_intervals].values
        return episode_demand_history
    
    def _get_episode_reactive_demand_history(self):
        """return the reactive power histories for all loads in an episode
        """
        episode_length = self.episode_limit
        history = self.history
        start = self._episode_start_interval + self._episode_start_hour * (60 // self.time_delta) + self._episode_start_day * 24 * (60 // self.time_delta)
        nr_intervals = episode_length + history + 1  # margin of 1
        episode_demand_history = self.reactive_demand_data[start:start + nr_intervals].values
        return episode_demand_history

    def _get_pv_history(self):
        """returns pv history
        """
        t = self.steps
        history = self.history
        return self.pv_histories[t:t+history, :]

    def _get_active_demand_history(self):
        """return the history demand
        """
        t = self.steps
        history = self.history
        return self.active_demand_histories[t:t+history, :]
    
    def _get_reactive_demand_history(self):
        """return the history demand
        """
        t = self.steps
        history = self.history
        return self.reactive_demand_histories[t:t+history, :]

    def _set_demand_and_pv(self, add_noise=True):
        """optionally update the demand and pv production according to the histories with some i.i.d. gaussian noise
        """ 
        pv = copy.copy(self._get_pv_history()[0, :])

        # add uncertainty to pv data with unit truncated gaussian (only positive accepted)
        if add_noise:
            pv += self.pv_std * np.abs(np.random.randn(*pv.shape))
        active_demand = copy.copy(self._get_active_demand_history()[0, :])

        # add uncertainty to active power of demand data with unit truncated gaussian (only positive accepted)
        if add_noise:
            active_demand += self.active_demand_std * np.abs(np.random.randn(*active_demand.shape))
        reactive_demand = copy.copy(self._get_reactive_demand_history()[0, :])

        # add uncertainty to reactive power of demand data with unit truncated gaussian (only positive accepted)
        if add_noise:
            reactive_demand += self.reactive_demand_std * np.abs(np.random.randn(*reactive_demand.shape))

        # update the record in the pandapower
        self.powergrid.sgen["p_mw"] = pv
        self.powergrid.load["p_mw"] = active_demand
        self.powergrid.load["q_mvar"] = reactive_demand

    def _set_reactive_power_boundary(self):
        """set the boundary of reactive power
        """
        self.factor = 1.2
        self.p_max = self.pv_data.to_numpy(copy=True).max(axis=0)
        self.s_max = self.factor * self.p_max
        print (f"This is the s_max: \n{self.s_max}")

    def _get_clusters_info(self):
        """return the clusters of info
        the clusters info is divided by predefined zone
        distributed: each zone is equipped with several PV generators and each PV generator is an agent
        decentralised: each zone is controlled by an agent and each agent may have variant number of actions
        """
        clusters = dict()
        if self.args.mode == "distributed":
            for i in range(len(self.powergrid.sgen)):
                zone = self.powergrid.sgen["name"][i]
                sgen_bus = self.powergrid.sgen["bus"][i]
                pv = self.powergrid.sgen["p_mw"][i]
                q = self.powergrid.sgen["q_mvar"][i]
                zone_res_buses = self.powergrid.res_bus.sort_index().loc[self.powergrid.bus["zone"]==zone]
                clusters[f"sgen{i}"] = (zone_res_buses, zone, pv, q, sgen_bus)
        elif self.args.mode == "decentralised":
            for i in range(self.n_agents):
                zone_res_buses = self.powergrid.res_bus.sort_index().loc[self.powergrid.bus["zone"]==f"zone{i+1}"]
                sgen_res_buses = self.powergrid.sgen["bus"].loc[self.powergrid.sgen["name"] == f"zone{i+1}"]
                pv = self.powergrid.sgen["p_mw"].loc[self.powergrid.sgen["name"] == f"zone{i+1}"]
                q = self.powergrid.sgen["q_mvar"].loc[self.powergrid.sgen["name"] == f"zone{i+1}"]
                clusters[f"zone{i+1}"] = (zone_res_buses, pv, q, sgen_res_buses)

        return clusters
    
    def _take_action(self, actions):
        """take the control variables
        the control variables we consider are the exact reactive power
        of each distributed generator and the active/reactive power of ESS.
        """
        # Convert to ndarray and enforce normalized action bounds [-1, 1]
        actions = np.asarray(actions, dtype=float)
        actions = np.clip(actions, -1.0, 1.0)

        # Store (possibly clipped) normalized actions for use in next observation
        self.last_actions = actions.copy()

        # Unpack actions: PVs (first 6 agents) and ESS (last agent)
        pv_q_actions = actions[:(self.n_agents - 1), 1]
        ess_agent_id = self.n_agents - 1
        ess_p_cmd_norm, ess_q_cmd_norm = actions[ess_agent_id, 0], actions[ess_agent_id, 1]

        # ------------------------------------------------------------
        # [Fix] ESS normalized action -> physical MW/MVar conversion
        # - p_cmd_norm, q_cmd_norm in [-1, 1]
        # - convert using ESS rating (sn_mva)
        # - enforce inverter capability circle: p^2 + q^2 <= sn_mva^2 (priority to P, clip Q)
        # ------------------------------------------------------------
        ess_sn_mva = float(self.powergrid.storage.at[0, "sn_mva"]) if "sn_mva" in self.powergrid.storage.columns else 1.0
        ess_p_cmd_mw = ess_p_cmd_norm * ess_sn_mva

        # SoC physics integration & safety clipping
        dt = self.time_delta / 60.0  # time_delta is in minutes, convert to hours
        
        # Prevent discharging when empty
        if self.soc <= 0 and ess_p_cmd_mw > 0:
            ess_p_cmd_mw = 0.0
        # Prevent charging when full
        if self.soc >= 1 and ess_p_cmd_mw < 0:
            ess_p_cmd_mw = 0.0

        # Update stored last action for ESS after safety clipping (keep it normalized for obs)
        if ess_sn_mva > 0:
            self.last_actions[ess_agent_id, 0] = ess_p_cmd_mw / ess_sn_mva
        else:
            self.last_actions[ess_agent_id, 0] = 0.0

        # Update SoC based on the (potentially clipped) action
        # Note: discharging (p>0) decreases soc; ess_p_cmd_mw is in MW, ess_capacity is in MWh
        self.soc = self.soc - ess_p_cmd_mw * dt / self.ess_capacity
        
        # Clip SoC to be within [0, 1]
        self.soc = np.clip(self.soc, 0, 1)

        # Convert ESS reactive power action with capability constraint
        ess_q_max_mvar = np.sqrt(max(0.0, ess_sn_mva ** 2 - ess_p_cmd_mw ** 2))
        ess_q_cmd_mvar = ess_q_cmd_norm * ess_q_max_mvar
        # Reflect the actually applied normalized q after capability clipping
        if ess_q_max_mvar > 0:
            self.last_actions[ess_agent_id, 1] = ess_q_cmd_mvar / ess_q_max_mvar
        else:
            self.last_actions[ess_agent_id, 1] = 0.0

        # Write actions to pandapower grid
        # Apply PV reactive power actions
        self.powergrid.sgen["q_mvar"] = self._clip_reactive_power(pv_q_actions, self.powergrid.sgen["p_mw"])
        
        # Apply ESS active and reactive power actions. Note the sign convention for pandapower storage:
        # positive p_mw means charging, negative p_mw means discharging.
        # Our agent action ess_p_cmd_mw > 0 means discharging.
        self.powergrid.storage.at[0, 'p_mw'] = -ess_p_cmd_mw
        self.powergrid.storage.at[0, 'q_mvar'] = ess_q_cmd_mvar

        # solve power flow
        try:
            pp.runpp(self.powergrid)
            return True
        except ppException:
            print ("The power flow for the actions cannot be solved.")
            print (f"This is the pv: \n{self.powergrid.sgen['p_mw']}")
            print (f"This is the pv q: \n{self.powergrid.sgen['q_mvar']}")
            print (f"This is the ess p: \n{self.powergrid.storage['p_mw']}")
            print (f"This is the ess q: \n{self.powergrid.storage['q_mvar']}")
            print (f"This is the active demand: \n{self.powergrid.load['p_mw']}")
            print (f"This is the reactive demand: \n{self.powergrid.load['q_mvar']}")
            print (f"This is the res_bus: \n{self.powergrid.res_bus}")
            return False
    
    def _clip_reactive_power(self, reactive_actions, active_power):
        """clip the reactive power to the hard safety range
        """
        reactive_power_constraint = np.sqrt(self.s_max**2 - active_power**2)
        return reactive_power_constraint * reactive_actions
    
    def _calc_reward(self, info={}):
        """
        Reward function
        Combines voltage control reward with ESS-specific rewards.
        """
        # --- 1. Standard Info Dict Population ---
        v = self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)
        percent_of_v_out_of_control = ( np.sum(v < self.v_lower) + np.sum(v > self.v_upper) ) / v.shape[0]
        info["percentage_of_v_out_of_control"] = percent_of_v_out_of_control
        info["percentage_of_lower_than_lower_v"] = np.sum(v < self.v_lower) / v.shape[0]
        info["percentage_of_higher_than_upper_v"] = np.sum(v > self.v_upper) / v.shape[0]
        info["totally_controllable_ratio"] = 0. if percent_of_v_out_of_control > 1e-3 else 1.

        v_ref = 1.0 # Target voltage
        info["average_voltage_deviation"] = np.mean( np.abs( v - v_ref ) )
        info["average_voltage"] = np.mean(v)
        info["max_voltage_drop_deviation"] = np.max( (v < self.v_lower) * (self.v_lower - v) )
        info["max_voltage_rise_deviation"] = np.max( (v > self.v_upper) * (v - self.v_upper) )

        line_loss = np.sum(self.powergrid.res_line["pl_mw"])
        info["total_line_loss"] = line_loss
        r_loss = -10.0 * info["total_line_loss"]

        q = self.powergrid.res_sgen["q_mvar"].sort_index().to_numpy(copy=True)
        q_loss = np.mean(np.abs(q))
        info["q_loss"] = q_loss

        info["destroy"] = 0.0

        # --- 2. Get ESS State for Reward Calculation ---
        # res_storage p_mw is negative for discharging, positive for charging
        ess_p = self.powergrid.res_storage['p_mw'].iloc[0]
        soc = self.soc

        # --- 3. Calculate Reward Components ---
        # Voltage penalty via configured barrier (bowl, l1, courant_beltrami, etc.)
        r_voltage = -500.0 * np.sum(self.voltage_barrier.step(v))

        # SoC composite penalties
        deviation = max(0, abs(soc - 0.5) - 0.4)  # exceed 0.1/0.9
        r_soc_limit = -1000000.0 * (deviation ** 2)
        r_soc_center = -1000.0 * ((soc - 0.5) ** 2)
        r_ess_action = -10.0 * abs(ess_p)
        r_soc_total = r_soc_limit + r_soc_center + r_ess_action

        # --- 4. Combine Rewards ---
        reward = r_voltage + r_soc_total + r_loss

        # --- 5. Update Info Dict with Reward Components ---
        info["r_voltage"] = r_voltage
        info["r_soc_center"] = r_soc_center
        info["r_soc_limit"] = r_soc_limit
        info["r_ess_action"] = r_ess_action
        info["r_loss"] = r_loss

        return reward, info

    def _get_res_bus_v(self):
        v = self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)
        return v
    
    def _get_res_bus_active(self):
        active = self.powergrid.res_bus["p_mw"].sort_index().to_numpy(copy=True)
        return active

    def _get_res_bus_reactive(self):
        reactive = self.powergrid.res_bus["q_mvar"].sort_index().to_numpy(copy=True)
        return reactive

    def _get_res_line_loss(self):
        line_loss = self.powergrid.res_line["pl_mw"].sort_index().to_numpy(copy=True)
        return line_loss

    def _get_sgen_active(self):
        active = self.powergrid.sgen["p_mw"].to_numpy(copy=True)
        return active
    
    def _get_sgen_reactive(self):
        reactive = self.powergrid.sgen["q_mvar"].to_numpy(copy=True)
        return reactive
    
    def _init_render(self):
        from .rendering_voltage_control_env import Viewer
        self.viewer = Viewer()
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()
        return self.viewer.render(self, return_rgb_array=(mode == "rgb_array"))

    def res_pf_plot(self):
        if not os.path.exists("environments/var_voltage_control/plot_save"):
            os.mkdir("environments/var_voltage_control/plot_save")

        fig = pf_res_plotly(self.powergrid, 
                            aspectratio=(1.0, 1.0), 
                            filename="environments/var_voltage_control/plot_save/pf_res_plot.html", 
                            auto_open=False,
                            climits_volt=(0.9, 1.1),
                            line_width=5, 
                            bus_size=12,
                            climits_load=(0, 100),
                            cpos_load=1.1,
                            cpos_volt=1.0
                        )
        fig.write_image("environments/var_voltage_control/plot_save/pf_res_plot.jpeg")

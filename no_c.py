import numpy as np
import yaml
import pickle
import os
import argparse
from utilities.util import convert
from environments.var_voltage_control.voltage_control_env import VoltageControl

# ================= 配置区域 =================
# 必须和你的 test.py 保持一致
SCENARIO = "case33_3min_final"
TEST_DAY = 864
SAVE_DIR = r"C:\Users\Songy\Desktop\MAPDN-main"
OUTPUT_FILE = f"test_record_no_control_{SCENARIO}_day{TEST_DAY}.pickle"


# ===========================================

def run_no_control():
    print(f"🚀 开始运行无控制 (No Control) 基准测试...")
    print(f"场景: {SCENARIO}, 测试天数: Day {TEST_DAY}")

    # 1. 加载环境配置 (直接硬编码或读取yaml，这里为了简单直接复用逻辑)
    # 这里的路径假设你在项目根目录运行
    config_path = f"./args/env_args/var_voltage_control.yaml"
    with open(config_path, "r") as f:
        env_config_dict = yaml.safe_load(f)["env_args"]

    # 修正数据路径
    data_path = env_config_dict["data_path"].split("/")
    data_path[-1] = SCENARIO
    env_config_dict["data_path"] = "/".join(data_path)

    # 设置环境参数
    if SCENARIO == 'case33_3min_final':
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.8

    env_config_dict["mode"] = 'distributed'
    env_config_dict["voltage_barrier_type"] = 'l1'
    env_config_dict["episode_limit"] = 480  # 24小时

    # 2. 初始化环境
    env = VoltageControl(env_config_dict)
    n_agents = env.get_num_of_agents()
    n_actions = env.get_total_actions()  # 通常是1

    # 3. 手动重置到指定的一天
    # manual_reset(day, hour, interval) -> 23点开始? 原代码test.py里写的是23点，我们保持一致
    env.manual_reset(TEST_DAY, 23, 2)

    # 4. 准备记录容器
    record = {
        "bus_voltage": [],
        "total_line_loss": []
    }

    # 5. 开始循环 (480步 = 24小时)
    print("正在进行时域仿真 (Action = 0)...")
    for t in range(480):
        # 【修改点】的关键：生成全 0 的动作，并强制展平为一维数组！
        # 形状从 (n_agents, 1) 变为 (n_agents,)
        actions = np.zeros((n_agents, n_actions))

        # 这一步 env 会去执行 0 动作
        reward, done, info = env.step(actions, add_noise=False)

        # 记录电压
        v = env._get_res_bus_v()
        record["bus_voltage"].append(v)

        if done:
            break

    # 6. 保存结果
    save_path = os.path.join(SAVE_DIR, OUTPUT_FILE)
    with open(save_path, 'wb') as f:
        pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)

    print(f"✅ 无控制数据已保存: {save_path}")


if __name__ == "__main__":
    run_no_control()
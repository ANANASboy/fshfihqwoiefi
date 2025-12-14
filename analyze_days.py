import numpy as np
import yaml
import os
import sys
import pandas as pd
import multiprocessing
from environments.var_voltage_control.voltage_control_env import VoltageControl

# ================= 配置区域 =================
SCENARIO = "case33_3min_final"
SAVE_DIR = r"C:\Users\Songy\Desktop\MAPDN-main"
EXCEL_PREFIX = f"voltage_analysis_{SCENARIO}"
# ===========================================

# 全局变量：用于在每个子进程中存储独立的环境实例
worker_env = None


def init_worker(env_config):
    """
    子进程初始化函数。
    每个 CPU 核心只会运行一次这个函数，用于加载环境。
    """
    global worker_env
    # 不同的进程设置不同的随机种子（虽然这里我们用的是无噪声模式，但养成好习惯很重要）
    np.random.seed(os.getpid())
    try:
        worker_env = VoltageControl(env_config)
    except Exception as e:
        print(f"⚠️ 子进程 {os.getpid()} 初始化失败: {e}")


def analyze_single_day(day):
    """
    单天分析任务，将被并行执行。
    """
    global worker_env
    if worker_env is None:
        return None

    env = worker_env
    n_agents = env.get_num_of_agents()
    n_actions = env.get_total_actions()

    try:
        # 重置环境到指定天 (23点开始, 仿真24小时)
        env.manual_reset(day, 23, 2)
    except Exception:
        # 数据越界或其他错误
        return None

    max_v_daily = -1.0
    min_v_daily = 100.0

    # 运行 24 小时仿真 (480步)
    for t in range(480):
        # Action = 0 (No Control)
        actions = np.zeros((n_agents, n_actions)).flatten()
        _, done, _ = env.step(actions, add_noise=False)

        # 获取当前步的所有节点电压
        v = env._get_res_bus_v()

        # 更新当天的最大最小值
        max_v_daily = max(max_v_daily, np.max(v))
        min_v_daily = min(min_v_daily, np.min(v))

        if done:
            break

    # --- 分类逻辑 ---
    V_UPPER = 1.05
    V_LOWER = 0.95
    BUFFER = 0.005  # 0.005 p.u. 的缓冲区

    is_over_upper = max_v_daily > V_UPPER
    is_under_lower = min_v_daily < V_LOWER

    is_near_upper = (not is_over_upper) and (max_v_daily >= V_UPPER - BUFFER)
    is_near_lower = (not is_under_lower) and (min_v_daily <= V_LOWER + BUFFER)

    category = "Unknown"
    description = "未知"

    if is_over_upper and is_under_lower:
        category = "Over_Both"
        description = "既越上限又越下限 (最严重)"
    elif is_over_upper:
        category = "Over_Upper"
        description = "越上限日 (电压过高)"
    elif is_under_lower:
        category = "Under_Lower"
        description = "越下限日 (电压过低)"
    elif is_near_upper and is_near_lower:
        category = "Near_Both"
        description = "接近双边极限"
    elif is_near_upper:
        category = "Near_Upper"
        description = "接近上限"
    elif is_near_lower:
        category = "Near_Lower"
        description = "接近下限"
    else:
        category = "Normal"
        description = "正常日"

    return {
        "Day ID": day,
        "Max Voltage": max_v_daily,
        "Min Voltage": min_v_daily,
        "Category": category,
        "Description": description
    }


def analyze_days():
    # 解决 Windows 下多进程可能出现的 RuntimeError
    multiprocessing.freeze_support()

    print(f"🚀 初始化并行分析脚本...")
    print(f"场景: {SCENARIO}")

    # 1. 加载配置
    config_path = f"./args/env_args/var_voltage_control.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 找不到配置文件: {config_path}")
        return

    with open(config_path, "r") as f:
        env_config_dict = yaml.safe_load(f)["env_args"]

    # 修正路径
    data_path = env_config_dict["data_path"].split("/")
    data_path[-1] = SCENARIO
    env_config_dict["data_path"] = "/".join(data_path)

    if SCENARIO == 'case33_3min_final':
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.8
    env_config_dict["mode"] = 'distributed'
    env_config_dict["voltage_barrier_type"] = 'l1'
    env_config_dict["episode_limit"] = 480

    # 2. 临时创建一个环境来获取总天数（只需一次）
    try:
        print("正在读取数据范围...")
        temp_env = VoltageControl(env_config_dict)
        start_date = temp_env.pv_data.index[0]
        end_date = temp_env.pv_data.index[-1]
        total_days = (end_date - start_date).days
        VALID_END_DAY = total_days - 1
        print(f"✅ 数据范围: Day 0 - Day {VALID_END_DAY} (共 {total_days} 天)")
        del temp_env  # 释放内存
    except Exception as e:
        print(f"❌ 预加载环境失败: {e}")
        return

    excel_filename = f"{EXCEL_PREFIX}_day0-{VALID_END_DAY}.xlsx"

    # 3. 设置多进程
    # 获取 CPU 核心数，保留 1-2 个核心给系统，避免电脑卡死
    num_processes = max(1, multiprocessing.cpu_count() - 2)
    print(f"⚡ 启动并行计算池: {num_processes} 个核心同时工作...")
    print("=" * 50)

    all_data_records = []

    # 使用 Pool 进行并行计算
    # initializer=init_worker 会确保每个进程只加载一次 Pandas 数据
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(env_config_dict,)) as pool:
        # 创建任务列表
        days_to_process = range(0, VALID_END_DAY + 1)
        total_tasks = len(days_to_process)

        # 使用 imap_unordered 可以让结果无序返回（稍微快一点），我们后面再排序
        # 使用 enumerate 来显示进度
        for i, result in enumerate(pool.imap(analyze_single_day, days_to_process)):
            if result:
                all_data_records.append(result)

            # 简单的进度条
            percent = (i + 1) / total_tasks * 100
            sys.stdout.write(f"\r进度: [{i + 1}/{total_tasks}] {percent:.1f}%")
            sys.stdout.flush()

    print("\n\n" + "=" * 50)
    print("📊 分析完成，正在生成 Excel 报告...")

    # 转换为 DataFrame 并按 Day ID 排序
    df = pd.DataFrame(all_data_records)
    if not df.empty:
        df = df.sort_values(by="Day ID")

        # 保存 Excel
        excel_path = os.path.join(SAVE_DIR, excel_filename)
        try:
            df.to_excel(excel_path, index=False)
            print(f"✅ Excel 文件已生成: {excel_path}")
        except ImportError:
            print("❌ 请安装 openpyxl: pip install openpyxl")
            df.to_csv(excel_path.replace(".xlsx", ".csv"), index=False)
        except Exception as e:
            print(f"❌ 保存出错: {e}")

        # 统计摘要
        print("\n📈 统计摘要:")
        print(df["Description"].value_counts().to_string())
    else:
        print("⚠️ 未生成任何有效数据。")


if __name__ == "__main__":
    analyze_days()
import pickle
import os
import numpy as np

# ================= 配置区域 =================
# 你的目标文件路径
TARGET_DIR = r"C:\Users\Songy\Desktop\MAPDN-main"
FILE_NAME = "test_record_var_voltage_control-case33_3min_final-distributed-matd3-l1-reproduction_l1_batch.pickle"

# 拼接完整路径
FILE_PATH = os.path.join(TARGET_DIR, FILE_NAME)


# ===========================================

def load_and_print_results():
    print(f"正在尝试读取文件: {FILE_PATH} ...\n")

    if not os.path.exists(FILE_PATH):
        print(f"❌ 错误: 找不到文件！")
        print(f"请确认文件是否存在于: {TARGET_DIR}")
        print("提示: 如果文件名包含日期或别名(alias)，请修改脚本中的 FILE_NAME 变量。")
        return

    try:
        with open(FILE_PATH, 'rb') as f:
            data = pickle.load(f)

        print("✅ 文件读取成功！以下是测试结果分析：")
        print("=" * 70)
        print(f"{'指标名称 (Key)':<45} | {'数值 (Mean ± 2*Std)':<20}")
        print("-" * 70)

        # 核心论文指标提取
        # 论文主要看两个指标: Controllable Ratio (CR) 和 Average Voltage Deviation (AVD)

        # 1. 遍历所有数据并打印
        for key, value in data.items():
            if isinstance(value, (tuple, list)) and len(value) == 2:
                mean_val = value[0]
                two_std_val = value[1]
                print(f"{key:<45} | {mean_val:.4f} ± {two_std_val:.4f}")
            else:
                print(f"{key:<45} | {value}")

        print("=" * 70)
        print("\n📊 【论文核心指标解读】")

        # 尝试提取并解读关键指标
        cr_key = 'mean_test_totally_controllable_ratio'
        avd_key = 'mean_test_average_voltage_deviation'

        if cr_key in data:
            cr_mean = data[cr_key][0]
            print(f"1. 可控率 (CR / Controllable Ratio):")
            print(f"   结果: {cr_mean * 100:.2f}%")
            print(f"   解读: 这是最重要的指标。论文中 MATD3 在 case33 上的 SOTA 结果通常在 99% 以上。")
            print(f"         如果你的结果在 85% 左右，说明训练还需要继续，或者模型还在收敛中。")

        if avd_key in data:
            avd_mean = data[avd_key][0]
            print(f"\n2. 平均电压偏差 (AVD / Avg Voltage Deviation):")
            print(f"   结果: {avd_mean:.4f}")
            print(f"   解读: 该数值越小越好，表示电压波动被控制得越平稳。")

    except Exception as e:
        print(f"❌ 读取过程中发生错误: {e}")


if __name__ == "__main__":
    load_and_print_results()
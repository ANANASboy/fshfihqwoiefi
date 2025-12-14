import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置区域 =================
# 请确保这里的文件名是你刚刚用 single 模式跑出来的新文件！
# 文件名通常包含 "dayXXX" 字样
TARGET_DIR = r"C:\Users\Songy\Desktop\MAPDN-main"
# 举例: test_record_var_voltage_control-case33_3min_final-distributed-matd3-l1-0_day730.pickle
# 请替换为你实际生成的文件名
FILE_NAME = "test_record_var_voltage_control-case33_3min_final-distributed-matd3-l1-reproduction_l1_day730.pickle"

FILE_PATH = os.path.join(TARGET_DIR, FILE_NAME)


# ===========================================

def plot_voltage_curves():
    if not os.path.exists(FILE_PATH):
        print(f"❌ 找不到文件: {FILE_PATH}")
        print("请先运行: python test.py ... --test-mode single ...")
        return

    print(f"正在读取: {FILE_NAME} ...")
    with open(FILE_PATH, 'rb') as f:
        data = pickle.load(f)

    # data['bus_voltage'] 是一个 list，需要转换为 numpy 数组
    # 形状通常是 (时间步数, 节点数)
    voltages = np.array(data['bus_voltage'])

    steps, n_nodes = voltages.shape
    print(f"数据加载成功: 共 {steps} 个时间步 (3分钟/步), {n_nodes} 个节点")

    # 开始画图
    plt.figure(figsize=(12, 6), dpi=100)

    # 绘制所有节点的电压曲线
    # 使用灰色细线绘制普通节点，突出显示电压波动大的节点
    time_axis = np.arange(steps)

    print("正在绘图...")
    for i in range(n_nodes):
        # 简单过滤：只画出电压波动比较明显的节点，避免图太乱
        # 或者你可以选择画出所有节点: plt.plot(time_axis, voltages[:, i], alpha=0.5)
        plt.plot(time_axis, voltages[:, i], alpha=0.6, linewidth=1)

    # 绘制安全范围红线 (0.95 - 1.05 p.u.)
    plt.axhline(y=1.05, color='red', linestyle='--', linewidth=2, label='Upper Limit (1.05)')
    plt.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='Lower Limit (0.95)')
    plt.axhline(y=1.00, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Ref (1.00)')

    # 设置图表信息
    plt.title(f"Bus Voltages Over Time (Day 730)\nScenario: Case33 | Alg: MATD3", fontsize=14)
    plt.xlabel("Time Steps (3 min intervals)", fontsize=12)
    plt.ylabel("Voltage (p.u.)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.90, 1.10)  # 设置Y轴范围，聚焦于关键区域

    # 保存图片
    save_name = "voltage_plot_result.png"
    plt.savefig(os.path.join(TARGET_DIR, save_name))
    print(f"✅ 图片已保存为: {save_name}")
    plt.show()


if __name__ == "__main__":
    plot_voltage_curves()
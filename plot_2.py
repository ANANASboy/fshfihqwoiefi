import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置区域 =================
TARGET_DIR = r"C:\Users\Songy\Desktop\MAPDN-main"

# 文件 1: 无控制 (由 run_no_control.py 生成)
FILE_NO_CONTROL = "test_record_no_control_case33_3min_final_day864.pickle"

# 文件 2: 有控制 (由 test.py 生成，就是你刚才那个)
# 请确保文件名完全正确！
FILE_WITH_CONTROL = "test_record_var_voltage_control-case33_3min_final-distributed-matd3-l1-reproduction_l1_day734.pickle"


# ===========================================

def load_data(filename):
    path = os.path.join(TARGET_DIR, filename)
    if not os.path.exists(path):
        print(f"❌ 找不到文件: {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def plot_comparison():
    print("正在加载数据...")
    data_nc = load_data(FILE_NO_CONTROL)
    data_wc = load_data(FILE_WITH_CONTROL)

    if data_nc is None or data_wc is None:
        return

    # 提取电压数据 (Steps, Nodes)
    v_nc = np.array(data_nc['bus_voltage'])
    v_wc = np.array(data_wc['bus_voltage'])

    steps = v_nc.shape[0]
    time_axis = np.arange(steps)

    # 绘图设置
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, dpi=100)

    # --- 子图 1: 无控制 (No Control) ---
    ax1.set_title("Before Control (Baseline)", fontsize=14, fontweight='bold')
    ax1.plot(v_nc, color='gray', alpha=0.5, linewidth=1)
    # 标记越限区域
    ax1.axhline(1.05, color='red', linestyle='--', linewidth=2)
    ax1.axhline(0.95, color='red', linestyle='--', linewidth=2)
    ax1.set_ylabel("Voltage (p.u.)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.90, 1.10)
    # 在图里写个注释
    max_v_nc = np.max(v_nc)
    min_v_nc = np.min(v_nc)
    ax1.text(10, 1.08, f"Max V: {max_v_nc:.4f}\nMin V: {min_v_nc:.4f}",
             bbox=dict(facecolor='white', alpha=0.8))

    # --- 子图 2: 有控制 (With MATD3) ---
    ax2.set_title("After Control (MATD3)", fontsize=14, fontweight='bold')
    ax2.plot(v_wc, color='blue', alpha=0.5, linewidth=1)  # 用蓝色表示受控
    ax2.axhline(1.05, color='red', linestyle='--', linewidth=2, label='Limit')
    ax2.axhline(0.95, color='red', linestyle='--', linewidth=2)
    ax2.set_ylabel("Voltage (p.u.)", fontsize=12)
    ax2.set_xlabel("Time Steps (3 min)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.90, 1.10)

    max_v_wc = np.max(v_wc)
    min_v_wc = np.min(v_wc)
    ax2.text(10, 1.08, f"Max V: {max_v_wc:.4f}\nMin V: {min_v_wc:.4f}",
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()

    save_path = os.path.join(TARGET_DIR, "voltage_comparison.png")
    plt.savefig(save_path)
    print(f"✅ 对比图已保存: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_comparison()
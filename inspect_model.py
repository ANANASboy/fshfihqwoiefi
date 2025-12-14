import pandapower as pp
import pandas as pd

# 1. 加载模型 (请确保路径对应您实际的数据位置)
# 假设您想看 Case33 的数据
net = pp.from_pickle("./environments/var_voltage_control/data/case33_3min_final/model.p")

print("=== 1. 基本信息 ===")
print(net)  # 这会打印出电网的摘要，包括有多少个 bus, line, sgen 等

print("\n=== 2. 节点 (Bus) 信息 ===")
# 这里的 index 就是节点编号，vn_kv 是电压等级
print(net.bus)

print("\n=== 3. 光伏 (PV) 位置信息 ===")
# 在 pandapower 中，光伏通常存储在 static generators (sgen) 表里
# 查看 'bus' 列，就知道光伏装在哪个节点上了
# 查看 'p_mw' 列，可以看到额定有功功率
print(net.sgen[['name', 'bus', 'p_mw', 'q_mvar']])

print("\n=== 4. 线路 (Line) 信息 ===")
print(net.line[['from_bus', 'to_bus', 'length_km']])
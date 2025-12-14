import os
import argparse
import pandas as pd
import shutil
import time
from datetime import datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_scalars_to_csv(event_file_path, base_output_dir):
    """
    读取 TensorBoard 事件文件，导出 CSV，并自动备份源文件到带时间戳的文件夹中。
    """
    # 1. 检查源文件
    if not os.path.exists(event_file_path):
        print(f"错误: 找不到文件 {event_file_path}")
        return

    # 2. 生成时间戳 (例如: 20251207_143005)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 3. 创建带时间戳的专属输出目录
    # 例如: ./exported_data/20251207_143005_backup
    final_output_dir = os.path.join(base_output_dir, f"{timestamp_str}_backup")

    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
        print(f"✅ 已创建任务目录: {final_output_dir}")

    # 4. 备份源文件 (Raw Data Backup)
    print(f"📦 正在备份原始数据文件...")

    # (A) 备份 tfevents 二进制文件
    try:
        shutil.copy2(event_file_path, final_output_dir)
        print(f"   - 已备份: {os.path.basename(event_file_path)}")
    except Exception as e:
        print(f"   - 备份 tfevents 文件失败: {e}")

    # (B) 备份同目录下的 log.txt
    source_dir = os.path.dirname(event_file_path)
    log_txt_path = os.path.join(source_dir, "log.txt")

    if os.path.exists(log_txt_path):
        try:
            shutil.copy2(log_txt_path, final_output_dir)
            print(f"   - 已备份: log.txt")
        except Exception as e:
            print(f"   - 备份 log.txt 失败: {e}")
    else:
        print(f"   - 提示: 未在源目录找到 log.txt，跳过备份。")

    print("-" * 50)
    print(f"🚀 正在解析 TensorBoard 数据... (文件较大时请耐心等待)")

    # 5. 加载事件文件
    ea = EventAccumulator(event_file_path, size_guidance={'scalars': 0})
    ea.Reload()

    # 6. 获取所有标量标签
    tags = ea.Tags()['scalars']

    if not tags:
        print("❌ 未在文件中找到任何标量数据 (Scalars/曲线图)。")
        return

    print(f"📊 找到 {len(tags)} 条曲线数据，开始导出 CSV...")

    count = 0
    for tag in tags:
        # 获取数据
        events = ea.Scalars(tag)

        steps = [x.step for x in events]
        values = [x.value for x in events]
        wall_times = [x.wall_time for x in events]

        # 转换为 DataFrame
        df = pd.DataFrame({
            'step': steps,
            'value': values,
            'wall_time': wall_times
        })

        # 7. 生成文件名 (带时间戳)
        # 将 tag 中的 '/' 替换为 '_', 并加上时间后缀
        # 例如: data_r_soc_limit_20251207_143005.csv
        clean_tag_name = tag.replace('/', '_').replace('\\', '_')
        filename = f"{clean_tag_name}_{timestamp_str}.csv"
        output_path = os.path.join(final_output_dir, filename)

        # 保存
        df.to_csv(output_path, index=False)
        print(f"   - 导出: {filename}")
        count += 1

    print("-" * 50)
    print(f"🎉 全部完成！")
    print(f"📂 结果已保存在: {os.path.abspath(final_output_dir)}")
    print(f"   包含: {count} 个 CSV 数据表 + 原始数据备份")


if __name__ == '__main__':
    # 配置部分

    # 默认输出总目录 (脚本会在这个目录下自动新建带时间戳的子目录)
    DEFAULT_OUTPUT_ROOT = "./exported_data"

    # 默认日志文件路径 (你可以改成你的路径，或者让脚本自动搜)
    # 这里写相对路径即可
    # DEFAULT_LOG_DIR = "debug_logs/tensorboard"
    DEFAULT_LOG_DIR = "results/tensorboard/var_voltage_control-case33_3min_final-distributed-matd3-l1-production_run_v1_1207"
    parser = argparse.ArgumentParser(description="Export TensorBoard events to CSV with Backup.")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR, help="Directory containing the tfevents file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_ROOT, help="Root directory to save exports")

    args = parser.parse_args()

    # 自动寻找最新的日志文件
    target_log_file = None

    # 如果用户给的是个文件路径，直接用
    if os.path.isfile(args.log_dir):
        target_log_file = args.log_dir
    # 如果给的是目录，去里面找最新的 tfevents
    elif os.path.isdir(args.log_dir):
        print(f"🔍 正在目录 '{args.log_dir}' 中搜索最新的日志文件...")
        all_files = []
        for root, dirs, files in os.walk(args.log_dir):
            for file in files:
                if "events.out.tfevents" in file:
                    full_path = os.path.join(root, file)
                    all_files.append(full_path)

        if all_files:
            # 按修改时间排序，取最新的
            target_log_file = max(all_files, key=os.path.getmtime)
            print(f"👉 自动选中最新文件: {target_log_file}")
        else:
            print(f"❌ 错误: 在 '{args.log_dir}' 下没有找到任何 tfevents 文件。")
            exit(1)
    else:
        print(f"❌ 错误: 路径 '{args.log_dir}' 不存在。")
        exit(1)

    # 执行主逻辑
    extract_scalars_to_csv(target_log_file, args.output_dir)
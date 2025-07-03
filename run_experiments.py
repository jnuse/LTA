import subprocess
import os
import sys

# --- 配置 ---
PYTHON_EXECUTABLE = sys.executable  # 获取当前虚拟环境的 Python 解释器路径
CONFIG_FILE = "config_Clinc_LSTM.json"
NUM_RUNS = 10
DEVICE_ID = 0
RESULTS_DIR = "results"

def run_command(command):
    """执行一个 shell 命令并检查其返回值"""
    print(f"Executing: {' '.join(command)}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Return code: {result.returncode}")
        sys.exit(1) # 如果任何一步失败，则终止整个脚本

def main():
    """主函数，用于运行所有实验"""
    # 确保结果目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for i in range(1, NUM_RUNS + 1):
        print("\n" + "="*60)
        print(f"======== Running Experiment {i} of {NUM_RUNS} ========")
        print("="*60 + "\n")

        # --- 第一阶段：预训练 ---
        print(f"--- Stage 1: Pre-training for run {i} ---")
        stage1_command = [
            PYTHON_EXECUTABLE, "train.py",
            "--config", CONFIG_FILE,
            "-d", str(DEVICE_ID)
        ]
        run_command(stage1_command)

        # --- 第二阶段：元学习/微调 ---
        print(f"\n--- Stage 2: Meta-learning for run {i} ---")
        # 我们将 run_id 传递给训练脚本，以便它知道如何保存结果文件
        stage2_command = [
            PYTHON_EXECUTABLE, "train.py",
            "--config", CONFIG_FILE,
            "-st2", "True",
            "-d", str(DEVICE_ID),
            "--run_id", str(i) # 传递当前运行的ID
        ]
        run_command(stage2_command)

    print("\n" + "="*60)
    print("All 10 experiment runs completed successfully.")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
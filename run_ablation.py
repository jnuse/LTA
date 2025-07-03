import subprocess
import os
import sys
import json

# --- 配置 ---
PYTHON_EXECUTABLE = sys.executable  # 获取当前虚拟环境的 Python 解释器路径
CONFIG_STEP1 = "config_step1_pretrain.json"
CONFIG_FILES_STEP2 = [
    "config_Clinc_LSTM.json", # Baseline
    "config_ablation_no_init.json",
    "config_ablation_no_proto_adapt.json",
    "config_ablation_no_feat_adapt.json"
]
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
    """主函数，用于运行所有消融实验"""
    # 确保结果目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 第一阶段：预训练 (如果需要则运行一次) ---
    proto_file = "data/ver1/Clinc/protos_lstm.pkl"
    print("\n" + "="*60)
    if not os.path.exists(proto_file):
        print(f"======== Pre-trained proto file not found. Running Pre-training (Step 1) ========")
        print("="*60 + "\n")
        stage1_command = [
            PYTHON_EXECUTABLE, "train.py",
            "--config", CONFIG_STEP1,
            "-d", str(DEVICE_ID)
        ]
        run_command(stage1_command)
    else:
        print(f"======== Found pre-trained proto file. Skipping Pre-training (Step 1) ========")
        print("="*60 + "\n")


    # --- 第二阶段：元学习/微调 (为每个配置运行) ---
    for config_file in CONFIG_FILES_STEP2:
        # 从配置文件名生成一个唯一的 run_id
        run_id = os.path.splitext(os.path.basename(config_file))[0]

        print("\n" + "="*60)
        print(f"======== Running Ablation Experiment (Step 2) for: {run_id} ========")
        print("="*60 + "\n")

        stage2_command = [
            PYTHON_EXECUTABLE, "train.py",
            "--config", config_file,
            "-d", str(DEVICE_ID),
            "--run_id", run_id # 传递唯一的 run_id
        ]
        run_command(stage2_command)

    print("\n" + "="*60)
    print("All ablation experiments completed successfully.")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
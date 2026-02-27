import os
import sys
import shutil
import subprocess

# 导入你工程中原生的测试模块
import evaluate
from run_baselines import run_baselines


def main():
    loads = [80, 100, 120, 140, 160]

    main_ckpt = "your ckpt_path"
    if not os.path.exists(main_ckpt):
        main_ckpt = "your ckpt_path"

    print(f"🔍 锁定 TP-GNN-PPO 权重路径: {main_ckpt}")


    has_backup = False
    if os.path.exists("eval_results.csv"):
        shutil.move("eval_results.csv", "eval_results_backup.csv")
        has_backup = True

    try:
        for req in loads:
            print(f"\n{'=' * 50}")
            print(f"🚀 开始测试流量负载: {req} Requests / Seed")
            print(f"{'=' * 50}")

            # 1. 运行传统启发式 -> 自动追加到 eval_results.csv
            run_baselines(num_eval_episodes=5, requests_per_ep=req)

            # 2. 运行TP-GNN-PPO -> 自动追加到 eval_results.csv
            if os.path.exists(main_ckpt):
                evaluate.evaluate_model(ckpt_path=main_ckpt, num_eval_episodes=5, requests_per_ep=req)
            else:
                print(f"❌ 警告: 找不到主模型权重 {main_ckpt}，跳过 TP-GNN-PPO。")

            # 3. 运行 PPO (w/o GNN) -> 自动追加到 eval_results.csv
            subprocess.run([
                sys.executable, "run_ppo_only.py",
                "--skip_train",
                "--eval_req", str(req)
            ], check=True)

        if os.path.exists("eval_results.csv"):
            shutil.move("eval_results.csv", "pressure_test_results.csv")
            print("\n🎉 所有真实压测已完成！纯净数据已保存至: pressure_test_results.csv")
        else:
            print("\n❌ 压测未能生成任何结果文件！")

    finally:

        if has_backup:
            if os.path.exists("eval_results.csv"):
                os.remove("eval_results.csv")  # 清理中途异常残留
            shutil.move("eval_results_backup.csv", "eval_results.csv")
            print("🔄 原 eval_results.csv 数据已安全恢复。")


if __name__ == "__main__":
    main()
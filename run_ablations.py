import os
import pandas as pd
import torch

# 导入你的核心代码
import main
import evaluate
from configs.Config import Config


def run_ablation_experiment(ablation_name, modify_cfg_func, enforce_trust_flag=True):
    print(f"\n{'=' * 50}")
    print(f"🚀 开始跑消融实验: {ablation_name}")
    print(f"{'=' * 50}")


    modify_cfg_func()

    exp_dir = f"ABLATION_{ablation_name.replace('/', '_').replace(' ', '_')}"
    ckpt_dir = f"checkpoints/{exp_dir}/main_seed42"

    best_ckpt = os.path.join(ckpt_dir, "your ckpt_path")
    final_ckpt = os.path.join(ckpt_dir, "your ckpt_path")

    target_ckpt = None
    if os.path.exists(best_ckpt):
        target_ckpt = best_ckpt
        print(f"🎯 发现已存在的 {best_ckpt}，直接跳过训练！")
    elif os.path.exists(final_ckpt):
        target_ckpt = final_ckpt
        print(f"🎯 发现已跑完的 {final_ckpt}，直接跳过训练！")
    else:
        print("⏳ 未发现完整的训练记录，训练 250 轮...")
        main.run_train(
            exp_name=exp_dir,
            mode="main",
            seed=42,
            num_episodes=250,
            requests_per_episode=120,
            train_gnn=True,
            enforce_trust=enforce_trust_flag,
            ckpt_every=50,
            eval_every=999,
            resume_ckpt=None,
            init_ckpt=None,
            gnn_out_dim=None,
        )

        if os.path.exists(best_ckpt):
            target_ckpt = best_ckpt
        elif os.path.exists(final_ckpt):
            target_ckpt = final_ckpt
        else:
            pts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
            if pts: target_ckpt = os.path.join(ckpt_dir, sorted(pts)[-1])

    if target_ckpt is None or not os.path.exists(target_ckpt):
        raise FileNotFoundError(f"在 {ckpt_dir} 下找不到任何可用于评估的模型！")

    print(f"\n📊 训练/查找完毕，开始评估 {ablation_name} ...")
    print(f"-> 即将用于评估的模型是: {target_ckpt}")

    csv_path = "eval_results.csv"
    old_len = len(pd.read_csv(csv_path)) if os.path.exists(csv_path) else 0

    evaluate.evaluate_model(ckpt_path=target_ckpt, num_eval_episodes=5, requests_per_ep=120)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        new_len = len(df)
        if new_len > old_len:
            df.loc[old_len:new_len - 1, "algorithm"] = ablation_name
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"✅ {ablation_name} 的结果已成功追加并重命名")


# ==========================================
# 实验 1: w/o curriculum (无课程学习)
# ==========================================
def disable_curriculum():
    Config.TRUST_ENFORCE_WARMUP_EP = 0
    Config.TRUST_ENFORCE_RAMP_EP = 0
    Config.TRUST_THRESHOLD_START = 0.72
    Config.TRUST_THRESHOLD = 0.72


# ==========================================
# 实验 2: w/o trust mask (无信任掩码拦截)
# ==========================================
def keep_curriculum_but_no_mask():

    Config.TRUST_ENFORCE_WARMUP_EP = 20
    Config.TRUST_ENFORCE_RAMP_EP = 100
    Config.TRUST_THRESHOLD_START = 0.66
    Config.TRUST_THRESHOLD = 0.72

if __name__ == "__main__":
    run_ablation_experiment("w/o trust mask", keep_curriculum_but_no_mask, enforce_trust_flag=False)


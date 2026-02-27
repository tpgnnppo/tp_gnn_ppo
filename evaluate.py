import os
import time
import json
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import torch


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from configs.Config import Config
from env.network_env import PhysicalNetwork
from models.ppo_agent import PPOAgent
from models.tp_gnn import TPGNN

from main import StateProjector, make_base_h, gnn_forward


# ------------------------- Utils -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------- 主评估循环 -------------------------
def evaluate_model(ckpt_path: str, num_eval_episodes: int = 5, requests_per_ep: int = 120):
    print(f"Loading checkpoint from: {ckpt_path}")

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_env = PhysicalNetwork(seed=42)
    feat_dim = int(dummy_env.get_node_features().shape[1])
    num_nodes = int(cfg.NUM_NODES)
    out_dim = int(getattr(cfg, "GNN_OUT_DIM", 128))
    max_delay_tol = float(getattr(cfg, "SFC_MAX_DELAY_TOL", 60.0))  # 读取地狱难度时延

    gnn = TPGNN(input_dim=feat_dim, hidden_dim=cfg.GNN_HIDDEN_DIM, output_dim=out_dim).to(device)
    ppo = PPOAgent(state_dim=out_dim, action_dim=num_nodes, device=device)
    projector = StateProjector(h_dim=out_dim, cond_dim=6).to(device)

    print("Loading weights (strict=True)...")
    payload = torch.load(ckpt_path, map_location=device)

    if "ppo" in payload:
        ppo.load_state_dict(payload["ppo"], strict=True)
    if "gnn" in payload:
        gnn.load_state_dict(payload["gnn"], strict=True)
    if "projector" in payload:
        projector.load_state_dict(payload["projector"], strict=True)
        print("✅ Projector 权重加载成功！")
    else:
        print("⚠️ 警告：Checkpoint 中依然没有找到 Projector 权重！")

    gnn.eval()
    ppo.policy.eval()
    projector.eval()

    cond_div = torch.tensor([100.0, 100.0, 100.0, 1.0, 1.0, 1.0], device=device)

    results = []
    eval_seeds = [1001, 2500, 1003, 1004, 1005][:num_eval_episodes]

    for ep, test_seed in enumerate(eval_seeds):
        set_seed(test_seed)  # 设置 numpy/torch 随机种子以保证请求生成的随机性
        env = PhysicalNetwork(seed=42)
        env.reset()
        env.propagate_trust()

        ep_succ = 0
        ep_delay_sum = 0.0

        fail_trust_empty = 0
        fail_no_path = 0
        fail_resource = 0

        print(
            f"  Evaluating Test Sequence {ep + 1}/{num_eval_episodes} (Req Seed: {test_seed}, {requests_per_ep} reqs)...")

        trust_scores_np = np.array([float(env.graph.nodes[i].get("trust_score", 1.0)) for i in range(num_nodes)],
                                   dtype=np.float32)

        with torch.no_grad():
            for req_id in range(requests_per_ep):
                env.reset_request_budget()

                sub_seed = (test_seed * 1000003 + ep * 9176 + req_id * 131) % (2 ** 32 - 1)
                rng = np.random.default_rng(sub_seed)

                vnf_num = int(rng.integers(cfg.VNF_NUM_RANGE[0], cfg.VNF_NUM_RANGE[1] + 1))
                vnf_list = [
                    {"cpu_req": int(rng.integers(cfg.VNF_CPU_REQ_RANGE[0], cfg.VNF_CPU_REQ_RANGE[1] + 1)),
                     "mem_req": int(rng.integers(cfg.VNF_MEM_REQ_RANGE[0], cfg.VNF_MEM_REQ_RANGE[1] + 1))}
                    for _ in range(vnf_num)
                ]
                bw_req = float(rng.integers(cfg.SFC_BW_REQ_RANGE[0], cfg.SFC_BW_REQ_RANGE[1] + 1))

                last_node = None
                req_ok = True
                req_delay = 0.0

                node_emb_req = gnn_forward(env, gnn, device=device)

                for k, vnf in enumerate(vnf_list):
                    is_last_vnf = (k == len(vnf_list) - 1)

                    resource_mask = env.get_action_mask(vnf, last_node=last_node, enforce_trust=False, episode_idx=999,
                                                        sfc_bw_req=0.0, max_delay_tol=max_delay_tol).astype(bool)
                    link_mask = env.get_action_mask(vnf, last_node=last_node, enforce_trust=False, episode_idx=999,
                                                    sfc_bw_req=bw_req, max_delay_tol=max_delay_tol).astype(bool)
                    trust_mask = link_mask & (trust_scores_np >= cfg.TRUST_THRESHOLD)

                    mask_to_use = trust_mask

                    if mask_to_use.sum() == 0:
                        req_ok = False
                        if link_mask.sum() > 0:
                            fail_trust_empty += 1  # 物理上通，但全是不安全节点
                        elif resource_mask.sum() > 0:
                            fail_no_path += 1  # 节点有CPU，但没带宽或者断连
                        else:
                            fail_resource += 1  # 连基本的CPU/内存都没了
                        break

                    # 构造 State
                    base_h = make_base_h(node_emb_req, last_node)
                    cpu_d = float(vnf["cpu_req"])
                    mem_d = float(vnf["mem_req"])
                    prog = float(k) / max(1.0, float(len(vnf_list) - 1))
                    delay_ratio = float(env.req_delay_used / max(1e-6, max_delay_tol))

                    cond = torch.tensor([cpu_d, mem_d, float(bw_req), prog, delay_ratio, 1.0], dtype=torch.float32,
                                        device=device)
                    state_vec = projector(base_h, cond / cond_div)

                    # ---- 策略选择 ----
                    mask_t = torch.tensor(mask_to_use, dtype=torch.bool, device=device)
                    logits = ppo.policy.actor(state_vec).squeeze(-1)
                    masked_logits = logits.masked_fill(~mask_t, -1e9)



                    probs = torch.nn.functional.softmax(masked_logits, dim=-1)
                    dist = torch.distributions.Categorical(probs=probs)
                    action = int(dist.sample().item())

                    # 物理步进
                    _, _, done, info = env.step(
                        int(action), vnf, bw_req, last_node=last_node,
                        enforce_trust=True, episode_idx=999,
                        max_delay_tol=max_delay_tol,
                        trust_threshold=cfg.TRUST_THRESHOLD
                    )

                    req_delay += float(info.get("delay_added", 0.0))
                    last_node = int(action)

                    if done:
                        req_ok = False
                        fr = info.get("fail_reason", "")
                        if "trust" in fr:
                            fail_trust_empty += 1
                        elif "bw" in fr or "path" in fr:
                            fail_no_path += 1
                        else:
                            fail_resource += 1
                        break

                if req_ok:
                    ep_succ += 1
                    ep_delay_sum += req_delay

        acc = ep_succ / requests_per_ep
        avg_delay = (ep_delay_sum / ep_succ) if ep_succ > 0 else 0.0

        no_path_rate = fail_no_path / requests_per_ep
        trust_empty_rate = fail_trust_empty / requests_per_ep
        resource_fail_rate = fail_resource / requests_per_ep

        results.append({
            "algorithm": "TP-GNN-PPO",
            "seed": test_seed,
            "requests": requests_per_ep,
            "acc": acc,
            "avg_delay": avg_delay,
            "no_path_rate": no_path_rate,
            "trust_mask_empty_rate": trust_empty_rate,
            "resource_fail_rate": resource_fail_rate
        })
        print(f"    -> Acc: {acc:.4f} | Avg Delay: {avg_delay:.4f}")
        print(
            f"      -> 💀 死因解剖: No_Path={no_path_rate:.4f} | Trust_Violation={trust_empty_rate:.4f} | Resource_Out={resource_fail_rate:.4f}")

    df = pd.DataFrame(results)
    out_csv = "eval_results.csv"

    if os.path.exists(out_csv):
        df.to_csv(out_csv, mode='a', index=False, header=False)
    else:
        df.to_csv(out_csv, index=False)

    print(f"\n✅ 评估闭环完成，结果已追加至 {out_csv}")
    print("\n==== 测试集平均表现 ====")
    print(df.mean(numeric_only=True))


if __name__ == "__main__":
    CKPT_PATH = r"D:\tp_gnn_oo\checkpoints\TP_GNN_PPO_MAIN_ULTIMATE\main_seed42\ep0800.pt"

    if not os.path.exists(CKPT_PATH):
        print(f"❌ 找不到权重文件：{CKPT_PATH}，请确认训练是否完成，或路径是否正确！")
    else:
        evaluate_model(CKPT_PATH, num_eval_episodes=5, requests_per_ep=120)
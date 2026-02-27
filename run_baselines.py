import os
import random
import numpy as np
import pandas as pd

# 🚨 核心修改 1：全局导入唯一的 Config！
from configs.Config import Config
from env.network_env import PhysicalNetwork


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_baselines(num_eval_episodes=5, requests_per_ep=120):
    print("\n========== 启动启发式基线评估==========")
    cfg = Config()
    eval_seeds = [1001, 2500, 1003, 1004, 1005][:num_eval_episodes]

    methods = ["Random-Feasible", "Greedy-Resource"]
    results = []

    for method in methods:
        print(f"\n>>> 开始评估 Baseline: {method} <<<")
        for ep, seed in enumerate(eval_seeds):
            set_seed(seed)
            env = PhysicalNetwork(seed=seed)
            env.reset()
            env.propagate_trust()  # 确保信任值传播

            ep_succ = 0
            ep_delay_sum = 0.0

            fail_trust_empty = 0
            fail_no_path = 0
            fail_resource = 0

            for req_id in range(requests_per_ep):
                env.reset_request_budget()
                sub_seed = (seed * 1000003 + ep * 9176 + req_id * 131) % (2 ** 32 - 1)
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

                trust_scores_np = np.array(
                    [float(env.graph.nodes[i].get("trust_score", 1.0)) for i in range(cfg.NUM_NODES)])

                for k, vnf in enumerate(vnf_list):
                    resource_mask = env.get_action_mask(vnf, last_node=last_node, enforce_trust=False, episode_idx=999,
                                                        sfc_bw_req=0.0, max_delay_tol=cfg.SFC_MAX_DELAY_TOL).astype(
                        bool)
                    link_mask = env.get_action_mask(vnf, last_node=last_node, enforce_trust=False, episode_idx=999,
                                                    sfc_bw_req=bw_req, max_delay_tol=cfg.SFC_MAX_DELAY_TOL).astype(bool)
                    trust_mask = link_mask & (trust_scores_np >= cfg.TRUST_THRESHOLD)

                    mask_to_use = trust_mask

                    if mask_to_use.sum() == 0:
                        req_ok = False
                        if link_mask.sum() > 0:
                            fail_trust_empty += 1
                        elif resource_mask.sum() > 0:
                            fail_no_path += 1
                        else:
                            fail_resource += 1
                        break

                    valid_nodes = np.where(mask_to_use)[0]

                    if method == "Random-Feasible":
                        action = rng.choice(valid_nodes)

                    elif method == "Greedy-Resource":
                        best_action = valid_nodes[0]
                        max_res = -1
                        for node in valid_nodes:
                            node_data = env.graph.nodes[node]
                            cpu_rem = node_data.get('cpu_remain', node_data.get('cpu', 0))
                            mem_rem = node_data.get('mem_remain', node_data.get('mem', 0))
                            res_score = float(cpu_rem) + float(mem_rem)
                            if res_score > max_res:
                                max_res = res_score
                                best_action = node
                        action = best_action
                    # =================================

                    _, _, done, info = env.step(
                        int(action), vnf, bw_req, last_node=last_node,
                        enforce_trust=True, episode_idx=999,
                        max_delay_tol=cfg.SFC_MAX_DELAY_TOL,
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
                "algorithm": method,
                "seed": seed,
                "requests": requests_per_ep,
                "acc": acc,
                "avg_delay": avg_delay,
                "no_path_rate": no_path_rate,
                "trust_mask_empty_rate": trust_empty_rate,
                "resource_fail_rate": resource_fail_rate
            })
            print(f"    Seed {seed}: Acc = {acc:.4f} | Avg Delay = {avg_delay:.4f}")
            print(
                f"      -> 💀 死因解剖: No_Path={no_path_rate:.4f} | Trust_Violation={trust_empty_rate:.4f} | Resource_Out={resource_fail_rate:.4f}")

    # 保存并追加 CSV
    df = pd.DataFrame(results)
    out_csv = "eval_results.csv"

    if os.path.exists(out_csv):
        df.to_csv(out_csv, mode='a', index=False, header=False)
    else:
        df.to_csv(out_csv, index=False)

    print(f"\n✅ 基线评估完成，结果已追加至 {out_csv}")


if __name__ == "__main__":
    run_baselines(num_eval_episodes=5, requests_per_ep=120)
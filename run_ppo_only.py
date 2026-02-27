
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from env.network_env import PhysicalNetwork
from configs.Config import Config
from models.ppo_agent import PPOAgent


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_eval_csv(csv_path: str, rows: list[dict]):
    df_new = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def sample_request(cfg: Config, rng: np.random.Generator):
    vnf_num = int(rng.integers(cfg.VNF_NUM_RANGE[0], cfg.VNF_NUM_RANGE[1] + 1))
    vnf_list = [
        {
            "cpu_req": int(rng.integers(cfg.VNF_CPU_REQ_RANGE[0], cfg.VNF_CPU_REQ_RANGE[1] + 1)),
            "mem_req": int(rng.integers(cfg.VNF_MEM_REQ_RANGE[0], cfg.VNF_MEM_REQ_RANGE[1] + 1)),
        }
        for _ in range(vnf_num)
    ]
    bw_req = float(rng.integers(cfg.SFC_BW_REQ_RANGE[0], cfg.SFC_BW_REQ_RANGE[1] + 1))
    return vnf_list, bw_req


def thr_schedule(ep: int, thr_final: float, thr_start: float = 0.66, warmup: int = 50, ramp: int = 150) -> float:
    """Stable->main style threshold ramp."""
    if ep <= warmup:
        return thr_start
    if ep <= warmup + ramp:
        t = (ep - warmup) / float(ramp)
        return thr_start + t * (thr_final - thr_start)
    return thr_final


def build_state(env: PhysicalNetwork,
                node_feats_np: np.ndarray,
                trust_scores_np: np.ndarray,
                last_node: int | None,
                vnf: dict,
                bw_req: float,
                step_k: int,
                chain_len: int,
                max_delay_tol: float,
                thr: float,
                base_mask: np.ndarray,
                trust_mask: np.ndarray) -> np.ndarray:

    cpu = node_feats_np[:, 0].astype(np.float32)
    mem = node_feats_np[:, 1].astype(np.float32)
    tr  = trust_scores_np.astype(np.float32)

    # A) global stats
    global_stats = np.array([
        cpu.mean(), cpu.std(), cpu.min(),
        mem.mean(), mem.std(), mem.min(),
        tr.mean(),  tr.std(),  tr.min(),
        float((tr >= thr).mean()),
    ], dtype=np.float32)

    # B) base feasible set stats
    bm = base_mask.astype(bool)
    if bm.any():
        cpu_b = cpu[bm]; mem_b = mem[bm]; tr_b = tr[bm]
        base_stats = np.array([
            float(bm.mean()),          # |A_base|/N
            cpu_b.mean(), mem_b.mean(), tr_b.mean(),
            tr_b.min(),
            cpu_b.min(),
        ], dtype=np.float32)
    else:
        base_stats = np.zeros((6,), dtype=np.float32)

    # C) trust feasible set stats
    tm = trust_mask.astype(bool)
    if tm.any():
        cpu_t = cpu[tm]; mem_t = mem[tm]; tr_t = tr[tm]
        trust_stats = np.array([
            float(tm.mean()),          # |A_trust|/N
            cpu_t.mean(), mem_t.mean(), tr_t.mean(),
        ], dtype=np.float32)
    else:
        trust_stats = np.zeros((4,), dtype=np.float32)

    # D) last node features
    if last_node is None:
        last = np.zeros((3,), dtype=np.float32)
    else:
        last = np.array([cpu[last_node], mem[last_node], tr[last_node]], dtype=np.float32)

    # E) request context
    cpu_req = float(vnf["cpu_req"]) / 100.0
    mem_req = float(vnf["mem_req"]) / 100.0
    bw_norm = float(bw_req) / 100.0
    prog = float(step_k) / max(1.0, float(chain_len - 1))
    delay_ratio = float(env.req_delay_used / max(1e-6, float(max_delay_tol)))
    ctx = np.array([cpu_req, mem_req, bw_norm, prog, delay_ratio, float(thr)], dtype=np.float32)

    return np.concatenate([global_stats, base_stats, trust_stats, last, ctx], axis=0)


def train(cfg: Config,
          topo_seed: int,
          train_seed: int,
          episodes: int,
          requests_per_ep: int,
          max_delay_tol: float,
          thr_final: float,
          lr: float,
          save_path: str):
    set_seed(train_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = PhysicalNetwork(seed=topo_seed)
    env.reset()
    n_nodes = int(env.config.NUM_NODES)

    state_dim = 29  # enhanced state with candidate-set stats
    ppo = PPOAgent(state_dim=state_dim, action_dim=n_nodes, device=device, lr=lr)

    rng_base = np.random.default_rng(train_seed)

    print(f"[PPO-only-v3] Train | topo_seed={topo_seed} episodes={episodes} thr_final={thr_final}")

    for ep in range(1, episodes + 1):
        env.reset()
        env.propagate_trust()

        thr_ep = thr_schedule(ep, thr_final)
        succ = 0
        delay_sum = 0.0

        ep_rng = np.random.default_rng(int(rng_base.integers(0, 2**32 - 1)))

        for _ in range(requests_per_ep):
            env.reset_request_budget()
            vnf_list, bw_req = sample_request(cfg, ep_rng)

            trust_scores_np = np.array(
                [float(env.graph.nodes[i].get("trust_score", 1.0)) for i in range(env.config.NUM_NODES)],
                dtype=np.float32
            )

            last_node = None
            req_ok = True
            req_delay = 0.0

            for k, vnf in enumerate(vnf_list):
                base_mask = env.get_action_mask(
                    vnf, last_node=last_node,
                    enforce_trust=False, episode_idx=ep,
                    sfc_bw_req=bw_req, max_delay_tol=max_delay_tol
                ).astype(bool)

                if base_mask.sum() == 0:
                    req_ok = False
                    break  # IMPORTANT: no action -> do NOT append reward/done

                trust_mask = base_mask & (trust_scores_np >= thr_ep)
                if trust_mask.sum() == 0:
                    req_ok = False
                    break

                node_feats_np = env.get_node_features()
                state = build_state(env, node_feats_np, trust_scores_np, last_node, vnf, bw_req, k, len(vnf_list),
                                    max_delay_tol=max_delay_tol, thr=thr_ep,
                                    base_mask=base_mask, trust_mask=trust_mask)

                mask_t = torch.tensor(trust_mask, dtype=torch.bool, device=device)
                action = ppo.select_action(state, action_mask=mask_t)

                _, reward, done, info = env.step(
                    int(action), vnf, bw_req, last_node=last_node,
                    enforce_trust=True, episode_idx=ep,
                    max_delay_tol=max_delay_tol,
                    trust_threshold=float(thr_ep)
                )

                # Only append reward/terminal AFTER an action was taken (PPOAgent requires lengths match)
                terminal = bool(done) or (k == len(vnf_list) - 1) or (not info.get("success", False))
                r = float(reward)
                # NEW: request-level terminal bonus for better credit assignment
                if terminal:
                    r += 1.0 if info.get("success", False) else 0.0
                ppo.buffer.rewards.append(r)
                ppo.buffer.is_terminals.append(terminal)

                if not info.get("success", False) or done:
                    req_ok = False
                    break

                req_delay += float(info.get("delay_added", 0.0))
                last_node = int(action)

            if req_ok:
                succ += 1
                delay_sum += req_delay

        ppo.update()

        if ep % 20 == 0:
            acc = succ / max(1, requests_per_ep)
            avg_delay = delay_sum / max(1, succ) if succ > 0 else 0.0
            print(f"  Ep {ep:4d} | thr={thr_ep:.3f} | Acc={acc:.3f} | AvgDelay={avg_delay:.2f}")

    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    torch.save({
        "ppo": ppo.state_dict(),
        "state_dim": state_dim,
        "action_dim": n_nodes,
        "topo_seed": topo_seed,
        "train_seed": train_seed,
        "episodes": episodes,
        "thr_final": thr_final,
        "max_delay_tol": max_delay_tol,
    }, save_path)
    print(f"[PPO-only-v3] Saved -> {save_path}")
    return save_path


@torch.no_grad()
def evaluate(cfg: Config,
             model_path: str,
             topo_seed: int,
             req_seeds: list[int],
             requests: int,
             max_delay_tol: float,
             thr_eval: float,
             out_csv: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)

    ppo = PPOAgent(state_dim=int(ckpt["state_dim"]), action_dim=int(ckpt["action_dim"]), device=device)
    ppo.load_state_dict(ckpt["ppo"], strict=True)

    # PPOAgent is not nn.Module; set internal nets to eval if available
    if hasattr(ppo, "policy") and hasattr(ppo.policy, "eval"):
        ppo.policy.eval()
    if hasattr(ppo, "policy_old") and hasattr(ppo.policy_old, "eval"):
        ppo.policy_old.eval()

    env = PhysicalNetwork(seed=topo_seed)

    rows = []
    for rs in req_seeds:
        env.reset()
        env.propagate_trust()
        rng = np.random.default_rng(rs)

        succ = 0
        delay_sum = 0.0
        no_path = 0
        trust_empty = 0
        resource_fail = 0

        trust_scores_np = np.array(
            [float(env.graph.nodes[i].get("trust_score", 1.0)) for i in range(env.config.NUM_NODES)],
            dtype=np.float32
        )

        for _ in range(requests):
            env.reset_request_budget()
            vnf_list, bw_req = sample_request(cfg, rng)

            last_node = None
            req_ok = True
            req_delay = 0.0

            for k, vnf in enumerate(vnf_list):
                base_mask = env.get_action_mask(
                    vnf, last_node=last_node,
                    enforce_trust=False, episode_idx=999,
                    sfc_bw_req=bw_req, max_delay_tol=max_delay_tol
                ).astype(bool)

                if base_mask.sum() == 0:
                    req_ok = False
                    resource_fail += 1
                    break

                trust_mask = base_mask & (trust_scores_np >= thr_eval)
                if trust_mask.sum() == 0:
                    req_ok = False
                    trust_empty += 1
                    break

                node_feats_np = env.get_node_features()
                state = build_state(env, node_feats_np, trust_scores_np, last_node, vnf, bw_req, k, len(vnf_list),
                                    max_delay_tol=max_delay_tol, thr=thr_eval,
                                    base_mask=base_mask, trust_mask=trust_mask)
                mask_t = torch.tensor(trust_mask, dtype=torch.bool, device=device)

                action, bad = ppo.select_action_deterministic(state, action_mask=mask_t)
                if bad:
                    req_ok = False
                    trust_empty += 1
                    break

                _, _, done, info = env.step(
                    int(action), vnf, bw_req, last_node=last_node,
                    enforce_trust=True, episode_idx=999,
                    max_delay_tol=max_delay_tol,
                    trust_threshold=float(thr_eval)
                )

                if not info.get("success", False):
                    req_ok = False
                    fr = info.get("fail_reason", "")
                    if fr == "no_path":
                        no_path += 1
                    elif fr in ("cpu", "mem", "bw", "delay"):
                        resource_fail += 1
                    elif fr == "trust":
                        trust_empty += 1
                    break

                req_delay += float(info.get("delay_added", 0.0))
                last_node = int(action)
                if done:
                    req_ok = False
                    break

            if req_ok:
                succ += 1
                delay_sum += req_delay

        acc = succ / max(1, requests)
        avg_delay = delay_sum / max(1, succ) if succ > 0 else 0.0

        rows.append({
            "algorithm": "PPO-only-v3",
            "seed": int(rs),
            "requests": int(requests),
            "acc": float(acc),
            "avg_delay": float(avg_delay),
            "no_path_rate": float(no_path / max(1, requests)),
            "trust_mask_empty_rate": float(trust_empty / max(1, requests)),
            "resource_fail_rate": float(resource_fail / max(1, requests)),
        })
        print(f"[PPO-only-v3][seed={rs}] Acc={acc:.4f} Delay={avg_delay:.4f}")

    append_eval_csv(out_csv, rows)
    print(f"[PPO-only-v3] Appended -> {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo_seed", type=int, default=42)
    ap.add_argument("--train_seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--train_req", type=int, default=120)
    ap.add_argument("--eval_req", type=int, default=120)
    ap.add_argument("--req_seeds", type=str, default="1001,1002,1003,1004,1005")
    ap.add_argument("--thr_final", type=float, default=0.72)
    ap.add_argument("--max_delay", type=float, default=60.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out_csv", type=str, default="eval_results.csv")
    ap.add_argument("--model_path", type=str, default="checkpoints/baselines/ppo_only_v3.pt")
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    req_seeds = [int(x.strip()) for x in args.req_seeds.split(",") if x.strip()]

    if (not args.skip_train) or (not os.path.exists(args.model_path)):
        train(cfg, topo_seed=args.topo_seed, train_seed=args.train_seed,
              episodes=args.episodes, requests_per_ep=args.train_req,
              max_delay_tol=args.max_delay, thr_final=args.thr_final,
              lr=args.lr, save_path=args.model_path)

    evaluate(cfg, model_path=args.model_path, topo_seed=args.topo_seed,
             req_seeds=req_seeds, requests=args.eval_req,
             max_delay_tol=args.max_delay, thr_eval=args.thr_final,
             out_csv=args.out_csv)


if __name__ == "__main__":
    main()

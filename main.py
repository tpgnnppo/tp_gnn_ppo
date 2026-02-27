
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import random
import csv
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import torch

from configs.Config import Config
from env.network_env import PhysicalNetwork
from models.ppo_agent import PPOAgent
from models.tp_gnn import TPGNN


# ------------------------- Utils -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def save_ckpt(path, *, episode, global_step, ppo: PPOAgent, gnn: TPGNN, projector: torch.nn.Module, metrics: dict,
              config: dict):
    payload = {
        "episode": episode,
        "global_step": global_step,
        "ppo": ppo.state_dict(),
        "gnn": gnn.state_dict(),
        "projector": projector.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    torch.save(payload, path)


def load_ckpt(path, *, ppo: PPOAgent, gnn: TPGNN, projector: torch.nn.Module, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    if "ppo" in payload:
        ppo.load_state_dict(payload["ppo"], strict=True)
    if "gnn" in payload:
        try:
            gnn.load_state_dict(payload["gnn"], strict=True)
        except Exception:
            gnn.load_state_dict(payload["gnn"], strict=False)
    if "projector" in payload:
        projector.load_state_dict(payload["projector"], strict=True)
    return payload


def load_pt_payload(pt_path: str, map_location: str = "cpu"):
    if pt_path is None:
        return None
    if not isinstance(pt_path, str):
        raise TypeError(f"pt_path must be str, got {type(pt_path)}")
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Checkpoint not found: {pt_path}")

    payload = torch.load(pt_path, map_location=map_location)
    if payload is None:
        raise ValueError(f"Empty checkpoint payload: {pt_path}")
    return payload


def annealed_trust_threshold(cfg: Config, episode_idx: int) -> float:
    thr_final = float(getattr(cfg, "TRUST_THRESHOLD", 0.70))
    thr_start = float(getattr(cfg, "TRUST_THRESHOLD_START", 0.50))
    use_anneal = bool(getattr(cfg, "TRUST_THRESHOLD_ANNEAL", True))
    if not use_anneal:
        return thr_final

    warm = int(getattr(cfg, "TRUST_ENFORCE_WARMUP_EP", 0))
    ramp_end = int(getattr(cfg, "TRUST_ENFORCE_RAMP_EP", warm))

    if episode_idx <= warm:
        return thr_start
    if ramp_end <= warm:
        return thr_final

    alpha = (episode_idx - warm) / max(1e-6, float(ramp_end - warm))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return thr_start + alpha * (thr_final - thr_start)


def trust_enforce_on(cfg: Config, episode_idx: int, base_enforce: bool) -> bool:
    if not base_enforce:
        return False
    warm = int(getattr(cfg, "TRUST_ENFORCE_WARMUP_EP", 0))
    return episode_idx >= warm


@dataclass
class EpisodeLog:
    ep: int
    thr: float
    trust_hard: int
    trust_expected_steps: int
    trust_fallback_steps: int
    fallback_rate: float
    avg_req_reward: float
    ma_reward: float
    acc: float
    avg_delay: float
    m_res: float
    m_link: float
    m_trust: float
    cand_res_avg: float
    cand_link_avg: float
    cand_trust_avg: float
    cand_res_min: int
    cand_link_min: int
    cand_trust_min: int
    trust_all_mean: float
    trust_all_p10: float
    trust_all_p50: float
    trust_all_p90: float
    kl: float
    entropy: float
    actor_loss: float
    dt_sec: float
    fails_json: str


class CSVLogger:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        ensure_dir(os.path.dirname(csv_path))

    def write(self, row: EpisodeLog):
        fieldnames = list(EpisodeLog.__annotations__.keys())
        write_header = (not os.path.exists(self.csv_path)) or (os.path.getsize(self.csv_path) == 0)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow(row.__dict__)


@torch.no_grad()
def build_edge_index_and_prop(env: PhysicalNetwork, device):
    g = env.graph
    src, dst, props = [], [], []
    for u, v, data in g.edges(data=True):
        pc = float(data.get("prop_coeff", 1.0))
        src.append(int(u));
        dst.append(int(v));
        props.append(pc)
        src.append(int(v));
        dst.append(int(u));
        props.append(pc)

    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        prop_coeffs = torch.zeros((0,), dtype=torch.float32, device=device)
        return edge_index, prop_coeffs

    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    prop_coeffs = torch.tensor(props, dtype=torch.float32, device=device)
    return edge_index, prop_coeffs


@torch.no_grad()
def build_trust_scores(env: PhysicalNetwork, num_nodes: int, device):
    g = env.graph
    ts = [float(g.nodes[i].get("trust_score", 1.0)) for i in range(num_nodes)]
    return torch.tensor(ts, dtype=torch.float32, device=device)


@torch.no_grad()
def gnn_forward(env: PhysicalNetwork, gnn: TPGNN, device):
    node_feats = env.get_node_features()
    x = torch.tensor(node_feats, dtype=torch.float32, device=device)
    num_nodes = x.shape[0]
    edge_index, prop_coeffs = build_edge_index_and_prop(env, device=device)
    trust_scores = build_trust_scores(env, num_nodes=num_nodes, device=device)
    out = gnn(x, edge_index=edge_index, trust_scores=trust_scores, prop_coeffs=prop_coeffs)
    return out


class StateProjector(torch.nn.Module):
    def __init__(self, h_dim: int, cond_dim: int = 6):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(h_dim + cond_dim, h_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(h_dim, h_dim),
        )

    def forward(self, base_h: torch.Tensor, cond: torch.Tensor):
        x = torch.cat([base_h, cond], dim=-1)
        return self.net(x)


@torch.no_grad()
def make_base_h(node_emb: torch.Tensor, last_node: Optional[int]):
    g = node_emb.mean(dim=0)
    if last_node is None:
        return g
    return 0.5 * g + 0.5 * node_emb[last_node]


# ------------------------- Training -------------------------
def run_train(
        exp_name: str = "TPGNN_PPO_MAIN",
        mode: str = "stable",
        seed: int = 42,
        num_episodes: int = 350,
        requests_per_episode: int = 50,
        train_gnn: bool = False,
        enforce_trust: bool = True,
        ckpt_every: int = 25,
        eval_every: int = 25,
        resume_ckpt: Optional[str] = None,
        init_ckpt: Optional[str] = None,
        gnn_out_dim: Optional[int] = None,
):
    cfg = Config()

    init_payload = None
    if init_ckpt is not None:
        init_payload = load_pt_payload(init_ckpt)
        ckpt_cfg = init_payload.get("cfg", {}) if isinstance(init_payload, dict) else {}
        ckpt_out_dim = ckpt_cfg.get("GNN_OUT_DIM", None)
        if gnn_out_dim is None and ckpt_out_dim is not None:
            cfg.GNN_OUT_DIM = int(ckpt_out_dim)

    if gnn_out_dim is not None:
        cfg.GNN_OUT_DIM = int(gnn_out_dim)

    print("\n[CFG] TRUST_ENFORCE_WARMUP_EP =", int(getattr(cfg, "TRUST_ENFORCE_WARMUP_EP", 0)))
    print("[CFG] TRUST_ENFORCE_RAMP_EP   =", int(getattr(cfg, "TRUST_ENFORCE_RAMP_EP", 0)))
    print("[CFG] TRUST_THRESHOLD_FINAL   =", float(getattr(cfg, "TRUST_THRESHOLD", 0.7)))
    print("[CFG] TRUST_THRESHOLD_START   =", float(getattr(cfg, "TRUST_THRESHOLD_START", 0.5)))
    print("[CFG] REQUESTS_PER_EPISODE    =", int(requests_per_episode))
    print("[CFG] FAST_PATH               =", bool(getattr(cfg, "FAST_PATH", True)))
    print("[CFG] TRUST_FALLBACK_TRAIN    =", bool(getattr(cfg, "TRUST_FALLBACK_TRAIN", False)))

    torch.set_num_threads(int(getattr(cfg, "TORCH_NUM_THREADS", 0)) or os.cpu_count() or 8)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = PhysicalNetwork(seed=seed)
    env.reset()

    feat_dim = int(env.get_node_features().shape[1])
    num_nodes = int(cfg.NUM_NODES)

    hidden_dim = int(getattr(cfg, "GNN_HIDDEN_DIM", 128))
    out_dim = int(getattr(cfg, "GNN_OUT_DIM", 128))

    gnn = TPGNN(input_dim=feat_dim, hidden_dim=hidden_dim, output_dim=out_dim).to(device)

    ppo = PPOAgent(
        state_dim=out_dim,
        action_dim=num_nodes,
        device=device,
        lr=float(getattr(cfg, "LR", 3e-4)),
        gamma=float(getattr(cfg, "GAMMA", 0.99)),
        gae_lambda=float(getattr(cfg, "GAE_LAMBDA", 0.95)),
        eps_clip=float(getattr(cfg, "PPO_CLIP", 0.2)),
        K_epochs=int(getattr(cfg, "PPO_EPOCHS", 4)),
        entropy_coef=float(getattr(cfg, "ENTROPY_COEF", 0.01)),
        value_coef=float(getattr(cfg, "VALUE_COEF", 0.5)),
        max_grad_norm=float(getattr(cfg, "MAX_GRAD_NORM", 0.5)),
        minibatch_size=int(getattr(cfg, "MINIBATCH_SIZE", 256)),
        target_kl=float(getattr(cfg, "TARGET_KL", 0.02)),
    )

    projector = StateProjector(h_dim=out_dim, cond_dim=6).to(device)

    if not train_gnn:
        for p in gnn.parameters():
            p.requires_grad_(False)
        for p in projector.parameters():
            p.requires_grad_(False)
        gnn.eval()
        projector.eval()
    else:
        for p in projector.parameters():
            p.requires_grad_(True)
        projector.train()

    per_step_gnn = False

    ckpt_root = ensure_dir(os.path.join(cfg.CKPT_DIR, exp_name, f"{mode}_seed{seed}"))
    log_root = ensure_dir(os.path.join(ckpt_root, "logs"))
    csv_path = os.path.join(log_root, "train.csv")
    logger = CSVLogger(csv_path)

    print(f"\n=== EXP START | mode={mode} | nodes={num_nodes} | feat_dim={feat_dim} | per_step_gnn={per_step_gnn} ===")
    print(f"Checkpoints -> {ckpt_root}")
    print(f"Logs        -> {log_root}")

    start_ep = 1
    global_step = 0
    best_ma = -1e9
    ma_reward = None
    best_comp = -1e9
    best_comp_score = None

    if (resume_ckpt is None) and (init_ckpt is not None) and os.path.exists(init_ckpt):
        print(f"[INIT] warm-start weights from {init_ckpt} | reset optimizer/episode")
        payload = init_payload if init_payload is not None else load_pt_payload(init_ckpt)
        if isinstance(payload, dict):
            if "ppo" in payload:
                ppo.load_state_dict(payload["ppo"], strict=False)
                print(f"[INIT] PPO loaded (strict=False)")
            if "gnn" in payload:
                try:
                    gnn.load_state_dict(payload["gnn"], strict=False)
                    print(f"[INIT] GNN loaded (strict=False)")
                except Exception as e:
                    print(f"[INIT] GNN load failed: {e}")
            if "projector" in payload:
                try:
                    projector.load_state_dict(payload["projector"], strict=False)
                    print(f"[INIT] Projector loaded (strict=False)")
                except Exception as e:
                    print(f"[INIT] Projector load failed: {e}")

    if resume_ckpt is not None and os.path.exists(resume_ckpt):
        payload = load_ckpt(resume_ckpt, ppo=ppo, gnn=gnn, projector=projector, map_location=device)
        start_ep = int(payload.get("episode", 0)) + 1
        global_step = int(payload.get("global_step", 0))
        best_ma = float(payload.get("metrics", {}).get("best_ma", best_ma))
        ma_reward = float(payload.get("metrics", {}).get("ma_reward", 0.0))
        best_comp = float(payload.get("metrics", {}).get("best_comp", best_comp))
        best_comp_score = float(payload.get("metrics", {}).get("best_comp_score",
                                                               best_comp_score if best_comp_score is not None else best_comp))
        print(f"[RESUME] from {resume_ckpt} | start_ep={start_ep}")

    cond_div = torch.tensor([100.0, 100.0, 100.0, 1.0, 1.0, 1.0], device=device)

    for ep in range(start_ep, num_episodes + 1):
        t0 = time.time()

        env.reset()
        env.propagate_trust()

        trust_scores_np = np.array(
            [float(env.graph.nodes[i].get("trust_score", 1.0)) for i in range(num_nodes)],
            dtype=np.float32
        )

        trust_hard = trust_enforce_on(cfg, ep, enforce_trust)
        thr_ep = annealed_trust_threshold(cfg, ep)
        max_delay_tol = float(getattr(cfg, "SFC_MAX_DELAY_TOL", 60.0))

        ep_req_rewards = []
        ep_succ = 0
        ep_delay_sum = 0.0

        trust_expected_steps = 0
        trust_fallback_steps = 0

        avg_res_mask = []
        avg_link_mask = []
        avg_trust_mask = []

        cand_res_list: List[int] = []
        cand_link_list: List[int] = []
        cand_trust_list: List[int] = []

        fails: Dict[str, int] = {
            "cpu": 0, "mem": 0, "bw": 0, "delay": 0, "trust": 0, "no_path": 0,
            "resource_mask_empty": 0, "link_mask_empty": 0, "trust_mask_empty": 0,
            "unknown": 0, "bad_mask_steps": 0,
        }

        for req_id in range(requests_per_episode):
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
            req_reward = 0.0
            req_delay = 0.0
            req_ok = True

            node_emb_req = gnn_forward(env, gnn, device=device)

            for k, vnf in enumerate(vnf_list):
                is_last_vnf = (k == len(vnf_list) - 1)
                node_emb = node_emb_req

                resource_mask = env.get_action_mask(
                    vnf, last_node=last_node,
                    enforce_trust=False, episode_idx=ep,
                    sfc_bw_req=bw_req, max_delay_tol=max_delay_tol
                ).astype(bool)

                avg_res_mask.append(float(resource_mask.sum()))
                if resource_mask.sum() == 0:
                    req_ok = False
                    fails["resource_mask_empty"] += 1
                    break

                link_mask = env.get_action_mask(
                    vnf, last_node=last_node,
                    enforce_trust=False, episode_idx=ep,
                    sfc_bw_req=bw_req, max_delay_tol=max_delay_tol
                ).astype(bool)

                avg_link_mask.append(float(link_mask.sum()))
                if link_mask.sum() == 0:
                    req_ok = False
                    fails["link_mask_empty"] += 1
                    break

                enforce_trust_step = bool(trust_hard)
                mask_to_use = link_mask
                used_trust_relax = False

                if trust_hard:
                    trust_expected_steps += 1
                    trust_mask = link_mask & (trust_scores_np >= thr_ep)
                    avg_trust_mask.append(float(trust_mask.sum()))

                    if trust_mask.sum() == 0:
                        fails["trust_mask_empty"] += 1
                        idx_link = np.where(link_mask)[0]
                        if getattr(cfg, "TRUST_FALLBACK_TRAIN", False) and idx_link.size > 0:
                            k_relax = int(getattr(cfg, "TRUST_RELAX_TOPK", 3))
                            k_relax = max(1, min(k_relax, int(idx_link.size)))
                            order = np.argsort(trust_scores_np[idx_link])
                            top_idx = idx_link[order[-k_relax:]]
                            relax_mask = np.zeros_like(link_mask, dtype=bool)
                            relax_mask[top_idx] = True

                            mask_to_use = relax_mask
                            enforce_trust_step = False
                            used_trust_relax = True
                            trust_fallback_steps += 1
                        else:
                            mask_to_use = trust_mask
                            enforce_trust_step = True
                    else:
                        mask_to_use = trust_mask
                        enforce_trust_step = True
                else:
                    avg_trust_mask.append(float(link_mask.sum()))
                    mask_to_use = link_mask
                    enforce_trust_step = False

                cand_res_list.append(int(resource_mask.sum()))
                cand_link_list.append(int(link_mask.sum()))
                cand_trust_list.append(int(mask_to_use.sum()))

                base_h = make_base_h(node_emb, last_node)

                cpu_d = float(vnf["cpu_req"])
                mem_d = float(vnf["mem_req"])
                bw_d = float(bw_req)
                prog = float(k) / max(1.0, float(len(vnf_list) - 1))
                delay_ratio = float(env.req_delay_used / max(1e-6, max_delay_tol))
                trust_flag = 1.0 if enforce_trust_step else 0.0

                cond = torch.tensor([cpu_d, mem_d, bw_d, prog, delay_ratio, trust_flag],
                                    dtype=torch.float32, device=device)
                cond_norm = cond / cond_div
                state_vec = projector(base_h, cond_norm)

                mask_t = torch.tensor(mask_to_use, dtype=torch.bool, device=device)

                # 平滑死亡：无路可走时的正确截断，保证长度对齐
                if mask_t.sum().item() == 0:
                    req_ok = False
                    fails["bad_mask_steps"] += 1
                    if k > 0 and len(ppo.buffer.is_terminals) > 0:
                        ppo.buffer.is_terminals[-1] = True
                    break

                action = ppo.select_action(state_vec, action_mask=mask_t)

                _, env_reward, done, info = env.step(
                    int(action), vnf, bw_req, last_node=last_node,
                    enforce_trust=enforce_trust_step, episode_idx=ep,
                    max_delay_tol=max_delay_tol,
                    trust_threshold=thr_ep
                )
                global_step += 1


                if done and not is_last_vnf:

                    reward = -10.0
                elif is_last_vnf:

                    reward = 10.0
                else:

                    reward = 0.1
                # ==========================================
                # ==========================================

                terminal = bool(done) or bool(is_last_vnf)
                ppo.buffer.rewards.append(float(reward))
                ppo.buffer.is_terminals.append(bool(terminal))

                req_reward += float(reward)
                req_delay += float(info.get("delay_added", 0.0))

                if done:
                    req_ok = False
                    fr = info.get("fail_reason", "unknown") or "unknown"
                    if fr not in fails:
                        fr = "unknown"
                    fails[fr] += 1
                    break

                last_node = int(action)
                if is_last_vnf:
                    req_ok = True
                    break

            ep_req_rewards.append(req_reward)
            if req_ok:
                ep_succ += 1
                ep_delay_sum += req_delay

        ppo.update()

        avg_req_r = float(np.mean(ep_req_rewards)) if ep_req_rewards else 0.0
        acc = ep_succ / max(1, requests_per_episode)
        avg_delay = (ep_delay_sum / max(1, ep_succ)) if ep_succ > 0 else 0.0
        ma_reward = avg_req_r if ma_reward is None else (0.95 * ma_reward + 0.05 * avg_req_r)

        fallback_rate = (trust_fallback_steps / max(1, trust_expected_steps)) if trust_expected_steps > 0 else 0.0
        m_res = float(np.mean(avg_res_mask)) if avg_res_mask else 0.0
        m_link = float(np.mean(avg_link_mask)) if avg_link_mask else 0.0
        m_trust = float(np.mean(avg_trust_mask)) if avg_trust_mask else 0.0

        cand_res_avg = float(np.mean(cand_res_list)) if cand_res_list else 0.0
        cand_link_avg = float(np.mean(cand_link_list)) if cand_link_list else 0.0
        cand_trust_avg = float(np.mean(cand_trust_list)) if cand_trust_list else 0.0
        cand_res_min = int(np.min(cand_res_list)) if cand_res_list else 0
        cand_link_min = int(np.min(cand_link_list)) if cand_link_list else 0
        cand_trust_min = int(np.min(cand_trust_list)) if cand_trust_list else 0

        _all = trust_scores_np.astype(np.float32)
        trust_all_mean = float(_all.mean())
        trust_all_p10 = float(np.quantile(_all, 0.10))
        trust_all_p50 = float(np.quantile(_all, 0.50))
        trust_all_p90 = float(np.quantile(_all, 0.90))

        dt = time.time() - t0

        ks = getattr(ppo, "last_update_stats", {}) or {}
        kl = float(ks.get("kl", 0.0)) if isinstance(ks, dict) else 0.0
        ent = float(ks.get("entropy", 0.0)) if isinstance(ks, dict) else 0.0
        aloss = float(ks.get("actor_loss", 0.0)) if isinstance(ks, dict) else 0.0

        print(
            f"Ep{ep} | thr={thr_ep:.3f} | trust_hard={int(trust_hard)} | exp={trust_expected_steps} | "
            f"R={avg_req_r:.2f} | MA={ma_reward:.2f} | Acc={acc:.3f} | D={avg_delay:.1f} | fb={fallback_rate:.3f} | "
            f"cand(res/link/tr)={cand_res_avg:.1f}/{cand_link_avg:.1f}/{cand_trust_avg:.1f} (min {cand_res_min}/{cand_link_min}/{cand_trust_min}) | "
            f"trust(p10/p50/p90)={trust_all_p10:.3f}/{trust_all_p50:.3f}/{trust_all_p90:.3f} mean={trust_all_mean:.3f} | "
            f"mask(res/link/tr)={m_res:.1f}/{m_link:.1f}/{m_trust:.1f} | {fails} | kl={kl:.4f} ent={ent:.3f} aloss={aloss:.3f} | {dt:.1f}s"
        )

        logger.write(EpisodeLog(
            ep=ep, thr=float(thr_ep), trust_hard=int(trust_hard),
            trust_expected_steps=int(trust_expected_steps),
            trust_fallback_steps=int(trust_fallback_steps),
            fallback_rate=float(fallback_rate),
            avg_req_reward=float(avg_req_r), ma_reward=float(ma_reward),
            acc=float(acc), avg_delay=float(avg_delay),
            m_res=float(m_res), m_link=float(m_link), m_trust=float(m_trust),
            cand_res_avg=float(cand_res_avg), cand_link_avg=float(cand_link_avg),
            cand_trust_avg=float(cand_trust_avg),
            cand_res_min=int(cand_res_min), cand_link_min=int(cand_link_min), cand_trust_min=int(cand_trust_min),
            trust_all_mean=float(trust_all_mean),
            trust_all_p10=float(trust_all_p10),
            trust_all_p50=float(trust_all_p50),
            trust_all_p90=float(trust_all_p90),
            kl=float(kl), entropy=float(ent), actor_loss=float(aloss),
            dt_sec=float(dt),
            fails_json=json.dumps(fails, ensure_ascii=False),
        ))

        if (ep % ckpt_every) == 0:
            save_ckpt(
                os.path.join(ckpt_root, f"ep{ep:04d}.pt"),
                episode=ep,
                global_step=global_step,
                ppo=ppo,
                gnn=gnn,
                projector=projector,
                metrics={"ma_reward": ma_reward, "best_ma": best_ma, "best_comp": best_comp, "acc": acc, "thr": thr_ep,
                         "fb": fallback_rate},
                config={"exp_name": exp_name, "mode": mode, "seed": seed, "train_gnn": train_gnn,
                        "enforce_trust": enforce_trust},
            )

        if ma_reward > best_ma:
            best_ma = float(ma_reward)
            save_ckpt(
                os.path.join(ckpt_root, "best.pt"),
                episode=ep,
                global_step=global_step,
                ppo=ppo,
                gnn=gnn,
                projector=projector,
                metrics={"ma_reward": ma_reward, "best_ma": best_ma, "best_comp": best_comp, "acc": acc, "thr": thr_ep,
                         "fb": fallback_rate},
                config={"exp_name": exp_name, "mode": mode, "seed": seed, "train_gnn": train_gnn,
                        "enforce_trust": enforce_trust},
            )

        trust_empty_rate = float(fails.get("trust_mask_empty", 0)) / float(max(1, requests_per_episode))
        delay_norm = float(avg_delay) / float(max(1e-6, max_delay_tol))
        score_comp = (
                1.0 * float(ma_reward)
                + 3.0 * float(acc)
                - 2.0 * float(trust_empty_rate)
                - 0.2 * float(delay_norm)
                - 0.5 * float(fallback_rate)
        )

        eligible = (float(acc) >= float(getattr(cfg, "BEST_GATE_ACC", 0.90))) and \
                   (trust_empty_rate <= float(getattr(cfg, "BEST_GATE_TRUST_EMPTY", 0.05)))
        if eligible and (score_comp > best_comp):
            best_comp = float(score_comp)
            best_comp_score = float(score_comp)
            save_ckpt(
                os.path.join(ckpt_root, "best_comp.pt"),
                episode=ep,
                global_step=global_step,
                ppo=ppo,
                gnn=gnn,
                projector=projector,
                metrics={
                    "ma_reward": ma_reward, "best_ma": best_ma,
                    "best_comp": best_comp, "best_comp_score": best_comp_score,
                    "acc": acc, "thr": thr_ep, "fb": fallback_rate,
                    "trust_empty_rate": trust_empty_rate, "avg_delay": avg_delay,
                },
                config={"exp_name": exp_name, "mode": mode, "seed": seed, "train_gnn": train_gnn,
                        "enforce_trust": enforce_trust},
            )

    print("\nTraining done.")
    print(f"Checkpoints saved to: {ckpt_root}")
    print(f"CSV logs saved to: {os.path.join(ckpt_root, 'logs', 'train.csv')}")


if __name__ == "__main__":
    run_train(
        exp_name="TP_GNN_PPO_MAIN_ULTIMATE",
        mode="main",
        seed=42,
        num_episodes=800,
        requests_per_episode=120,
        train_gnn=True,
        enforce_trust=True,
        ckpt_every=50,
        eval_every=50,
        resume_ckpt=None,
        init_ckpt=None,
        gnn_out_dim=None,
    )
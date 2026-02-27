# env/network_env.py  (REPLACE WHOLE FILE)

import numpy as np
import networkx as nx
from configs.Config import Config


class PhysicalNetwork:


    def __init__(self, seed: int = 42):
        self.config = Config()
        self.rng = np.random.RandomState(seed)
        self.seed = seed

        self.graph = self._generate_topology()

        self._trust_base = np.array([
            float(self.graph.nodes[i].get("trust_score", 0.0)) for i in self.graph.nodes()
        ], dtype=np.float32)

        self.req_delay_used = 0.0


        self._delay_dist = None
        self._delay_path = None
        self._build_delay_shortest_paths_once()

        self.reset_episode_stats()

    def _build_delay_shortest_paths_once(self):
        fast = bool(getattr(self.config, "FAST_PATH", True))
        if not fast:
            self._delay_path = None
            self._delay_dist = None
            return

        self._delay_path = dict(nx.all_pairs_dijkstra_path(self.graph, weight="delay"))
        self._delay_dist = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight="delay"))

    def reset_episode_stats(self):
        self.stats = {
            "total_reqs": 0,
            "succ_reqs": 0,
            "total_delay": 0.0,
            "fails": {
                "cpu": 0, "mem": 0, "bw": 0, "delay": 0, "trust": 0,
                "no_path": 0, "resource_mask_empty": 0, "trust_mask_empty": 0, "unknown": 0
            },
        }

    def _generate_topology(self):
        G = nx.powerlaw_cluster_graph(
            n=self.config.NUM_NODES, m=3, p=0.1, seed=self.seed
        )

        for i in G.nodes():
            cpu = self.rng.randint(*self.config.CPU_RANGE)
            mem = self.rng.randint(*self.config.MEM_RANGE)

            G.nodes[i]["cpu_capacity"] = cpu
            G.nodes[i]["cpu_remaining"] = cpu
            G.nodes[i]["mem_capacity"] = mem
            G.nodes[i]["mem_remaining"] = mem

            a = float(getattr(self.config, "TRUST_BETA_A", 5))
            b = float(getattr(self.config, "TRUST_BETA_B", 2))
            G.nodes[i]["trust_score"] = float(self.rng.beta(a, b))

        for u, v in G.edges():
            bw = self.rng.randint(*self.config.BW_RANGE)
            delay = float(self.rng.uniform(*self.config.DELAY_RANGE))

            G.edges[u, v]["bw_capacity"] = bw
            G.edges[u, v]["bw_remaining"] = bw
            G.edges[u, v]["delay"] = delay

            G.edges[u, v]["prop_coeff"] = float(self.rng.uniform(0.3, 1.0))

        return G

    def propagate_trust(self):

        new_scores = {}
        for node in self.graph.nodes():
            nbrs = list(self.graph.neighbors(node))
            if not nbrs:
                new_scores[node] = float(self.graph.nodes[node]["trust_score"])
                continue

            wsum, wtot = 0.0, 0.0
            for nb in nbrs:
                c = float(self.graph.edges[node, nb]["prop_coeff"])
                wsum += c * float(self.graph.nodes[nb]["trust_score"])
                wtot += c
            new_scores[node] = (wsum / wtot) if wtot > 1e-12 else float(self.graph.nodes[node]["trust_score"])

        for n, s in new_scores.items():
            self.graph.nodes[n]["trust_score"] = float(np.clip(s, 0.0, 1.0))

    def reset(self):
        for i in self.graph.nodes():
            n = self.graph.nodes[i]
            n["cpu_remaining"] = n["cpu_capacity"]
            n["mem_remaining"] = n["mem_capacity"]

        for u, v in self.graph.edges():
            e = self.graph.edges[u, v]
            e["bw_remaining"] = e["bw_capacity"]

        if hasattr(self, "_trust_base") and self._trust_base is not None:
            for i in self.graph.nodes():
                # node ids are 0...N-1 in this env
                self.graph.nodes[i]["trust_score"] = float(self._trust_base[int(i)])

        self.req_delay_used = 0.0
        self.reset_episode_stats()

        return self.get_node_features()

    def reset_request_budget(self):
        self.req_delay_used = 0.0

    def _trust_on(self, enforce_trust: bool, episode_idx: int):
        return bool(enforce_trust) and (episode_idx >= int(self.config.TRUST_ENFORCE_WARMUP_EP))

    def _path_bw_feasible(self, path, bw_req: float):
        if path is None or len(path) < 2:
            return False
        for a, b in zip(path[:-1], path[1:]):
            if float(self.graph.edges[a, b]["bw_remaining"]) < float(bw_req):
                return False
        return True

    def _path_delay(self, path):
        if path is None or len(path) < 2:
            return 0.0
        d = 0.0
        for a, b in zip(path[:-1], path[1:]):
            d += float(self.graph.edges[a, b]["delay"])
        return float(d)

    def _delay_shortest(self, src: int, dst: int):

        if self._delay_path is not None and self._delay_dist is not None:
            try:
                return self._delay_path[src][dst], float(self._delay_dist[src][dst])
            except Exception:
                return None, float("inf")
        try:
            path = nx.shortest_path(self.graph, source=src, target=dst, weight="delay")
            dist = nx.shortest_path_length(self.graph, source=src, target=dst, weight="delay")
            return path, float(dist)
        except Exception:
            return None, float("inf")

    def get_node_features(self):
        feats = []
        for i in self.graph.nodes():
            n = self.graph.nodes[i]
            feats.append([
                float(n["cpu_remaining"]) / max(1e-6, float(n["cpu_capacity"])),
                float(n["mem_remaining"]) / max(1e-6, float(n["mem_capacity"])),
                float(n["trust_score"]),
            ])
        return np.asarray(feats, dtype=np.float32)

    def get_action_mask(
        self,
        vnf: dict,
        last_node: int,
        enforce_trust: bool,
        episode_idx: int,
        sfc_bw_req: float,
        max_delay_tol: float,
    ):

        num_nodes = int(self.config.NUM_NODES)
        cpu_req = float(vnf["cpu_req"])
        mem_req = float(vnf["mem_req"])
        bw_req = float(sfc_bw_req)

        res_mask = np.ones((num_nodes,), dtype=bool)
        for i in range(num_nodes):
            n = self.graph.nodes[i]
            if float(n["cpu_remaining"]) < cpu_req or float(n["mem_remaining"]) < mem_req:
                res_mask[i] = False

        if not res_mask.any():
            return res_mask

        link_mask = np.zeros((num_nodes,), dtype=bool)

        if last_node is None or int(last_node) < 0:
            link_mask[res_mask] = True
        else:
            src = int(last_node)


            H = nx.Graph()
            H.add_nodes_from(self.graph.nodes())
            for u, v in self.graph.edges():
                if float(self.graph.edges[u, v]["bw_remaining"]) >= bw_req:
                    H.add_edge(u, v, delay=float(self.graph.edges[u, v]["delay"]))

            try:
                dist = nx.single_source_dijkstra_path_length(H, src, weight="delay")
            except Exception:
                dist = {}

            for i in range(num_nodes):
                if not res_mask[i]:
                    continue
                if i == src:
                    # staying at same node has zero added delay
                    if float(self.req_delay_used) <= float(max_delay_tol):
                        link_mask[i] = True
                    continue

                d = float(dist.get(i, float("inf")))
                if np.isfinite(d) and (float(self.req_delay_used) + d) <= float(max_delay_tol):
                    link_mask[i] = True

        if not link_mask.any():
            return link_mask

        if self._trust_on(enforce_trust, episode_idx):
            thr = float(getattr(self.config, "TRUST_THRESHOLD_FINAL", 0.7))
            trust_mask = np.zeros((num_nodes,), dtype=bool)
            for i in range(num_nodes):
                if link_mask[i] and float(self.graph.nodes[i]["trust_score"]) >= thr:
                    trust_mask[i] = True
            return trust_mask

        return link_mask

    def step(
        self,
        action: int,
        vnf: dict,
        bw_req: float,
        last_node: int,
        enforce_trust: bool,
        episode_idx: int,
        max_delay_tol: float,
        trust_threshold: float,
    ):

        info = {"success": False, "fail_reason": None, "delay_added": 0.0}

        a = int(action)
        cpu_req = float(vnf["cpu_req"])
        mem_req = float(vnf["mem_req"])
        bw_req = float(bw_req)

        self.stats["total_reqs"] += 1

        # resource check
        n = self.graph.nodes[a]
        if float(n["cpu_remaining"]) < cpu_req:
            self.stats["fails"]["cpu"] += 1
            info["fail_reason"] = "cpu"
            return self.get_node_features(), -3.5, False, info
        if float(n["mem_remaining"]) < mem_req:
            self.stats["fails"]["mem"] += 1
            info["fail_reason"] = "mem"
            return self.get_node_features(), -3.5, False, info

        # trust check (hard)
        if self._trust_on(enforce_trust, episode_idx):
            if float(n["trust_score"]) < float(trust_threshold):
                self.stats["fails"]["trust"] += 1
                info["fail_reason"] = "trust"
                return self.get_node_features(), -3.5, False, info

        # path check + delay update
        delay_added = 0.0
        if last_node is not None and int(last_node) >= 0:
            src = int(last_node)
            if src != a:
                path, d = self._delay_shortest(src, a)
                if path is None or not np.isfinite(d):
                    self.stats["fails"]["no_path"] += 1
                    info["fail_reason"] = "no_path"
                    return self.get_node_features(), -3.5, False, info
                if not self._path_bw_feasible(path, bw_req):
                    self.stats["fails"]["bw"] += 1
                    info["fail_reason"] = "bw"
                    return self.get_node_features(), -3.5, False, info
                if (float(self.req_delay_used) + float(d)) > float(max_delay_tol):
                    self.stats["fails"]["delay"] += 1
                    info["fail_reason"] = "delay"
                    return self.get_node_features(), -3.5, False, info

                # consume bw along path
                for u, v in zip(path[:-1], path[1:]):
                    self.graph.edges[u, v]["bw_remaining"] = float(self.graph.edges[u, v]["bw_remaining"]) - bw_req

                delay_added = float(d)
                self.req_delay_used = float(self.req_delay_used) + delay_added

        # commit resource consumption
        n["cpu_remaining"] = float(n["cpu_remaining"]) - cpu_req
        n["mem_remaining"] = float(n["mem_remaining"]) - mem_req

        info["success"] = True
        info["delay_added"] = delay_added

        self.stats["succ_reqs"] += 1
        self.stats["total_delay"] += float(self.req_delay_used)

        # # reward: succeed -> positive; shorter delay better
        # reward = 1.0
        # reward += float(getattr(self.config, "DELAY_REWARD_LAMBDA", -0.01)) * float(delay_added)
        #
        # return self.get_node_features(), float(reward), False, info
        # reward: succeed -> positive; shorter delay better

        success_bonus = 3.5


        if delay_added < 5.0:
            delay_penalty = -0.005 * float(delay_added)  # 短路径几乎不扣分
        else:
            delay_penalty = -0.05 * float(delay_added)  # 长路径重拳出击

        reward = success_bonus + delay_penalty
        # ======================================

        return self.get_node_features(), float(reward), False, info
import os


class Config:

    NUM_NODES = 120
    CPU_RANGE = (20, 100)
    MEM_RANGE = (20, 100)
    BW_RANGE = (50, 200)
    DELAY_RANGE = (1.0, 10.0)


    TRUST_THRESHOLD = 0.72
    TRUST_VARIANT = "A1"  # "A1" or "A2"
    TRUST_FALLBACK_TRAIN = False
    TRUST_FALLBACK_EVAL = False
    TRUST_EMPTY_EXTRA_PENALTY = 0.5
    TRUST_SOFT_SHAPING_LAMBDA = 0.5

    TRUST_ENFORCE_WARMUP_EP = 30
    TRUST_ENFORCE_RAMP_EP = 300

    TRUST_RELAX_TOPK = 3
    TRUST_RELAX_PENALTY_LAMBDA = 2.0

    BEST_GATE_ACC = 0.85
    BEST_GATE_TRUST_EMPTY = 0.05


    VNF_NUM_RANGE = (5, 9)  # (原本是 3~7)

    SFC_BW_REQ_RANGE = (10, 50)  # (原本是 1~30)


    SFC_MAX_DELAY_TOL = 25.0  # (原本是 60.0)

    VNF_CPU_REQ_RANGE = (1, 15)
    VNF_MEM_REQ_RANGE = (1, 15)


    R_STEP_BASE = 0.8
    R_TRUST_W = 1.2
    R_DELAY_W = 0.02
    R_BW_W = 0.001
    SUCCESS_BONUS = 5.0

    # ==========================================
    #  PPO 与 GNN 超参数
    # ==========================================
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    PPO_CLIP = 0.2
    PPO_EPOCHS = 4
    MINIBATCH_SIZE = 256
    LR = 3e-4
    ENTROPY_COEF = 0.03
    VALUE_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    TARGET_KL = 0.02

    GNN_HIDDEN_DIM = 128
    GNN_OUT_DIM = 128

    TRUST_THRESHOLD_START = 0.60
    TRUST_THRESHOLD_ANNEAL = True
    FAST_PATH = True

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
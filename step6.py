import json
import random
import os
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import pandas as pd


# =========================
# Config
# =========================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

INPUT_JSONL = "ipd_dpo/output/ipd_dpo_merged.jsonl"
CHECKPOINT = "output/step5_grpo_ckpt/policy_final.pt"

OUT_DIR = Path("output/step6")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GAMMA = 0.96


# =========================
# Payoff
# =========================
def payoff(agent_action, opp_action):
    if agent_action == "C" and opp_action == "C":
        return 3.0
    if agent_action == "C" and opp_action == "D":
        return 0.0
    if agent_action == "D" and opp_action == "C":
        return 5.0
    if agent_action == "D" and opp_action == "D":
        return 1.0
    raise ValueError(f"Invalid actions: {agent_action}, {opp_action}")


# =========================
# Opponent policies
# =========================
def always_cooperate(history):
    return "C"


def always_defect(history):
    return "D"


def tit_for_tat(history):
    if len(history) == 0:
        return "C"
    return history[-1][0]


def grim_trigger(history):
    for agent_a, _ in history:
        if agent_a == "D":
            return "D"
    return "C"


def alternating(history):
    return "C" if len(history) % 2 == 0 else "D"


def get_opponent_fn(name):
    n = name.lower()
    if "always cooperate" in n:
        return always_cooperate
    if "always defect" in n:
        return always_defect
    if "tit-for-tat" in n or "tit for tat" in n:
        return tit_for_tat
    if "grim" in n:
        return grim_trigger
    if "alternating" in n:
        return alternating
    raise ValueError(f"Unknown opponent: {name}")


# =========================
# Encoder / model
# =========================
OPPONENTS = [
    "Always Cooperate",
    "Always Defect",
    "Tit-for-Tat",
    "Grim Trigger",
    "Alternating",
]
OPP_TO_IDX = {name: i for i, name in enumerate(OPPONENTS)}


def normalize_opponent_name(name):
    n = name.lower()
    if "always cooperate" in n:
        return "Always Cooperate"
    if "always defect" in n:
        return "Always Defect"
    if "tit-for-tat" in n or "tit for tat" in n:
        return "Tit-for-Tat"
    if "grim" in n:
        return "Grim Trigger"
    if "alternating" in n:
        return "Alternating"
    return name


def predict_next_opponent_action(meta):
    opp_fn = get_opponent_fn(meta["opponent"])
    return opp_fn(meta["history"])


def encode_state(meta, mode="adaptive", fixed_assumption="Always Cooperate"):
    """
    IMPORTANT:
    Must match Step5 input dimension exactly: 13
    """
    actual_opponent = normalize_opponent_name(meta["opponent"])
    history = meta["history"]
    round_idx = meta["round_idx"]
    total_rounds = meta["total_rounds"]
    rounds_remaining = total_rounds - round_idx + 1

    if mode == "no_opponent":
        opponent_for_encoding = None
    elif mode == "fixed":
        opponent_for_encoding = fixed_assumption
    else:
        opponent_for_encoding = actual_opponent

    opp_vec = [0.0] * len(OPPONENTS)
    if opponent_for_encoding in OPP_TO_IDX:
        opp_vec[OPP_TO_IDX[opponent_for_encoding]] = 1.0

    hist_len = len(history)
    coop_count_agent = sum(1 for a, _ in history if a == "C")
    defect_count_agent = sum(1 for a, _ in history if a == "D")
    coop_count_opp = sum(1 for _, o in history if o == "C")
    defect_count_opp = sum(1 for _, o in history if o == "D")

    last_agent = 0.0 if hist_len == 0 else (1.0 if history[-1][0] == "C" else 0.0)
    last_opp = 0.0 if hist_len == 0 else (1.0 if history[-1][1] == "C" else 0.0)

    feat = []
    feat.extend(opp_vec)
    feat.extend([
        round_idx / total_rounds,
        rounds_remaining / total_rounds,
        coop_count_agent / max(1, hist_len),
        defect_count_agent / max(1, hist_len),
        coop_count_opp / max(1, hist_len),
        defect_count_opp / max(1, hist_len),
        last_agent,
        last_opp,
    ])

    return torch.tensor(feat, dtype=torch.float32)


class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


IDX_TO_ACTION = {0: "C", 1: "D"}


def choose_action(model, meta, mode, fixed_assumption="Always Cooperate"):
    feat = encode_state(meta, mode=mode, fixed_assumption=fixed_assumption)
    logits = model(feat.unsqueeze(0)).squeeze(0)
    idx = int(torch.argmax(logits).item())
    return IDX_TO_ACTION[idx]


def rollout_episode(model, meta, mode, fixed_assumption="Always Cooperate"):
    history = [tuple(x) for x in meta["history"]]
    actual_opponent = meta["opponent"]
    total_rounds = meta["total_rounds"]
    round_idx = meta["round_idx"]
    rounds_remaining = total_rounds - round_idx + 1
    opp_fn = get_opponent_fn(actual_opponent)

    total_return = 0.0
    discount = 1.0

    for t in range(rounds_remaining):
        current_meta = {
            "opponent": actual_opponent,
            "round_idx": round_idx + t,
            "total_rounds": total_rounds,
            "history": history,
        }

        # Level 3 is interpreted as explicit reasoning outside the NN input
        if mode == "recursive":
            predicted_next_opp = opp_fn(history)
            _ = predicted_next_opp  # reasoning step for analysis/debug
            action = choose_action(model, current_meta, mode="adaptive")
        else:
            action = choose_action(
                model,
                current_meta,
                mode=mode,
                fixed_assumption=fixed_assumption
            )

        opp_action = opp_fn(history)
        r = payoff(action, opp_action)
        total_return += discount * r
        discount *= GAMMA
        history.append((action, opp_action))

    return total_return


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            data.append(json.loads(line))
            if idx % 200 == 0 and idx > 0:
                print(f"loaded {idx} samples so far...")
    return data


def main():
    print("===== STEP6 PYTHON START =====")
    print("cwd:", os.getcwd())
    print("INPUT_JSONL exists:", Path(INPUT_JSONL).exists(), INPUT_JSONL)
    print("CHECKPOINT exists:", Path(CHECKPOINT).exists(), CHECKPOINT)
    print("OUT_DIR:", OUT_DIR.resolve())

    print("\nloading dataset...")
    dataset = load_jsonl(INPUT_JSONL)
    print("dataset size:", len(dataset))

    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    print("\nbuilding model...")
    example_meta = dataset[0]["metadata"]
    input_dim = len(encode_state(example_meta, mode="adaptive"))
    print("input_dim:", input_dim)

    model = PolicyNet(input_dim=input_dim, hidden_dim=64)

    print("\nloading checkpoint...")
    state_dict = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("checkpoint loaded successfully")

    levels = [
        ("Level 0", "no_opponent", None),
        ("Level 1", "fixed", "Always Cooperate"),
        ("Level 2", "adaptive", None),
        ("Level 3", "recursive", None),
    ]

    rows = []

    for level_name, mode, fixed_assumption in levels:
        print("\n==============================")
        print(f"Running {level_name}")
        print("mode:", mode)
        print("fixed_assumption:", fixed_assumption)
        print("==============================")

        returns_by_opp = defaultdict(list)

        for i, sample in enumerate(dataset):
            if i % 50 == 0:
                print(f"{level_name} progress: {i}/{len(dataset)}")

            meta = sample["metadata"]

            ret = rollout_episode(
                model,
                meta,
                mode=mode,
                fixed_assumption=fixed_assumption if fixed_assumption else "Always Cooperate"
            )

            opp_name = normalize_opponent_name(meta["opponent"])
            returns_by_opp[opp_name].append(ret)

        print(f"{level_name} finished rollout")

        for opp, vals in returns_by_opp.items():
            mean_val = sum(vals) / len(vals)
            print(f"{level_name} | {opp} | mean_return={mean_val:.4f} | n={len(vals)}")
            rows.append({
                "level": level_name,
                "opponent": opp,
                "mean_return": mean_val,
                "num_samples": len(vals),
            })

        overall = []
        for vals in returns_by_opp.values():
            overall.extend(vals)

        overall_mean = sum(overall) / len(overall)
        print(f"{level_name} | OVERALL | mean_return={overall_mean:.4f} | n={len(overall)}")

        rows.append({
            "level": level_name,
            "opponent": "OVERALL",
            "mean_return": overall_mean,
            "num_samples": len(overall),
        })

    print("\ncreating dataframe...")
    df = pd.DataFrame(rows)
    print(df.head())

    csv_path = OUT_DIR / "level_results.csv"
    json_path = OUT_DIR / "level_results.json"

    print("\nsaving results...")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print("\nSaved files:")
    print(csv_path)
    print(json_path)

    print("\n===== FINAL RESULTS =====")
    print(df)

    print("\n===== STEP6 PYTHON END =====")


if __name__ == "__main__":
    main()
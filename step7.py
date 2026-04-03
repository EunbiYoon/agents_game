import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd


# =========================
# Config
# =========================
INPUT_JSONL = "ipd_dpo/output/ipd_dpo_merged.jsonl"
CHECKPOINT = "output/step5_grpo_ckpt/policy_final.pt"

OUT_DIR = Path("output/step7")
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
# Model
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


def encode_state(meta):
    """
    IMPORTANT:
    Must match Step5 exactly -> 13 dims
    """
    opponent = normalize_opponent_name(meta["opponent"])
    history = meta["history"]
    round_idx = meta["round_idx"]
    total_rounds = meta["total_rounds"]
    rounds_remaining = total_rounds - round_idx + 1

    opp_vec = [0.0] * len(OPPONENTS)
    if opponent in OPP_TO_IDX:
        opp_vec[OPP_TO_IDX[opponent]] = 1.0

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


def choose_action(model, meta):
    feat = encode_state(meta)
    logits = model(feat.unsqueeze(0)).squeeze(0)
    idx = int(torch.argmax(logits).item())
    return IDX_TO_ACTION[idx]


def prompt_based_reasoning(model, meta):
    # history/opponent info is already included in state
    action = choose_action(model, meta)
    return action, None


def explicit_opponent_simulation(model, meta):
    # Explicitly predict opponent next move first
    opp_fn = get_opponent_fn(meta["opponent"])
    predicted_next_opp = opp_fn(meta["history"])

    # Simple best-response rule using predicted next action
    if predicted_next_opp == "C":
        action = "D"   # exploit cooperation
    else:
        action = "D"   # safest response to defection in one-step payoff

    return action, predicted_next_opp


def one_step_reward(meta, action):
    opp_fn = get_opponent_fn(meta["opponent"])
    opp_action = opp_fn(meta["history"])
    return payoff(action, opp_action), opp_action


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            data.append(json.loads(line))
            if i % 200 == 0 and i > 0:
                print(f"loaded {i} samples so far...")
    return data


def main():
    print("===== STEP7 PYTHON START =====")
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
    input_dim = len(encode_state(example_meta))
    print("input_dim:", input_dim)

    model = PolicyNet(input_dim=input_dim, hidden_dim=64)

    print("\nloading checkpoint...")
    state_dict = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("checkpoint loaded successfully")

    rows = []
    examples = []

    for i, sample in enumerate(dataset):
        if i % 50 == 0:
            print(f"STEP7 progress: {i}/{len(dataset)}")

        meta = sample["metadata"]
        opponent = normalize_opponent_name(meta["opponent"])

        action_a, pred_a = prompt_based_reasoning(model, meta)
        reward_a, actual_opp_a = one_step_reward(meta, action_a)

        action_b, pred_b = explicit_opponent_simulation(model, meta)
        reward_b, actual_opp_b = one_step_reward(meta, action_b)

        rows.append({
            "sample_idx": i,
            "opponent": opponent,
            "strategy": "Prompt-Based Opponent Reasoning",
            "chosen_action": action_a,
            "predicted_opponent_action": pred_a,
            "actual_opponent_action": actual_opp_a,
            "reward": reward_a,
        })
        rows.append({
            "sample_idx": i,
            "opponent": opponent,
            "strategy": "Explicit Opponent Simulation",
            "chosen_action": action_b,
            "predicted_opponent_action": pred_b,
            "actual_opponent_action": actual_opp_b,
            "reward": reward_b,
        })

        if i < 20:
            examples.append({
                "sample_idx": i,
                "opponent": opponent,
                "history": meta["history"],
                "prompt_based_action": action_a,
                "explicit_predicted_opponent_action": pred_b,
                "explicit_action": action_b,
            })

    df = pd.DataFrame(rows)

    results_path = OUT_DIR / "strategy_results.csv"
    summary_path = OUT_DIR / "strategy_summary.csv"
    examples_path = OUT_DIR / "inference_examples.json"

    df.to_csv(results_path, index=False)
    with open(examples_path, "w") as f:
        json.dump(examples, f, indent=2)

    summary = df.groupby(["strategy", "opponent"])["reward"].mean().reset_index()
    summary.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(results_path)
    print(summary_path)
    print(examples_path)

    print("\nSummary:")
    print(summary)
    print("\n===== STEP7 PYTHON END =====")


if __name__ == "__main__":
    main()
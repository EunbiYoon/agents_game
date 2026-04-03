import json
import math
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# Config
# =========================
INPUT_JSONL = "ipd_dpo/output/ipd_dpo_merged.jsonl"
SAVE_DIR = Path("output/step5_grpo_ckpt")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

GAMMA = 0.96
NUM_EPOCHS = 10
LR = 1e-3
NUM_ACTION_SAMPLES = 8     # sample multiple actions per state
ROLLOUTS_PER_ACTION = 4    # average multiple rollouts
HIDDEN_DIM = 64
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


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
    return history[-1][0]  # mirror agent's previous action


def grim_trigger(history):
    # Cooperate until agent defects once, then defect forever
    for agent_a, _ in history:
        if agent_a == "D":
            return "D"
    return "C"


def alternating(history):
    # simple alternating policy: C, D, C, D...
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
# Feature encoder
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


def encode_state(sample):
    meta = sample["metadata"]
    opponent = normalize_opponent_name(meta["opponent"])
    round_idx = meta["round_idx"]
    total_rounds = meta["total_rounds"]
    rounds_remaining = total_rounds - round_idx + 1
    history = meta["history"]

    # One-hot opponent
    opp_vec = [0.0] * len(OPPONENTS)
    if opponent in OPP_TO_IDX:
        opp_vec[OPP_TO_IDX[opponent]] = 1.0

    # Basic scalar features
    coop_count_agent = sum(1 for a, _ in history if a == "C")
    defect_count_agent = sum(1 for a, _ in history if a == "D")
    coop_count_opp = sum(1 for _, o in history if o == "C")
    defect_count_opp = sum(1 for _, o in history if o == "D")

    hist_len = len(history)
    if hist_len > 0:
        last_agent = 1.0 if history[-1][0] == "C" else 0.0
        last_opp = 1.0 if history[-1][1] == "C" else 0.0
    else:
        last_agent = 0.0
        last_opp = 0.0

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


# =========================
# Policy model
# Outputs logits for [C, D]
# =========================
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
ACTION_TO_IDX = {"C": 0, "D": 1}


# =========================
# Rollout
# =========================
@torch.no_grad()
def sample_action_from_policy(model, feat):
    logits = model(feat.unsqueeze(0)).squeeze(0)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    idx = dist.sample().item()
    logprob = torch.log(probs[idx] + 1e-8)
    return IDX_TO_ACTION[idx], logprob.item(), probs.tolist()


def simulate_rollout(model, history, opponent_name, first_action, rounds_remaining):
    """
    First action is fixed.
    Future actions are sampled from current policy.
    """
    opponent_fn = get_opponent_fn(opponent_name)
    h = [tuple(x) for x in history]
    total_return = 0.0
    discount = 1.0

    action = first_action

    for t in range(rounds_remaining):
        opp_action = opponent_fn(h)
        r = payoff(action, opp_action)
        total_return += discount * r
        h.append((action, opp_action))
        discount *= GAMMA

        # future action from policy, unless terminal
        if t < rounds_remaining - 1:
            fake_sample = {
                "metadata": {
                    "opponent": opponent_name,
                    "round_idx": len(h) + 1,
                    "total_rounds": len(h) + (rounds_remaining - t - 1),
                    "history": h,
                }
            }
            feat = encode_state(fake_sample)
            action, _, _ = sample_action_from_policy(model, feat)

    return total_return


@torch.no_grad()
def evaluate_candidate_action(model, sample, candidate_action):
    meta = sample["metadata"]
    opponent = meta["opponent"]
    history = meta["history"]
    rounds_remaining = meta["total_rounds"] - meta["round_idx"] + 1

    returns = []
    for _ in range(ROLLOUTS_PER_ACTION):
        ret = simulate_rollout(
            model=model,
            history=history,
            opponent_name=opponent,
            first_action=candidate_action,
            rounds_remaining=rounds_remaining,
        )
        returns.append(ret)

    return sum(returns) / len(returns)


# =========================
# Load data
# =========================
def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


dataset = load_jsonl(INPUT_JSONL)
random.shuffle(dataset)

input_dim = len(encode_state(dataset[0]))
model = PolicyNet(input_dim=input_dim, hidden_dim=HIDDEN_DIM)
optimizer = optim.Adam(model.parameters(), lr=LR)


# =========================
# GRPO-style training
# =========================
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    count = 0
    action_stats = defaultdict(int)

    random.shuffle(dataset)

    for sample in dataset:
        feat = encode_state(sample)

        # Step 1: Generate candidate actions by sampling from current policy
        sampled_actions = []
        sampled_logprobs = []

        logits = model(feat.unsqueeze(0)).squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)

        for _ in range(NUM_ACTION_SAMPLES):
            idx = dist.sample()
            sampled_actions.append(IDX_TO_ACTION[idx.item()])
            sampled_logprobs.append(torch.log(probs[idx] + 1e-8))

        # Ensure both actions are considered at least once
        if "C" not in sampled_actions:
            sampled_actions.append("C")
            sampled_logprobs.append(torch.log(probs[ACTION_TO_IDX["C"]] + 1e-8))
        if "D" not in sampled_actions:
            sampled_actions.append("D")
            sampled_logprobs.append(torch.log(probs[ACTION_TO_IDX["D"]] + 1e-8))

        # Step 2-3: rollout with opponent responses + compute rewards
        returns = []
        for a in sampled_actions:
            avg_ret = evaluate_candidate_action(model, sample, a)
            returns.append(avg_ret)

        returns_t = torch.tensor(returns, dtype=torch.float32)
        mean_r = returns_t.mean()
        std_r = returns_t.std(unbiased=False) + 1e-8

        # GRPO-style normalized advantages
        advantages = (returns_t - mean_r) / std_r

        # Step 4: update toward better actions
        loss = 0.0
        for logp, adv in zip(sampled_logprobs, advantages):
            loss = loss + (-logp * adv)

        loss = loss / len(sampled_logprobs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

        # track best action
        best_idx = int(torch.argmax(returns_t).item())
        best_action = sampled_actions[best_idx]
        action_stats[best_action] += 1

    avg_loss = total_loss / max(1, count)
    print(
        f"[Epoch {epoch+1}/{NUM_EPOCHS}] "
        f"avg_loss={avg_loss:.4f} "
        f"best_C={action_stats['C']} "
        f"best_D={action_stats['D']}"
    )

    ckpt_path = SAVE_DIR / f"policy_epoch{epoch+1}.pt"
    torch.save(model.state_dict(), ckpt_path)

final_path = SAVE_DIR / "policy_final.pt"
torch.save(model.state_dict(), final_path)
print(f"Saved final model to {final_path}")
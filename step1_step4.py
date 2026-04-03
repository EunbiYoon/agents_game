import os
import json
import random
import argparse
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "/work/pi_dagarwal_umass_edu/project_16/model/Qwen2.5-1.5B-Instruct"

PAYOFFS = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}

# Use the exact opponent names shown in the figure / config
OPPONENT_DESCRIPTIONS = {
    "Tit-for-Tat": "A reciprocal strategy. The agent cooperates if the opponent cooperates and defects if the opponent defects.",
    "Always Defect": "A fully competitive strategy. The agent always defects and never cooperates.",
    "Always Cooperate": "A fully cooperative strategy. The agent always cooperates regardless of the opponent.",
    "Alternating": "A non-stationary strategy. The agent alternates between cooperate and defect.",
    "Grim Trigger": "A strict long-term strategy. The agent cooperates until the opponent defects once, then defects forever.",
    "Learning-based adversary": "An adaptive strategy. The opponent changes behavior dynamically to exploit the agent."
}

ALL_OPPONENTS = list(OPPONENT_DESCRIPTIONS.keys())

# Optional aliases so old commands/configs still work
OPPONENT_NAME_MAP = {
    "tit_for_tat": "Tit-for-Tat",
    "always_defect": "Always Defect",
    "always_cooperate": "Always Cooperate",
    "alternating": "Alternating",
    "grim_trigger": "Grim Trigger",
    "learning_adversary": "Learning-based adversary",
    "Defect Once": "Grim Trigger",  # safe fallback if older config still uses this
}


@dataclass
class State:
    round_idx: int
    total_rounds: int
    history: List[Tuple[str, str]]
    opponent_name: str


class OpponentPolicy:
    def __init__(self, name: str):
        self.name = normalize_opponent_name(name)

    def act(self, history: List[Tuple[str, str]]) -> str:
        if self.name == "Tit-for-Tat":
            if len(history) == 0:
                return "C"
            return history[-1][0]

        if self.name == "Always Defect":
            return "D"

        if self.name == "Always Cooperate":
            return "C"

        if self.name == "Grim Trigger":
            if any(a == "D" for a, _ in history):
                return "D"
            return "C"

        if self.name == "Alternating":
            return "C" if len(history) % 2 == 0 else "D"

        if self.name == "Learning-based adversary":
            if len(history) == 0:
                return "C"

            recent = history[-3:]
            coop_rate = sum(1 for a, _ in recent if a == "C") / len(recent)
            defect_rate = sum(1 for a, _ in recent if a == "D") / len(recent)

            if coop_rate >= 0.67:
                return "D"
            if defect_rate >= 0.67:
                return "D"
            if history[-1][0] == "D":
                return "D"
            return "C"

        raise ValueError(f"Unknown opponent: {self.name}")


def normalize_opponent_name(name: str) -> str:
    if name in ALL_OPPONENTS:
        return name
    if name in OPPONENT_NAME_MAP:
        return OPPONENT_NAME_MAP[name]
    raise ValueError(
        f"Unknown opponent '{name}'. Valid options: {ALL_OPPONENTS} "
        f"or aliases: {list(OPPONENT_NAME_MAP.keys())}"
    )


def safe_filename(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def compute_payoff(agent_action: str, opp_action: str):
    return PAYOFFS[(agent_action, opp_action)]


def history_to_text(history: List[Tuple[str, str]]) -> str:
    if not history:
        return "No previous rounds."
    lines = []
    for i, (a, o) in enumerate(history, start=1):
        lines.append(f"Round {i}: Agent={a}, Opponent={o}")
    return "\n".join(lines)


def sample_state(opponent_name: str, total_rounds: int, rng: random.Random) -> State:
    opponent_name = normalize_opponent_name(opponent_name)
    opp = OpponentPolicy(opponent_name)
    history = []
    prefix_len = rng.randint(0, max(0, total_rounds - 2))

    for _ in range(prefix_len):
        agent_action = rng.choice(["C", "D"])
        opp_action = opp.act(history)
        history.append((agent_action, opp_action))

    return State(
        round_idx=len(history) + 1,
        total_rounds=total_rounds,
        history=history,
        opponent_name=opponent_name
    )


def rollout_value(
    state: State,
    first_action: str,
    gamma: float,
    rollout_horizon: int,
    rng: random.Random
) -> Dict:
    opp = OpponentPolicy(state.opponent_name)
    history = list(state.history)

    total_agent = 0.0
    total_opp = 0.0
    discount = 1.0

    opp_action = opp.act(history)
    a_pay, o_pay = compute_payoff(first_action, opp_action)
    total_agent += discount * a_pay
    total_opp += discount * o_pay
    history.append((first_action, opp_action))
    immediate_agent = a_pay
    discount *= gamma

    remaining_rounds = min(rollout_horizon, state.total_rounds - len(history))

    for _ in range(remaining_rounds):
        if len(history) == 0:
            agent_action = "C"
        else:
            last_opp = history[-1][1]
            p_coop = 0.65 if last_opp == "C" else 0.30
            agent_action = "C" if rng.random() < p_coop else "D"

        opp_action = opp.act(history)
        a_pay, o_pay = compute_payoff(agent_action, opp_action)
        total_agent += discount * a_pay
        total_opp += discount * o_pay
        history.append((agent_action, opp_action))
        discount *= gamma

    return {
        "immediate_agent": immediate_agent,
        "discounted_agent_return": round(total_agent, 4),
        "discounted_opp_return": round(total_opp, 4),
        "trajectory": history,
    }


def average_rollout_score(
    state: State,
    action: str,
    num_rollouts: int,
    gamma: float,
    rollout_horizon: int,
    rng: random.Random
) -> Dict:
    runs = []
    for _ in range(num_rollouts):
        local_rng = random.Random(rng.randint(0, 10**9))
        result = rollout_value(
            state=state,
            first_action=action,
            gamma=gamma,
            rollout_horizon=rollout_horizon,
            rng=local_rng
        )
        runs.append(result)

    avg_agent = sum(r["discounted_agent_return"] for r in runs) / len(runs)
    avg_opp = sum(r["discounted_opp_return"] for r in runs) / len(runs)
    avg_immediate = sum(r["immediate_agent"] for r in runs) / len(runs)

    return {
        "action": action,
        "avg_discounted_agent_return": round(avg_agent, 4),
        "avg_discounted_opp_return": round(avg_opp, 4),
        "avg_immediate_agent": round(avg_immediate, 4),
        "samples": runs,
    }


def build_prompt(state: State) -> str:
    opponent_desc = OPPONENT_DESCRIPTIONS[state.opponent_name]
    remaining = state.total_rounds - state.round_idx + 1

    return f"""You are playing an Iterated Prisoner's Dilemma game.

Opponent type:
{state.opponent_name}

Opponent description:
{opponent_desc}

Current round: {state.round_idx} / {state.total_rounds}
Rounds remaining including this one: {remaining}

History:
{history_to_text(state.history)}

Choose exactly one action for this round:
- C = Cooperate
- D = Defect

Respond in this format:
Action: <C or D>
Reason: <brief strategic explanation>""".strip()


def build_forced_reason_prompt(state: State, forced_action: str) -> str:
    opponent_desc = OPPONENT_DESCRIPTIONS[state.opponent_name]

    return f"""You are analyzing an Iterated Prisoner's Dilemma decision.

Opponent type:
{state.opponent_name}

Opponent description:
{opponent_desc}

Current round: {state.round_idx} / {state.total_rounds}

History:
{history_to_text(state.history)}

You must justify the following action for this round:
Action: {forced_action}

Respond exactly in this format:
Action: {forced_action}
Reason: <brief strategic explanation>""".strip()


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model


def generate_text(tokenizer, model, prompt: str, max_new_tokens: int = 80) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if decoded.startswith(text):
        return decoded[len(text):].strip()
    return decoded.strip()


def templated_response(action: str, opponent_name: str) -> str:
    if action == "C":
        reason = f"Cooperation can preserve future gains if reciprocity is possible against {opponent_name}."
    else:
        reason = f"Defection improves short-term protection or exploitation value against {opponent_name}."
    return f"Action: {action}\nReason: {reason}"


def make_pair_record(
    state: State,
    chosen_action: str,
    rejected_action: str,
    chosen_response: str,
    rejected_response: str,
    chosen_score: Dict,
    rejected_score: Dict
) -> Dict:
    return {
        "prompt": build_prompt(state),
        "chosen": chosen_response,
        "rejected": rejected_response,
        "metadata": {
            "game": "iterated_prisoners_dilemma",
            "opponent": state.opponent_name,
            "round_idx": state.round_idx,
            "total_rounds": state.total_rounds,
            "history": state.history,

            "step1_outcome_based_pairing": {
                "chosen_action": chosen_action,
                "rejected_action": rejected_action,
                "chosen_immediate_agent": chosen_score["avg_immediate_agent"],
                "rejected_immediate_agent": rejected_score["avg_immediate_agent"]
            },

            "step2_equilibrium_guided_labeling": {
                "chosen_discounted_agent_return": chosen_score["avg_discounted_agent_return"],
                "rejected_discounted_agent_return": rejected_score["avg_discounted_agent_return"],
                "label_rule": "Choose the action with higher discounted return against the specified opponent."
            },

            "step3_self_play_ranking": {
                "chosen_avg_rank_score": chosen_score["avg_discounted_agent_return"],
                "rejected_avg_rank_score": rejected_score["avg_discounted_agent_return"],
                "ranking_source": "Monte Carlo rollout ranking from the same state."
            },

            "step4_opponent_conditioned_contrastive_pair": {
                "opponent_name": state.opponent_name,
                "opponent_description": OPPONENT_DESCRIPTIONS[state.opponent_name],
                "same_state_different_reasoning": True
            }
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", type=str, required=True)
    parser.add_argument("--num_pairs", type=int, default=150)
    parser.add_argument("--total_rounds", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--rollout_horizon", type=int, default=6)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_model_rationales", action="store_true")
    args = parser.parse_args()

    args.opponent = normalize_opponent_name(args.opponent)

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)

    tokenizer, model = (None, None)
    if args.use_model_rationales:
        tokenizer, model = load_model()

    out_path = os.path.join(
        args.output_dir,
        f"ipd_dpo_{safe_filename(args.opponent)}_{args.num_pairs}.jsonl"
    )

    written = 0
    attempts = 0
    max_attempts = args.num_pairs * 5

    while written < args.num_pairs and attempts < max_attempts:
        attempts += 1

        state = sample_state(
            opponent_name=args.opponent,
            total_rounds=args.total_rounds,
            rng=rng
        )

        score_c = average_rollout_score(
            state=state,
            action="C",
            num_rollouts=args.num_rollouts,
            gamma=args.gamma,
            rollout_horizon=args.rollout_horizon,
            rng=rng
        )
        score_d = average_rollout_score(
            state=state,
            action="D",
            num_rollouts=args.num_rollouts,
            gamma=args.gamma,
            rollout_horizon=args.rollout_horizon,
            rng=rng
        )

        margin = abs(
            score_c["avg_discounted_agent_return"] -
            score_d["avg_discounted_agent_return"]
        )

        if margin < 0.15:
            continue

        if score_c["avg_discounted_agent_return"] > score_d["avg_discounted_agent_return"]:
            chosen_action, rejected_action = "C", "D"
            chosen_score, rejected_score = score_c, score_d
        else:
            chosen_action, rejected_action = "D", "C"
            chosen_score, rejected_score = score_d, score_c

        if args.use_model_rationales:
            chosen_response = generate_text(
                tokenizer, model, build_forced_reason_prompt(state, chosen_action)
            )
            rejected_response = generate_text(
                tokenizer, model, build_forced_reason_prompt(state, rejected_action)
            )

            if f"Action: {chosen_action}" not in chosen_response:
                chosen_response = f"Action: {chosen_action}\nReason: This action is strategically stronger in this state."
            if f"Action: {rejected_action}" not in rejected_response:
                rejected_response = f"Action: {rejected_action}\nReason: This action is strategically weaker in this state."
        else:
            chosen_response = templated_response(chosen_action, state.opponent_name)
            rejected_response = templated_response(rejected_action, state.opponent_name)

        record = make_pair_record(
            state=state,
            chosen_action=chosen_action,
            rejected_action=rejected_action,
            chosen_response=chosen_response,
            rejected_response=rejected_response,
            chosen_score=chosen_score,
            rejected_score=rejected_score
        )

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        written += 1
        if written % 10 == 0:
            print(f"[{args.opponent}] written {written}/{args.num_pairs}")

    print(f"Finished: {written} pairs saved to {out_path}")


if __name__ == "__main__":
    main()
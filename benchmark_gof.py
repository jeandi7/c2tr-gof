# benchmark_gof.py
#
# CIITR Benchmark: LLM alone vs LLM + Active Inference on GoF class diagram generation.
#
# Mode LLM_ONLY : objective -> prompt -> LLM -> JSON -> validate -> PlantUML -> Phi/E
# Mode LLM_AI   : AI detects pattern -> enriched prompt -> LLM -> JSON -> validate -> PlantUML -> Phi/E
#
# Hypothesis: (Phi/E)_AI >= (Phi/E)_LLM  AND  Rg > 0
# CPJ (Ci/E = Phi*Rg/E) is kept in CSV for reference but NOT used for comparison:
# Ci = Phi * Rg = 0 for LLM_ONLY by definition (frozen weights => Rg=0).
#
# Usage:
#   python benchmark_gof.py --session GOF-001
#   python benchmark_gof.py --session GOF-001 --remote
#   python benchmark_gof.py --no-ai
#   python benchmark_gof.py

import argparse
import json
import os
import re
from datetime import datetime

import yaml

from agent_ai.gof_ai_agent import GofAIAgent, TEXT_KEYWORDS
from agent_llm.llm import call_llm, MLX_MODEL_ID
from gof.generator import build_prompt
from gof.metamodel import validate_gof, compute_comprehension, compute_pi
from gof.plantuml import to_plantuml
from liss import LISS
from metrics.export import init_gof_csv, append_gof_row
from psis import PSIS

CPU_WATTS = 28.0   # generic CPU (Ollama local)
M3_WATTS  = 12.0   # Apple M-series Neural Engine + GPU (MLX)
MAX_ITER = 3
SESSIONS_FILE = "sessions_gof.yaml"

_liss = LISS.default()
_KEYWORD_SET = set(TEXT_KEYWORDS)


def load_sessions() -> list[dict]:
    with open(SESSIONS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [t for t in tokens if t in _KEYWORD_SET]


def extract_json(raw: str) -> str:
    start = None
    stack = []
    for i, c in enumerate(raw):
        if c in "{[":
            if start is None:
                start = i
            stack.append(c)
        elif c in "}]":
            if not stack:
                continue
            stack.pop()
            if not stack:
                return raw[start:i + 1]
    raise ValueError("No complete JSON found in LLM response")


def _call_and_parse(liss: LISS, psis: PSIS, local: bool, use_mlx: bool = False):
    """Returns (model_or_None, energy_j, errors, raw, plantuml)."""
    prompt = build_prompt(liss, psis)
    raw, elapsed = call_llm(prompt, local=local, use_mlx=use_mlx)
    energy = elapsed * (M3_WATTS if use_mlx else CPU_WATTS)

    if not raw:
        return None, energy, ["Empty LLM response"], raw, ""

    if raw.count("{") != raw.count("}"):
        return None, energy, ["Unbalanced braces — incomplete JSON"], raw, ""

    try:
        raw_json = extract_json(raw)
        # Fix missing commas between fields: "value"\n  "nextkey" → "value",\n  "nextkey"
        raw_json = re.sub(r'"([ \t]*\n[ \t]*)"', r'",\1"', raw_json)
        model = json.loads(raw_json)
    except ValueError as exc:
        return None, energy, [f"JSON parse error: {exc}"], raw, ""

    errors = validate_gof(model)
    if errors:
        return None, energy, errors, raw, ""

    return model, energy, [], raw, to_plantuml(model)


def _pattern_name(ai_context: dict | None, model: dict | None) -> str:
    if model:
        patterns = model.get("patterns")
        if isinstance(patterns, list) and len(patterns) > 1:
            return "+".join(patterns)
        p = model.get("pattern", "")
        if p:
            return p
    if ai_context:
        return ai_context["pattern_name"]
    return ""


def _save_raw(raw_dir: str, session_id: str, mode: str, iteration: int, raw: str):
    path = os.path.join(raw_dir, f"{session_id}_{mode}_iter{iteration}.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(raw or "")


def _run_iteration(
    liss: LISS,
    psis: PSIS,
    ai_agent: GofAIAgent | None,
    use_ai: bool,
    local: bool,
    csv_path: str,
    raw_dir: str,
    warmup: bool = False,
    use_mlx: bool = False,
) -> tuple[float, float, str, float, float, float]:
    """Run one iteration. Returns (phi_enr_e, energy_j, plantuml, comprehension, cpj, pi)."""
    mode = "LLM_AI" if psis.ai_context else "LLM_ONLY"
    model, energy, errors, raw, plantuml = _call_and_parse(liss, psis, local, use_mlx)

    _save_raw(raw_dir, psis.session_id, mode, psis.iteration, raw)

    pattern = _pattern_name(psis.ai_context, model)
    warmup_tag = " [warmup]" if warmup else ""

    if model is None:
        rg = ai_agent.rg_proxy if (use_ai and ai_agent) else 0.0
        append_gof_row(csv_path, psis.session_id, mode, psis.iteration,
                       False, energy, 0, 0.0, 0.0, 0.0, rg, pattern, errors, warmup=warmup)
        print(f"    iter {psis.iteration}: REJECTED{warmup_tag}  ({errors[0] if errors else 'unknown'})")
        return 0.0, energy, "", 0, 0.0, 0.0

    # Feed LLM JSON back to agent — CIITR continuous learning
    if use_ai and ai_agent:
        tokens = re.findall(r"[a-zA-Z]+", json.dumps(model).lower())
        learned = sum(ai_agent.learn_keyword(tok) for tok in tokens)
        for tok in tokens:
            ai_agent.observe_keyword(tok)
        vocab_size = len(ai_agent._keyword_index)
        print(f"    agent vocab: {vocab_size} keywords (+{learned} new this iter)")

    rg = ai_agent.rg_proxy if (use_ai and ai_agent) else 0.0
    comprehension = compute_comprehension(model)
    phi_e        = comprehension / (energy + 1e-12)
    ci           = comprehension * rg
    cpj          = ci / (energy + 1e-12)
    pi           = compute_pi(model, pattern)
    phi_enriched = comprehension * pi
    phi_enr_e    = phi_enriched / (energy + 1e-12)
    append_gof_row(csv_path, psis.session_id, mode, psis.iteration,
                   True, energy, comprehension, phi_e, ci, cpj, rg, pattern, [], warmup=warmup)
    print(f"    iter {psis.iteration}: ACCEPTED{warmup_tag}  E={energy:.3f}J  "
          f"Phi={comprehension:.2f}  Ci={ci:.2f}  Pi={pi:.2f}  Phi_enr={phi_enriched:.2f}  "
          f"(Phi_enr/E)={phi_enr_e*1000:.4f}/kJ  (Phi/E)={phi_e*1000:.4f}/kJ  "
          f"Rg={rg:.4f}  pattern={pattern}")
    return phi_enr_e, energy, plantuml, comprehension, cpj, pi


def _init_agent(ai_agent: GofAIAgent | None, objective: str) -> None:
    if ai_agent:
        ai_agent.reset()
        for kw in extract_keywords(objective):
            ai_agent.observe_keyword(kw)


def _run_mode(
    session_data: dict,
    ai_agent: GofAIAgent | None,
    local: bool,
    use_ai: bool,
    csv_path: str,
    mode: str,
    raw_dir: str,
    warmup_iters: int = 1,
    max_iter: int = MAX_ITER,
    use_mlx: bool = False,
):
    objective = session_data["objective"].strip()
    _init_agent(ai_agent, objective)

    best_score = 0.0
    best_pi = 0.0
    best_plantuml = ""
    best_iter = 0
    total_energy = 0.0
    n_accepted = 0
    last_cpj = 0.0
    last_comprehension = 0.0

    print(f"\n  [{mode}]")

    for iteration in range(1, max_iter + 1):
        is_warmup = iteration <= warmup_iters
        ai_context = ai_agent.get_pattern_context() if (use_ai and ai_agent) else None

        psis = PSIS(
            session_id=session_data["session_id"],
            objective=objective,
            iteration=iteration,
            last_cpj=last_cpj,
            last_comprehension=last_comprehension,
            ai_context=ai_context,
        )

        phi_e, energy, plantuml, comprehension, cpj, pi = _run_iteration(
            _liss, psis, ai_agent, use_ai, local, csv_path, raw_dir, warmup=is_warmup, use_mlx=use_mlx,
        )

        if comprehension > 0:
            last_cpj = cpj
            last_comprehension = comprehension

        if not is_warmup:
            total_energy += energy
            if comprehension > 0:
                n_accepted += 1
            if phi_e > best_score:
                best_score = phi_e
                best_plantuml = plantuml
                best_iter = iteration
                best_pi = pi

    status = "ACCEPTED" if n_accepted > 0 else "ALL_REJECTED"
    print(f"    SUMMARY: {status}  best_iter={best_iter}  "
          f"best_Π={best_pi:.3f}  best_Phi_enr/E={best_score*1000:.4f}/kJ  total_energy={total_energy:.3f}J")
    return best_score, best_plantuml, total_energy, n_accepted, best_pi


def _verdict(ratio: float) -> str:
    delta = (ratio - 1.0) * 100
    if ratio >= 1.05:
        return f"[BETTER]  +{delta:.1f}%  — AI context improved the solution"
    if ratio >= 0.95:
        return f"[EQUAL]   {delta:+.1f}%  — no significant difference"
    return f"[WORSE]   {delta:.1f}%  — AI context degraded the solution"


def _diagnosis(ctx: dict, acc_llm: int, acc_ai: int) -> str:
    conf = ctx["confidence"]
    if conf < 0.60:
        return f"low agent confidence ({conf:.2f}) — ambiguous keywords led to uncertain context"
    if acc_ai < acc_llm:
        return "more rejections in LLM+AI — enriched prompt may have confused JSON generation"
    return (f"pattern '{ctx['pattern_name']}' detected with confidence {conf:.2f} "
            f"but AI suggestion may have over-constrained the LLM output")


def _print_comparison(
    score_llm: float, score_ai: float,
    energy_llm: float, energy_ai: float,
    acc_llm: int, acc_ai: int,
    pi_llm: float, pi_ai: float,
    ai_agent: GofAIAgent | None,
):
    print("\n--- Π comparison (agent guidance quality) ---")
    pi_ratio = pi_ai / (pi_llm + 1e-12)
    print(f"  LLM-only  Π={pi_llm:.3f}")
    print(f"  LLM+AI    Π={pi_ai:.3f}")
    print(f"  Π ratio (AI/LLM): {pi_ratio:.2f}x  ->  {_verdict(pi_ratio)}")

    print("\n--- Phi_enr/E (global CIITR efficiency metric) ---")
    print(f"  LLM-only  Phi_enr/E={score_llm*1000:.4f}/kJ  energy={energy_llm:.3f}J  accepted={acc_llm}")
    print(f"  LLM+AI    Phi_enr/E={score_ai*1000:.4f}/kJ  energy={energy_ai:.3f}J  accepted={acc_ai}")

    score_ratio = score_ai / (score_llm + 1e-12)
    print(f"  Phi_enr/E ratio (AI/LLM): {score_ratio:.2f}x  ->  {_verdict(score_ratio)}")

    if score_ratio < 0.95 and ai_agent:
        print(f"  Diagnosis: {_diagnosis(ai_agent.get_pattern_context(), acc_llm, acc_ai)}")

    if ai_agent:
        ctx = ai_agent.get_pattern_context()
        rg = ai_agent.rg_proxy
        print(f"  AI detected : {ctx['pattern_name']}  (confidence={ctx['confidence']:.2f})"
              f"  top3={ctx['top3']}")
        print(f"  Rg (mean VFE): {rg:.4f}  |  Rg > 0: {rg > 0}"
              f"  — belief update {'active' if rg > 0 else 'inactive'}")


def run_session(session_data: dict, use_ai: bool, local: bool, warmup_iters: int = 1,
                max_iter: int = MAX_ITER, use_mlx: bool = False):
    ai_agent = GofAIAgent() if use_ai else None

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join("results", f"{session_data['session_id']}_gof_{timestamp}.csv")
    init_gof_csv(csv_path)

    raw_dir = os.path.join("results", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Session  : {session_data['session_id']}")
    print(f"Objective: {session_data['objective'].strip()[:80]}...")
    if use_mlx:
        llm_label = f"MLX M-series {MLX_MODEL_ID}"
    elif local:
        llm_label = "LOCAL llama3.1:8b (Ollama)"
    else:
        llm_label = "REMOTE devstral-small-2:24b"
    print(f"LLM      : {llm_label}  |  AI agent: {'ON' if use_ai else 'OFF'}")
    print(f"{'='*60}")

    score_llm, plantuml_llm, energy_llm, acc_llm, pi_llm = _run_mode(
        session_data, None, local, False, csv_path, "LLM_ONLY", raw_dir, warmup_iters, max_iter, use_mlx)

    if use_ai:
        score_ai, plantuml_ai, energy_ai, acc_ai, pi_ai = _run_mode(
            session_data, ai_agent, local, True, csv_path, "LLM_AI", raw_dir, warmup_iters, max_iter, use_mlx)

        _print_comparison(score_llm, score_ai, energy_llm, energy_ai, acc_llm, acc_ai, pi_llm, pi_ai, ai_agent)

        if plantuml_ai:
            puml_path = os.path.join("results", f"{session_data['session_id']}_AI.puml")
            with open(puml_path, "w", encoding="utf-8") as f:
                f.write(plantuml_ai)
            print(f"  PlantUML (AI): {puml_path}")

    if plantuml_llm:
        puml_path = os.path.join("results", f"{session_data['session_id']}_LLM.puml")
        with open(puml_path, "w", encoding="utf-8") as f:
            f.write(plantuml_llm)
        print(f"  PlantUML (LLM): {puml_path}")

    print(f"\nResults: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="CIITR GoF benchmark: LLM vs LLM+AI")
    parser.add_argument("--session", default=None,
                        help="session_id from sessions_gof.yaml (default: run all)")
    parser.add_argument("--remote", action="store_true", help="Use remote LLM")
    parser.add_argument("--mlx", action="store_true", help="Use MLX on Apple Silicon (M-series Neural Engine)")
    parser.add_argument("--no-ai", action="store_true", help="Run LLM-only mode")
    parser.add_argument("--warmup-iters", type=int, default=1,
                        help="Number of warmup iterations to exclude from stats (default: 1)")
    parser.add_argument("--max-iters", type=int, default=MAX_ITER,
                        help=f"Total iterations per mode (default: {MAX_ITER})")
    args = parser.parse_args()

    sessions = load_sessions()

    if args.session:
        matching = [s for s in sessions if s["session_id"] == args.session]
        if not matching:
            print(f"Session '{args.session}' not found in {SESSIONS_FILE}")
            return
        sessions_to_run = matching
    else:
        sessions_to_run = sessions

    use_mlx = args.mlx and not args.remote
    for session in sessions_to_run:
        run_session(session, use_ai=not args.no_ai, local=not args.remote,
                    warmup_iters=args.warmup_iters, max_iter=args.max_iters, use_mlx=use_mlx)


if __name__ == "__main__":
    main()

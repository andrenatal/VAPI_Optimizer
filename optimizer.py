"""
Vapi Voice Agent Optimizer
==========================
ML-driven system combining:
  1. DSPy (LLM-driven prompt generation) to create prompt component variants
  2. Optuna (Bayesian optimization via TPE) to search the combinatorial space
  3. Transcript feature extraction + failure clustering (scikit-learn)

Two-phase approach:
  Phase 1: DSPy iterative refinement (bad prompt → decent prompt fast)
  Phase 2: Optuna Bayesian search (fine-tune by mixing component variants)

Usage:
    export VAPI_API_KEY="your-key"
    export ANTHROPIC_API_KEY="your-key"
    python optimizer.py
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path

import dspy
import optuna
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────

VAPI_API_KEY = os.environ["VAPI_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
VAPI_BASE = "https://api.vapi.ai"
VAPI_HEADERS = {
    "Authorization": f"Bearer {VAPI_API_KEY}",
    "Content-Type": "application/json",
}

# Shared with test_call.py
SCHEDULER_PHONE_NUMBER = os.environ.get("VAPI_PHONE_A_NUMBER", "")   # scheduler answers here
PATIENT_PHONE_ID = os.environ.get("VAPI_PHONE_B_ID", "")            # patient calls from here

# Set after first test_call.py run
SCHEDULER_ASSISTANT_ID = os.environ.get("SCHEDULER_ASSISTANT_ID", "")
PATIENT_ASSISTANT_ID = os.environ.get("PATIENT_ASSISTANT_ID", "")

# Optimization settings
PHASE1_ITERATIONS = 4       # DSPy iterative refinement
PHASE2_OPTUNA_TRIALS = 6    # Bayesian search over component combos
CALLS_PER_EVAL = 1          # Calls per evaluation (keep low to save budget)
CALL_TIMEOUT = 240
MAX_CALL_DURATION = 180

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── DSPy Setup ─────────────────────────────────────────────────

lm = dspy.LM("anthropic/claude-sonnet-4-20250514", api_key=ANTHROPIC_API_KEY)
dspy.configure(lm=lm)


# ═══════════════════════════════════════════════════════════════
# COMPONENT 1: Vapi API Layer
# ═══════════════════════════════════════════════════════════════

def update_scheduler(prompt: str, first_message: str) -> dict:
    """PATCH the dental scheduler assistant's system prompt + greeting."""
    payload = {
        "model": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "messages": [{"role": "system", "content": prompt}],
            "temperature": 0.7,
            "maxTokens": 300,
        },
        "firstMessage": first_message,
    }
    resp = requests.patch(
        f"{VAPI_BASE}/assistant/{SCHEDULER_ASSISTANT_ID}",
        headers=VAPI_HEADERS, json=payload,
    )
    resp.raise_for_status()
    return resp.json()


def run_call_and_score() -> dict:
    """Run one test call (patient -> scheduler) and return scores."""
    # Initiate
    resp = requests.post(f"{VAPI_BASE}/call", headers=VAPI_HEADERS, json={
        "assistantId": PATIENT_ASSISTANT_ID,
        "phoneNumberId": PATIENT_PHONE_ID,
        "customer": {"number": SCHEDULER_PHONE_NUMBER},
    })
    resp.raise_for_status()
    call_id = resp.json()["id"]
    log.info(f"  Call started: {call_id}")

    # Poll until done
    time.sleep(5)
    start = time.time()
    while time.time() - start < CALL_TIMEOUT:
        r = requests.get(f"{VAPI_BASE}/call/{call_id}", headers=VAPI_HEADERS)
        r.raise_for_status()
        call = r.json()
        if call.get("status") == "ended":
            return _extract_scores(call)
        time.sleep(5)
    raise TimeoutError(f"Call {call_id} timed out")


def _extract_scores(call: dict) -> dict:
    """Pull all scoring signals from a completed call."""
    analysis = call.get("analysis", {})
    sd = analysis.get("structuredData", {})
    artifact = call.get("artifact", {})

    # Duration
    duration = 0
    try:
        t1 = datetime.fromisoformat(call["startedAt"].replace("Z", "+00:00"))
        t2 = datetime.fromisoformat(call["endedAt"].replace("Z", "+00:00"))
        duration = (t2 - t1).total_seconds()
    except Exception:
        pass

    # 6-item checklist
    checklist = [
        sd.get("schedulerGreetedProperly", False),
        sd.get("schedulerCollectedName", False),
        sd.get("schedulerOfferedTimes", False),
        sd.get("schedulerProvidedPricing", False),
        sd.get("schedulerConfirmedAppointment", False),
        sd.get("appointmentBooked", False),
    ]
    checklist_score = sum(checklist) / 6

    # Vapi 1-10 score
    try:
        vapi_score = int(analysis.get("successEvaluation", "0"))
    except (ValueError, TypeError):
        vapi_score = 0

    # Duration bonus (1.0 if <=90s, 0.0 if >=180s)
    duration_bonus = max(0.0, min(1.0, 1.0 - (duration - 90) / 90)) if duration > 90 else 1.0

    # Booking bonus
    booked = 1.0 if sd.get("appointmentBooked") else 0.0

    composite = (
        checklist_score * 0.50
        + (vapi_score / 10) * 0.20
        + duration_bonus * 0.15
        + booked * 0.15
    )

    return {
        "call_id": call.get("id"),
        "transcript": artifact.get("transcript", call.get("transcript", "")),
        "structured_data": sd,
        "vapi_score": vapi_score,
        "checklist": checklist,
        "checklist_score": checklist_score,
        "duration": duration,
        "duration_bonus": duration_bonus,
        "booked": booked,
        "composite": composite,
        "ended_reason": call.get("endedReason", ""),
        "cost": call.get("cost", 0),
    }


# ═══════════════════════════════════════════════════════════════
# COMPONENT 2: DSPy — LLM-Driven Prompt Generation
# ═══════════════════════════════════════════════════════════════

class AnalyzeAndImprove(dspy.Signature):
    """Analyze dental scheduler call failures and produce an improved prompt."""
    transcripts: str = dspy.InputField(desc="Call transcripts with scores")
    current_prompt: str = dspy.InputField(desc="Current system prompt")
    scores: str = dspy.InputField(desc="JSON scores from calls")
    failure_patterns: str = dspy.InputField(desc="ML-detected failure patterns from transcript analysis")

    improved_prompt: str = dspy.OutputField(desc="Complete improved system prompt. MUST include: clinic name, exact service prices in dollars, available hours with days/times, step-by-step booking flow, how to handle cost objections, cancellation policy.")
    improved_first_message: str = dspy.OutputField(desc="Greeting that identifies the clinic by name")


class GenerateComponentVariants(dspy.Signature):
    """Generate multiple distinct variants for a specific prompt component."""
    component_name: str = dspy.InputField(desc="e.g. 'services_and_pricing', 'booking_flow', 'objection_handling'")
    failure_context: str = dspy.InputField(desc="What failures this component should address")

    variant_a: str = dspy.OutputField(desc="First variant: concise and direct")
    variant_b: str = dspy.OutputField(desc="Second variant: detailed and thorough")
    variant_c: str = dspy.OutputField(desc="Third variant: warm and conversational")


class DSPyOptimizer:
    """Uses DSPy ChainOfThought for iterative prompt improvement."""

    def __init__(self):
        self.improver = dspy.ChainOfThought(AnalyzeAndImprove)
        self.variant_gen = dspy.ChainOfThought(GenerateComponentVariants)

    def improve(self, transcripts, current_prompt, scores, failure_patterns):
        """Analyze failures and generate an improved prompt."""
        result = self.improver(
            transcripts=transcripts,
            current_prompt=current_prompt,
            scores=scores,
            failure_patterns=failure_patterns,
        )
        return result.improved_prompt, result.improved_first_message

    def generate_variants(self, component_name, failure_context):
        """Generate 3 variants for a prompt component using DSPy."""
        result = self.variant_gen(
            component_name=component_name,
            failure_context=failure_context,
        )
        return [result.variant_a, result.variant_b, result.variant_c]


# ═══════════════════════════════════════════════════════════════
# COMPONENT 3: Transcript Feature Extraction (scikit-learn)
# ═══════════════════════════════════════════════════════════════

class TranscriptAnalyzer:
    """Extract features from transcripts + cluster failure modes."""

    HEDGE_WORDS = [
        "typically", "ranges", "varies", "depends", "unfortunately",
        "i don't have access", "i can't", "it varies", "approximately",
    ]
    CONFUSION_SIGNALS = [
        "i'm not a dental", "call directly", "contact the dental",
        "i don't actually handle", "i'm here to provide information",
        "i'm not a", "draft a script",
    ]

    def __init__(self):
        self.transcripts = []
        self.features = []
        self.scores = []

    def extract_features(self, transcript, scores):
        """Extract numerical features from a single call transcript."""
        lines = transcript.strip().split("\n")
        scheduler_lines = [l.replace("User: ", "") for l in lines if l.startswith("User:")]
        caller_lines = [l.replace("AI: ", "") for l in lines if l.startswith("AI:")]
        scheduler_text = " ".join(scheduler_lines).lower()

        features = {
            "total_turns": len(lines),
            "scheduler_turns": len(scheduler_lines),
            "caller_turns": len(caller_lines),
            "scheduler_avg_words": float(np.mean([len(l.split()) for l in scheduler_lines])) if scheduler_lines else 0,
            "caller_avg_words": float(np.mean([len(l.split()) for l in caller_lines])) if caller_lines else 0,
            "hedge_count": sum(1 for h in self.HEDGE_WORDS if h in scheduler_text),
            "confusion_count": sum(1 for c in self.CONFUSION_SIGNALS if c in scheduler_text),
            "mentioned_price": 1 if any(c in scheduler_text for c in ["$", "dollar", "120", "cost is"]) else 0,
            "mentioned_clinic": 1 if any(c in scheduler_text for c in ["bright smile", "dental clinic", "our clinic"]) else 0,
            "mentioned_hours": 1 if any(c in scheduler_text for c in ["monday", "friday", "8 am", "5 pm", "saturday"]) else 0,
            "confirmed_booking": 1 if any(c in scheduler_text for c in ["confirmed", "scheduled", "booked", "all set"]) else 0,
            "checklist_score": scores.get("checklist_score", 0),
            "composite_score": scores.get("composite", 0),
            "duration": scores.get("duration", 0),
        }

        self.transcripts.append(transcript)
        self.features.append(features)
        self.scores.append(scores)
        return features

    def get_failure_patterns(self):
        """Summarize detected failure patterns as text for DSPy."""
        if not self.features:
            return "No data yet."

        recent = self.features[-4:]  # last 4 calls
        avg_hedges = np.mean([f["hedge_count"] for f in recent])
        avg_confusion = np.mean([f["confusion_count"] for f in recent])
        avg_price = np.mean([f["mentioned_price"] for f in recent])
        avg_clinic = np.mean([f["mentioned_clinic"] for f in recent])
        avg_hours = np.mean([f["mentioned_hours"] for f in recent])
        avg_confirmed = np.mean([f["confirmed_booking"] for f in recent])
        avg_words = np.mean([f["scheduler_avg_words"] for f in recent])

        patterns = []
        if avg_confusion > 0.3:
            patterns.append(f"CRITICAL: Identity confusion in {avg_confusion*100:.0f}% of calls - agent doesn't know it IS the dental office")
        if avg_hedges > 1.5:
            patterns.append(f"HIGH: Excessive hedging ({avg_hedges:.1f} hedge phrases/call) - agent gives vague non-answers")
        if avg_price < 0.5:
            patterns.append(f"HIGH: Pricing not provided in {(1-avg_price)*100:.0f}% of calls")
        if avg_clinic < 0.5:
            patterns.append(f"MEDIUM: Clinic not identified in {(1-avg_clinic)*100:.0f}% of calls")
        if avg_hours < 0.5:
            patterns.append(f"MEDIUM: Hours not mentioned in {(1-avg_hours)*100:.0f}% of calls")
        if avg_confirmed < 0.5:
            patterns.append(f"HIGH: Booking not confirmed in {(1-avg_confirmed)*100:.0f}% of calls")
        if avg_words > 40:
            patterns.append(f"LOW: Agent is verbose (avg {avg_words:.0f} words/turn)")

        return "\n".join(patterns) if patterns else "No significant failure patterns detected."

    def cluster_failures(self):
        """K-Means clustering on TF-IDF of scheduler responses."""
        if len(self.transcripts) < 3:
            return {"clusters": [], "n_transcripts": len(self.transcripts)}

        scheduler_texts = []
        for t in self.transcripts:
            lines = [l.replace("User: ", "") for l in t.split("\n") if l.startswith("User:")]
            scheduler_texts.append(" ".join(lines))

        vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
        tfidf = vectorizer.fit_transform(scheduler_texts)
        feature_names = list(vectorizer.get_feature_names_out())

        n_clusters = min(3, len(scheduler_texts))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(tfidf)

        clusters = []
        for i in range(n_clusters):
            mask = [j for j, l in enumerate(labels) if l == i]
            top_idx = km.cluster_centers_[i].argsort()[-5:][::-1]
            top_terms = [feature_names[k] for k in top_idx]
            avg_score = float(np.mean([self.features[j]["checklist_score"] for j in mask]))
            clusters.append({
                "id": i, "size": len(mask),
                "top_terms": top_terms,
                "avg_checklist": round(avg_score, 3),
            })

        return {"clusters": clusters, "labels": [int(l) for l in labels]}


# ═══════════════════════════════════════════════════════════════
# COMPONENT 4: Bayesian Optimization (Optuna TPE)
# ═══════════════════════════════════════════════════════════════

class BayesianPromptOptimizer:
    """
    Uses Optuna's TPE (Tree-structured Parzen Estimator) to search
    over prompt component combinations.

    DSPy generates 3 variants per component (concise / detailed / warm).
    Optuna picks which combination to try next, using a probabilistic
    surrogate model to explore the 3^6 = 729 possible combinations
    efficiently in just 6 trials.
    """

    COMPONENTS = [
        "identity",
        "services_and_pricing",
        "hours_and_availability",
        "booking_flow",
        "objection_handling",
        "rules_and_guardrails",
    ]

    def __init__(self, dspy_optimizer):
        self.dspy_opt = dspy_optimizer
        self.variants = {}
        self.study = None
        self.trial_results = []

    def generate_all_variants(self, failure_context):
        """Use DSPy to generate 3 variants per prompt component."""
        log.info("  Generating prompt component variants via DSPy...")
        for component in self.COMPONENTS:
            log.info(f"    {component}...")
            variants = self.dspy_opt.generate_variants(component, failure_context)
            self.variants[component] = variants

    def _assemble_prompt(self, selections):
        """Assemble a full prompt from selected component variants."""
        parts = []
        for component in self.COMPONENTS:
            idx = selections[component]
            title = component.replace("_", " ").title()
            parts.append(f"## {title}\n{self.variants[component][idx]}")
        return "\n\n".join(parts)

    def _objective(self, trial):
        """Optuna objective: select variants, deploy, call, score."""
        selections = {}
        for component in self.COMPONENTS:
            selections[component] = trial.suggest_categorical(component, [0, 1, 2])

        prompt = self._assemble_prompt(selections)
        first_message = "Hello! Thank you for calling Bright Smile Dental Clinic. How can I help you today?"

        log.info(f"\n  Optuna Trial {trial.number}: {selections}")

        update_scheduler(prompt, first_message)
        time.sleep(2)

        scores_list = []
        for i in range(CALLS_PER_EVAL):
            try:
                scores = run_call_and_score()
                scores_list.append(scores)
                log.info(f"    Call: composite={scores['composite']:.3f} checklist={scores['checklist_score']:.3f} booked={scores['booked']}")
            except Exception as e:
                log.error(f"    Call failed: {e}")
                scores_list.append({"composite": 0})

        avg = float(np.mean([s["composite"] for s in scores_list]))

        self.trial_results.append({
            "trial": trial.number,
            "selections": selections,
            "prompt": prompt,
            "first_message": first_message,
            "scores": scores_list,
            "avg_composite": avg,
        })

        return avg

    def optimize(self, n_trials):
        """Run Bayesian optimization over prompt component space."""
        log.info(f"  Search space: {len(self.COMPONENTS)} components x 3 variants = {3**len(self.COMPONENTS)} combos")
        log.info(f"  Optuna TPE will explore {n_trials} trials efficiently")

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        self.study.optimize(self._objective, n_trials=n_trials)

        best = self.study.best_trial
        best_sel = {c: best.params[c] for c in self.COMPONENTS}
        best_prompt = self._assemble_prompt(best_sel)

        log.info(f"\n  Optuna best: trial {best.number}, score {best.value:.3f}")
        log.info(f"  Selections: {best_sel}")

        return {
            "best_prompt": best_prompt,
            "best_first_message": "Hello! Thank you for calling Bright Smile Dental Clinic. How can I help you today?",
            "best_score": best.value,
            "best_selections": best_sel,
            "all_trials": self.trial_results,
            "n_trials": n_trials,
            "search_space_size": 3 ** len(self.COMPONENTS),
        }


# ═══════════════════════════════════════════════════════════════
# MAIN: Two-Phase Optimization Pipeline
# ═══════════════════════════════════════════════════════════════

def optimize():
    """
    Phase 1 — DSPy iterative refinement:
        Rapidly improve from terrible to decent by analyzing transcripts.
    Phase 2 — Optuna Bayesian search:
        Fine-tune by generating component variants and searching combos.
    """
    log.info("=" * 60)
    log.info("VAPI VOICE AGENT OPTIMIZER")
    log.info("DSPy + Optuna (Bayesian) + scikit-learn (clustering)")
    log.info("=" * 60)

    dspy_opt = DSPyOptimizer()
    analyzer = TranscriptAnalyzer()
    history = []

    current_prompt = "You are a receptionist at a dental office. Help people who call."
    current_first_message = "Hello! How can I help you today?"
    best_score = 0.0
    best_prompt = current_prompt
    best_first_message = current_first_message

    # ── PHASE 1: DSPy Iterative Refinement ─────────────────────

    log.info(f"\n{'=' * 60}")
    log.info("PHASE 1: DSPy Iterative Refinement")
    log.info(f"{'=' * 60}")

    for iteration in range(PHASE1_ITERATIONS):
        log.info(f"\n--- Iteration {iteration + 1}/{PHASE1_ITERATIONS} ---")

        update_scheduler(current_prompt, current_first_message)
        time.sleep(2)

        iter_scores = []
        for c in range(CALLS_PER_EVAL):
            log.info(f"  Call {c+1}...")
            try:
                scores = run_call_and_score()
                iter_scores.append(scores)
                analyzer.extract_features(scores["transcript"], scores)
                log.info(f"    composite={scores['composite']:.3f} checklist={sum(scores['checklist'])}/6 vapi={scores['vapi_score']} dur={scores['duration']:.0f}s booked={scores['booked']>0}")
            except Exception as e:
                log.error(f"    Failed: {e}")
                iter_scores.append({"composite": 0, "checklist_score": 0, "transcript": ""})

        avg = float(np.mean([s["composite"] for s in iter_scores]))
        avg_cl = float(np.mean([s.get("checklist_score", 0) for s in iter_scores]))

        if avg > best_score:
            best_score, best_prompt, best_first_message = avg, current_prompt, current_first_message
            log.info(f"  * New best: {best_score:.3f}")

        history.append({
            "phase": 1, "iteration": iteration + 1,
            "prompt": current_prompt, "first_message": current_first_message,
            "scores": iter_scores, "avg_composite": avg, "avg_checklist": avg_cl,
        })
        with open(RESULTS_DIR / f"phase1_iter{iteration+1}.json", "w") as f:
            json.dump(history[-1], f, indent=2, default=str)

        if avg_cl >= 0.95:
            log.info("  Target reached!")
            break

        # DSPy improvement — fed by ML failure pattern detection
        failure_patterns = analyzer.get_failure_patterns()
        log.info(f"  ML failure patterns:\n    " + failure_patterns.replace("\n", "\n    "))

        transcripts_text = "\n---\n".join([
            f"Call (score={s.get('composite',0):.2f}):\n{s.get('transcript','')}" for s in iter_scores
        ])
        scores_text = json.dumps([{
            "composite": s.get("composite", 0),
            "checklist": s.get("checklist", []),
            "structured_data": s.get("structured_data", {}),
        } for s in iter_scores], indent=2)

        try:
            new_prompt, new_fm = dspy_opt.improve(
                transcripts_text, current_prompt, scores_text, failure_patterns,
            )
            current_prompt, current_first_message = new_prompt, new_fm
            log.info(f"  DSPy generated new prompt ({len(current_prompt)} chars)")
        except Exception as e:
            log.error(f"  DSPy failed: {e}")

    # ── PHASE 2: Bayesian Optimization ─────────────────────────

    log.info(f"\n{'=' * 60}")
    log.info("PHASE 2: Bayesian Optimization (Optuna TPE)")
    log.info(f"{'=' * 60}")

    failure_context = analyzer.get_failure_patterns()
    bayesian = BayesianPromptOptimizer(dspy_opt)
    bayesian.generate_all_variants(failure_context)

    optuna_result = bayesian.optimize(n_trials=PHASE2_OPTUNA_TRIALS)

    if optuna_result["best_score"] > best_score:
        best_score = optuna_result["best_score"]
        best_prompt = optuna_result["best_prompt"]
        best_first_message = optuna_result["best_first_message"]
        log.info(f"  * Optuna beat Phase 1: {best_score:.3f}")

    for td in optuna_result["all_trials"]:
        history.append({
            "phase": 2, "iteration": td["trial"] + 1,
            "prompt": td["prompt"], "first_message": td["first_message"],
            "scores": td["scores"], "avg_composite": td["avg_composite"],
            "optuna_selections": td["selections"],
        })

    with open(RESULTS_DIR / "phase2_optuna.json", "w") as f:
        json.dump(optuna_result, f, indent=2, default=str)

    # ── Final Validation ───────────────────────────────────────

    log.info(f"\n{'=' * 60}")
    log.info("FINAL VALIDATION")
    log.info(f"{'=' * 60}")

    update_scheduler(best_prompt, best_first_message)
    time.sleep(2)

    log.info("  Running final validation call with best prompt...")
    try:
        final = run_call_and_score()
        analyzer.extract_features(final["transcript"], final)
        log.info(f"  FINAL: composite={final['composite']:.3f} checklist={sum(final['checklist'])}/6 booked={final['booked']>0}")
    except Exception as e:
        log.error(f"  Validation failed: {e}")
        final = {}

    # ── Report ─────────────────────────────────────────────────

    cluster_result = analyzer.cluster_failures()
    starting = history[0]["avg_composite"] if history else 0

    report = {
        "summary": {
            "starting_prompt": "You are a receptionist at a dental office. Help people who call.",
            "best_prompt": best_prompt,
            "best_first_message": best_first_message,
            "starting_score": starting,
            "best_score": best_score,
            "improvement": best_score - starting,
            "improvement_pct": ((best_score - starting) / max(starting, 0.001)) * 100,
            "total_calls": sum(len(h.get("scores", [])) for h in history),
            "total_cost": sum(s.get("cost", 0) for h in history for s in h.get("scores", [])),
        },
        "improvement_curve": [
            {"phase": h["phase"], "iteration": h["iteration"], "avg_composite": h["avg_composite"]}
            for h in history
        ],
        "failure_clusters": cluster_result,
        "optuna_search": {
            "search_space_size": optuna_result.get("search_space_size"),
            "trials_run": optuna_result.get("n_trials"),
            "best_selections": optuna_result.get("best_selections"),
        },
        "component_variants": bayesian.variants,
        "final_validation": final,
        "full_history": history,
    }

    with open(RESULTS_DIR / "final_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info(f"\n{'=' * 60}")
    log.info(f"  Start:       {starting:.3f}")
    log.info(f"  Best:        {best_score:.3f}")
    log.info(f"  Improvement: +{report['summary']['improvement']:.3f} ({report['summary']['improvement_pct']:.0f}%)")
    log.info(f"  Calls:       {report['summary']['total_calls']}")
    log.info(f"  Cost:        ${report['summary']['total_cost']:.2f}")
    log.info(f"  Report:      {RESULTS_DIR}/final_report.json")
    log.info(f"{'=' * 60}")

    return report


if __name__ == "__main__":
    optimize()
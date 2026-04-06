# Vapi Voice Agent Optimizer

ML-driven system that automatically improves a Vapi voice agent through iterative evaluation and prompt optimization.

## What is this

Vapi is a developer platform for building voice agents using STT, LLMs, and TTS. It abstracts away the complexity of coordinating these subsystems through a web platform and APIs, while giving developers full control over provider selection and configuration.

But this raises a fundamental challenge in the LLM era: how do you systematically optimize your agent's prompts? Manual prompt tweaking is brittle, tedious, and empirically driven — a human changes words until they "think" it's good enough, deploys, observes, and repeats. It's rarely data-driven.

This repository automates that process. It introduces a framework that creates baseline agents, evaluates them via real phone calls between an instructed bot and an agent by using structured scoring, applies DSPy-driven prompt optimization and Bayesian search (Optuna TPE) over prompt configurations, and iterates until convergence — replacing human intuition with quantitative feedback loops.

## How It Works

### 0. tl;dr

1. Call recording prior to optimization



https://github.com/user-attachments/assets/dadfe9b1-e965-4ce0-a437-e8007bfda7c0



2. Call recording after optimization

### 1. Agent Creation & Baseline Evaluation — `create_agents.py`

Creates two Vapi assistants: a dental office scheduler (gpt-4o-mini) with a minimal, unoptimized prompt, and a simulated patient caller (gpt-4o) with a challenging persona that pushes back on vague pricing, asks about tooth pain, and demands cancellation policy details. The scheduler is assigned to an inbound phone number, the patient calls it, and the resulting call is scored via Vapi's built-in analysis plan — establishing the baseline performance that the optimizer will improve upon.

### 2. Optimization Loop — `optimizer.py`

#### 2.1 DSPy Iterative Refinement

Runs 4 iterations of prompt improvement. Each iteration deploys the current prompt to the scheduler, triggers a test call, and collects Vapi's structured evaluation (6-boolean checklist + NumericScale 1-10). A scikit-learn feature extractor analyzes the transcript for quantitative failure signals — hedge word frequency, identity confusion, missing pricing, and booking confirmation gaps. These patterns are fed into a DSPy `Chain Of Thought` module (powered by Claude Sonnet) that analyzes the failures and generates an improved prompt. The loop repeats until the checklist score reaches 95% or all iterations are exhausted.

#### 2.2 Bayesian Prompt Search (Optuna TPE)

Decomposes the best prompt from Phase 1 into 6 modular components: identity, services & pricing, hours & availability, booking flow, objection handling, and rules & guardrails. DSPy generates 3 stylistic variants per component (concise, detailed, warm), creating a search space of 3⁶ = 729 possible prompt configurations. Optuna's Tree-structured Parzen Estimator explores this space efficiently in 6 trials, using a probabilistic surrogate model to select promising combinations without exhaustive search. Each trial assembles a prompt, deploys it, runs a real call, and reports the composite score back to the optimizer.

#### 3. Final Validation — `optimizer.py` + `test_call.py`

The optimizer deploys the highest-scoring prompt and runs a final validation call to confirm performance. Additionally, test_call.py can be used for manual re-testing at any time — it triggers a call using the existing assistants and prints the full scorecard.

## Steps to reproduce

### Prerequisites

- Python 3.10+
- Vapi account with 2 phone numbers (A twillio number might be required since the iterative process can exhaust outbound calls)
- Anthropic API key (for DSPy optimizer)

#### Environment Variables

These variables are required to be changed in the .env file

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"

# get these from Vapi's panel
export VAPI_API_KEY="your-vapi-key"
export VAPI_PHONE_A_NUMBER=""  # scheduler's inbound number.
export VAPI_PHONE_B_ID=""  # patient's phone ID

# From create_agents.py output (run it once first)
export SCHEDULER_ASSISTANT_ID="your-scheduler-assistant-id"
export PATIENT_ASSISTANT_ID="your-patient-assistant-id"
```

#### Step by step execution

#### 1. Run `uv run create_agents.py` which will:

- Create the dental scheduler assistant (gpt-4o-mini + low quality prompt)
- Create the simulated patient assistant (gpt-4o + hard persona + analysisPlan)
- Assign the scheduler to Phone A
- Make one test call (patient → scheduler)
- Print results + assistant IDs
- After it finishes, grab the two assistant IDs from the output and add them to your .env

That will print a summary of the call where you can note the transcription and the Structured Data output by the Analysis Plan. A very simple and vague instruction prompt is provided to the agent. That's what we want to optimize. All the logs from the call are stored inside the folder `call_results`, including its recording.

```
────────────────────────────────────────
TRANSCRIPT
────────────────────────────────────────
User: Hello. How can I help you today?
AI: Hi. Um, I wanted to schedule a teeth cleaning. If possible.
User: I can help you with that. However, I don't have the ability to schedule appointments directly. I recommend calling your dental office or visiting their website to book your teeth cleaning If you need help finding a dental office or have any other questions, let me know.
AI: Oh, okay. Sorry. I thought this was the dental office. I'll try calling back. Thanks.
User: No problem at all. If you need any help preparing for your appointment, or have questions about dental care, feel free to ask. Have a great day.


────────────────────────────────────────
ANALYSIS
────────────────────────────────────────

Summary: The AI attempted to schedule a teeth cleaning appointment, mistakenly believing they had reached their dental office. The User clarified that they could not directly schedule appointments and advised the AI to contact their dental office or visit their website. The AI then realized their error and ended the call to contact the correct office.

Structured Data:
{
  "serviceRequested": "a teeth cleaning",
  "appointmentBooked": false,
  "schedulerOfferedTimes": false,
  "schedulerCollectedName": false,
  "schedulerGreetedProperly": true,
  "schedulerProvidedPricing": false,
  "schedulerConfirmedAppointment": false
}

Success Evaluation: 1
Vapi Score: 1/10 | Duration: 42.0s
```

Update `SCHEDULER_ASSISTANT_ID` and `PATIENT_ASSISTANT_ID` in the `.env` with the ids output by this step.

#### 2. Now run `uv run optimizer.py.`

This can take ~30-40 minutes, ~10-11 calls, and cost ~$2-3 usd.

- Phase 1 — DSPy Iterative Refinement (4 iterations)
  - Deploy bad prompt to scheduler via PATCH /assistant
  - Run test call (patient → scheduler)
  - Score call using Vapi's structured data + NumericScale
  - Extract transcript features (hedges, confusion, pricing mentions) via scikit-learn
  - Feed transcripts + scores + failure patterns to DSPy (Claude Sonnet)
  - DSPy generates improved prompt
  - Repeat 4 times or until checklist hits 95%

- Phase 2 — Bayesian Search (6 trials)
  - DSPy generates 3 variants for each of 6 prompt components (18 variants total)
  - Optuna TPE picks a combination of variants
  - Deploy assembled prompt, run call, score it
  - Report score back to Optuna
  - Repeat 6 times, Optuna learns which combos work

- Final step
  - Deploy best prompt found across both phases
  - Run one validation call
  - Save report to results/final_report.json

After this you'll have an optimized prompt automatically deployed to the agent's configuration and the results of the final evaluation saved to `results/final_report.json`.

#### 3. Final test

Run `uv run test_call.py` to verify the optimized agent with a fresh call. This uses the existing assistants (no creation) and prints the full scorecard — checklist, Vapi score, booking status, and duration. Results are saved to `call_results/`. Compare it to the first call.

## Architecture

### Two-Assistant Test Framework

Three different models, three different jobs:

- **gpt-4o-mini** — the scheduler being optimized (cheap, weak. we want to extract the most of it)
- **gpt-4o** — the simulated patient (strong, fixed, challenging tester)
- **Claude Sonnet** — the optimizer brain (DSPy analysis, prompt generation, variant creation)

The bot patient calls the scheduler via Vapi's telephony. Real voice calls are made, transcribed, and scored automatically.

### Scoring Pipeline

After each call, Vapi's `analysisPlan` extracts:

- `structuredData`: Boolean checklist of scheduler behaviors
- `successEvaluation`: 1-10 NumericScale score from LLM judge

The optimizer combines these with call duration and booking status into a single composite metric that DSPy optimizes against.

## Results

Starting from a minimal prompt ("You are a receptionist at a dental office. Help people who call."), the optimizer discovers that the prompt needs:

- Clinic identity (name in greeting)
- Specific service prices (not ranges)
- Available hours
- Structured booking flow
- Objection handling instructions
- Cancellation policy

| Metric             | Before         | After |
| ------------------ | -------------- | ----- |
| Checklist          | 0-2/6          | 6/6   |
| NumericScale       | 1-7            | 10    |
| Appointment Booked | ❌             | ✅    |
| Call Duration      | 3min (timeout) | ~70s  |
| Cost per call      | $0.24          | $0.09 |

## Tradeoffs & Limitations

- **Single call per evaluation.** Each iteration runs one test call to save budget (~$0.24/call). This introduces variance — a single unlucky call can misrepresent a prompt's quality. A production system would run 3-5 calls per evaluation and average the scores, at the cost of longer runtime and higher spend.

- **Fixed test persona.** The simulated patient always plays the same character (Maria Garcia, anxious about pricing, tooth pain, cancellation policy). This means the optimizer can overfit to this specific scenario. A more robust system would rotate through multiple patient personas with different personalities, requests, and objection styles.

- **Limited Optuna budget.** Phase 2 explores only 6 trials out of 729 possible prompt configurations. TPE makes this efficient, but more trials would yield better coverage of the search space. The tradeoff is deliberate — each trial costs a real phone call, and the assignment is scoped for ~$2-3 total spend.

- **Composite metric weights** are hand-tuned. The scoring formula (checklist 50%, NumericScale 20%, duration 15%, booking 15%) reflects my judgment about what matters for a dental scheduler. These weights could themselves be optimized, but that adds a meta-optimization layer that's out of scope.

- **Prompt-only optimization.** The system only optimizes the system prompt and first message. It doesn't touch model parameters (temperature, max tokens), voice settings, or transcriber configuration. These could be added as additional Optuna dimensions, but prompt quality is the highest-leverage lever for this use case.

- **Evaluation relies on Vapi's analysis plan.** The structured data extraction and NumericScale scoring are performed by Vapi's built-in LLM judge. This introduces its own biases — for example, the NumericScale often gave 7-9 to calls where the appointment was never booked. The boolean checklist proved more reliable than the subjective score, which is why it carries 50% of the composite weight.

## Sequence Diagrams

### Create Agents & Test Call (`create_agents.py`)

```mermaid
sequenceDiagram
    participant Script as create_agents.py
    participant Vapi as Vapi API
    participant PhoneB as Phone B (Patient)
    participant PhoneA as Phone A (Scheduler)
    participant Results as call_results/

    Note over Script,Vapi: Step 1-2: Create Assistants
    Script->>Vapi: POST /assistant (Dental Scheduler)
    Vapi-->>Script: { id: scheduler_id }
    Script->>Vapi: POST /assistant (Simulated Patient + analysisPlan)
    Vapi-->>Script: { id: patient_id }

    Note over Script,Vapi: Step 3: Assign Scheduler to Phone A
    Script->>Vapi: PATCH /phone-number/{phoneA} (assistantId=scheduler)
    Vapi-->>Script: OK

    Note over Script,Vapi: Step 4: Initiate Call
    Script->>Vapi: POST /call (assistantId=patient, phoneNumberId=B, customer=A)
    Vapi-->>Script: { id: call_id, status: "queued" }

    Vapi->>PhoneB: Activate patient assistant
    PhoneB->>PhoneA: Outbound call (PSTN)
    Vapi->>PhoneA: Activate scheduler assistant (inbound)

    Note over PhoneB,PhoneA: Real voice conversation<br/>Patient (bot) ↔ Scheduler (user)

    PhoneA-->>PhoneB: endCallPhrase detected → hang up

    Vapi->>Vapi: Run analysisPlan<br/>• structuredData extraction<br/>• successEvaluation (NumericScale 1-10)<br/>• summary generation

    Note over Script,Vapi: Step 5: Poll & Collect Results
    loop Poll every 5s
        Script->>Vapi: GET /call/{call_id}
        Vapi-->>Script: { status: "in-progress" }
    end

    Script->>Vapi: GET /call/{call_id}
    Vapi-->>Script: { status: "ended", analysis, artifact, ... }

    Script->>Script: Compute verdict<br/>• checklist score (6 criteria)<br/>• duration check (>60s = fail)<br/>• final pass/fail

    Script->>Results: Save call.json
    Script->>Results: Save transcript.txt
    Script->>Vapi: GET recordingUrl
    Vapi-->>Script: audio data (WAV)
    Script->>Results: Save recording.wav
    Script->>Results: Save verdict.json
```

### Optimization Loop (`optimizer.py`)

```mermaid
sequenceDiagram
    participant Optimizer as optimizer.py
    participant DSPy as DSPy (Claude Sonnet)
    participant Vapi as Vapi API
    participant PhoneB as Phone B (Patient)
    participant PhoneA as Phone A (Scheduler)

    Note over Optimizer: Phase 1: DSPy Iterative Refinement

    Optimizer->>Vapi: PATCH /assistant/{scheduler}<br/>Deploy candidate prompt

    loop For each evaluation call
        Optimizer->>Vapi: POST /call (patient → scheduler)
        Vapi->>PhoneB: Activate patient
        PhoneB->>PhoneA: Voice call
        Note over PhoneB,PhoneA: AI-to-AI conversation
        PhoneA-->>PhoneB: Call ends

        loop Poll until ended
            Optimizer->>Vapi: GET /call/{id}
        end

        Vapi-->>Optimizer: transcript + structuredData + successEvaluation
    end

    Optimizer->>Optimizer: Extract features from transcripts<br/>• turn count, hedge words<br/>• identity confusion, response length

    Optimizer->>Optimizer: K-Means clustering on TF-IDF<br/>Identify failure modes

    Optimizer->>DSPy: AnalyzeAndImprove<br/>transcripts + scores + failure patterns
    DSPy-->>Optimizer: improved_prompt + improved_first_message

    Note over Optimizer: Repeat Phase 1 (4 iterations)

    Note over Optimizer: Phase 2: Optuna Bayesian Search

    Optimizer->>DSPy: GenerateComponentVariants<br/>Break best prompt into 6 components
    DSPy-->>Optimizer: 3 variants per component (729 combos)

    loop Optuna trials (6 iterations)
        Optimizer->>Optimizer: TPE selects component combination
        Optimizer->>Vapi: PATCH /assistant/{scheduler}<br/>Assembled prompt
        Optimizer->>Vapi: POST /call → poll → score
        Vapi-->>Optimizer: composite score
        Optimizer->>Optimizer: Report score to Optuna
    end

    Optimizer->>Optimizer: Select best combination
    Optimizer->>Vapi: PATCH /assistant/{scheduler}<br/>Final optimized prompt
    Optimizer->>Vapi: POST /call (validation run)
    Vapi-->>Optimizer: Final scores

    Optimizer->>Optimizer: Save results/final_report.json
```

### Full System Overview

```mermaid
sequenceDiagram
    participant User
    participant CreateAgents as create_agents.py
    participant TestCall as test_call.py
    participant Optimizer as optimizer.py
    participant Visualize as visualize.py
    participant Vapi as Vapi API

    User->>CreateAgents: Create assistants & run test call
    CreateAgents->>Vapi: POST /assistant (scheduler + patient)
    CreateAgents->>Vapi: PATCH /phone-number (assign scheduler)
    CreateAgents->>Vapi: POST /call → poll → score
    Vapi-->>CreateAgents: results
    CreateAgents-->>User: assistant IDs + call_results/{id}/

    User->>TestCall: Quick re-test (existing assistants)
    TestCall->>Vapi: POST /call → poll → score
    Vapi-->>TestCall: results
    TestCall-->>User: call_results/{id}/ (json + audio + transcript)

    User->>Optimizer: Run optimization
    Optimizer->>Vapi: Iterative PATCH + call + score loop
    Note over Optimizer,Vapi: Phase 1: DSPy refinement (4 iterations)<br/>Phase 2: Optuna search (6 trials)
    Optimizer-->>User: results/final_report.json

    User->>Visualize: View results
    Visualize-->>User: Improvement curve + scores + prompts
```

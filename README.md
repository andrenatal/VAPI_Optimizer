# Vapi Voice Agent Optimizer

ML-driven system that automatically improves a Vapi voice agent through iterative evaluation and prompt optimization.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                   Optimization Loop                      │
│                                                          │
│  1. Deploy candidate prompt → PATCH /assistant           │
│  2. Run test calls → POST /call (patient calls scheduler)│
│  3. Score calls → GET /call/{id} (structured data + eval)│
│  4. Analyze failures → DSPy (LLM-driven analysis)       │
│  5. Generate improved prompt → DSPy + Bayesian search    │
│  6. Repeat until convergence                             │
└─────────────────────────────────────────────────────────┘
```

### ML Components

1. **DSPy Prompt Optimization** — Uses `ChainOfThought` to analyze failing transcripts and generate improved prompts. Two-stage pipeline: failure analysis → prompt generation.

2. **Transcript Feature Extraction** — Extracts numerical features (turn count, hedge words, identity confusion, response length) from each call transcript.

3. **Failure Clustering** — K-Means clustering on TF-IDF vectors of scheduler responses to identify distinct failure modes (e.g., "pricing loop", "identity confusion", "never confirms").

4. **Composite Scoring** — Weighted metric combining:
   - Checklist score (50%): 6 boolean criteria from Vapi structured data
   - Vapi NumericScale (20%): 1-10 LLM judge score
   - Duration bonus (15%): Penalty for calls over 90 seconds
   - Booking bonus (15%): Whether appointment was actually booked

### Evaluation Criteria (Checklist)

| Criterion                       | What It Measures                        |
| ------------------------------- | --------------------------------------- |
| `schedulerGreetedProperly`      | Identified clinic by name in greeting   |
| `schedulerCollectedName`        | Asked for and recorded patient's name   |
| `schedulerOfferedTimes`         | Proactively offered available hours     |
| `schedulerProvidedPricing`      | Gave specific (not vague) pricing       |
| `schedulerConfirmedAppointment` | Confirmed booking details before ending |
| `appointmentBooked`             | Actually completed the booking          |

## Setup

### Prerequisites

- Python 3.10+
- Vapi account with 2 phone numbers and 2 assistants configured
- Anthropic API key (for DSPy optimizer)
- OpenAI API key (used by Vapi for the scheduler model)

### Environment Variables

```bash
export VAPI_API_KEY="your-vapi-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Shared with test_call.py (you already have these)
export VAPI_PHONE_A_NUMBER="+16282441616"           # scheduler's inbound number
export VAPI_PHONE_B_ID=""  # patient's phone ID

# From test_call.py output (run it once first)
export SCHEDULER_ASSISTANT_ID="your-scheduler-assistant-id"
export PATIENT_ASSISTANT_ID="your-patient-assistant-id"
```

### Install & Run

```bash
pip install -r requirements.txt
python optimizer.py
```

### View Results

```bash
python visualize.py
```

## Sequence Diagram

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

## Architecture

### Two-Assistant Test Framework

Three different models, three different jobs:

- **gpt-4o-mini** — the scheduler being optimized (cheap, weak — the whole point)
- **gpt-4o** — the simulated patient (strong, fixed, challenging tester)
- **Claude Sonnet** — the optimizer brain (DSPy analysis, prompt generation, variant creation)

The patient calls the scheduler via Vapi's telephony. Real voice calls are made, transcribed, and scored automatically.

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

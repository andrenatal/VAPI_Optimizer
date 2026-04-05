"""
Vapi Voice Agent Test: Two assistants talking to each other.

Setup:
1. Create two free Vapi phone numbers in the dashboard
2. Set your VAPI_API_KEY env variable
3. Set PHONE_NUMBER_A_ID and PHONE_NUMBER_B_ID (from dashboard)
4. Run this script

Flow:
- Assistant A (dental scheduler) is assigned to Phone Number A
- Assistant B (simulated patient) calls Phone Number A from Phone Number B
- They have a real voice conversation
- We pull the transcript and analysis when it ends
"""

import os
import time
import json
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("VAPI_API_KEY")
BASE_URL = "https://api.vapi.ai"

# ── You need to set these from your Vapi dashboard ──
PHONE_NUMBER_A_ID = os.environ.get("VAPI_PHONE_A_ID", "")  # dental scheduler's number
PHONE_NUMBER_B_ID = os.environ.get("VAPI_PHONE_B_ID", "")  # simulated patient's number
PHONE_NUMBER_A_NUMBER = os.environ.get("VAPI_PHONE_A_NUMBER", "")  # e.g. "+14155551234"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


# ─────────────────────────────────────────────
# Step 1: Create the Dental Scheduler (Assistant A)
# ─────────────────────────────────────────────

DENTAL_SCHEDULER_PROMPT = """Help people who call.
"""


def create_dental_scheduler():
    """Create Assistant A: the dental office scheduler."""
    payload = {
        "name": "Dental Scheduler (Optimization Target)",
        "firstMessage": "Hello! How can I help you today?",
        "firstMessageMode": "assistant-speaks-first",
        "model": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": DENTAL_SCHEDULER_PROMPT}
            ],
            "temperature": 0.7,
            "maxTokens": 300,
        },
        "voice": {
            "provider": "vapi",
            "voiceId": "Elliot",
        },
        "maxDurationSeconds": 180,  # 3 min max per test call
        "endCallPhrases": [
            "goodbye",
            "bye",
            "have a great day",
            "have a wonderful day",
            "take care",
        ],
    }

    resp = requests.post(f"{BASE_URL}/assistant", headers=HEADERS, json=payload)
    resp.raise_for_status()
    assistant = resp.json()
    print(f"✅ Created dental scheduler: {assistant['id']}")
    return assistant


# ─────────────────────────────────────────────
# Step 2: Create the Simulated Patient (Assistant B)
# ─────────────────────────────────────────────

SIMULATED_PATIENT_PROMPT = """You are Maria Garcia, a 34-year-old woman calling a dental office.

## Your Situation
- You want a teeth cleaning
- You'd prefer next Tuesday at 2pm, but you're flexible
- You don't have dental insurance and are very worried about cost
- You haven't been to a dentist in 2 years
- You're also wondering if you might need a root canal because you've had some pain in your back molar

## How to Behave
- Be polite but anxious
- BEFORE agreeing to book, ask how much a cleaning costs
- If they give a vague range instead of a specific price, push back: "Can you give me an exact number?"
- Mention your tooth pain and ask if they can look at that too
- Ask what happens if you need to cancel last minute
- If at any point they can't answer a question, express frustration
- Provide your name when asked: "Maria Garcia"
- Once all your questions are answered, confirm the appointment

## Important
- You are the CALLER. Wait for the receptionist to greet you first.
- Keep your responses short and natural, like a real phone call.
- Do NOT reveal that you are an AI or a test.
- Do NOT make it easy — you have real concerns that need addressing.
"""


def create_simulated_patient():
    """Create Assistant B: the simulated patient caller."""
    payload = {
        "name": "Simulated Patient (Tester)",
        "model": {
            "provider": "openai",
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": SIMULATED_PATIENT_PROMPT}
            ],
            "temperature": 0.8,
            "maxTokens": 200,
        },
        "voice": {
            "provider": "vapi",
            "voiceId": "Jess",
        },
        "firstMessageMode": "assistant-waits-for-user",
        "maxDurationSeconds": 180,
        "endCallPhrases": [
            "goodbye",
            "bye",
            "have a great day",
            "have a wonderful day",
            "take care",
        ],
        "analysisPlan": {
            "structuredDataPlan": {
                "enabled": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "schedulerGreetedProperly": {
                            "type": "boolean",
                            "description": "Whether the user (dental receptionist) greeted the caller warmly",
                        },
                        "schedulerCollectedName": {
                            "type": "boolean",
                            "description": "Whether the user (dental receptionist) asked for and collected the caller's full name",
                        },
                        "schedulerOfferedTimes": {
                            "type": "boolean",
                            "description": "Whether the user (dental receptionist) offered available appointment times",
                        },
                        "schedulerConfirmedAppointment": {
                            "type": "boolean",
                            "description": "Whether the user (dental receptionist) confirmed the appointment details before ending",
                        },
                        "schedulerProvidedPricing": {
                            "type": "boolean",
                            "description": "Whether the user (dental receptionist) provided pricing when asked",
                        },
                        "serviceRequested": {
                            "type": "string",
                            "description": "The dental service the caller (bot) asked about",
                        },
                        "appointmentDate": {
                            "type": "string",
                            "description": "The date confirmed by the user (dental receptionist)",
                        },
                        "appointmentTime": {
                            "type": "string",
                            "description": "The time confirmed by the user (dental receptionist)",
                        },
                        "appointmentBooked": {
                            "type": "boolean",
                            "description": "Whether an appointment was successfully booked",
                        },
                    },
                    "required": [
                        "schedulerGreetedProperly",
                        "schedulerCollectedName",
                        "schedulerConfirmedAppointment",
                        "appointmentBooked",
                        "serviceRequested",
                    ],
                },
            },
            "successEvaluationPlan": {
                "enabled": True,
                "rubric": "NumericScale",
            },
            "summaryPlan": {
                "enabled": True,
            },
        },
    }

    resp = requests.post(f"{BASE_URL}/assistant", headers=HEADERS, json=payload)
    resp.raise_for_status()
    assistant = resp.json()
    print(f"✅ Created simulated patient: {assistant['id']}")
    return assistant


# ─────────────────────────────────────────────
# Step 3: Assign dental scheduler to Phone Number A
# ─────────────────────────────────────────────

def assign_assistant_to_phone(phone_number_id, assistant_id):
    """Assign an assistant to handle inbound calls on a phone number."""
    payload = {
        "assistantId": assistant_id,
    }
    resp = requests.patch(
        f"{BASE_URL}/phone-number/{phone_number_id}",
        headers=HEADERS,
        json=payload,
    )
    resp.raise_for_status()
    print(f"✅ Assigned assistant {assistant_id} to phone {phone_number_id}")
    return resp.json()


# ─────────────────────────────────────────────
# Step 4: Make the call (Patient B calls Scheduler A)
# ─────────────────────────────────────────────

def make_test_call(patient_assistant_id, caller_phone_id, destination_number):
    """
    Initiate outbound call:
    - The simulated patient (assistant B) calls from phone B
    - To the dental scheduler's phone number (phone A)
    """
    payload = {
        "assistantId": patient_assistant_id,
        "phoneNumberId": caller_phone_id,
        "customer": {
            "number": destination_number,
        },
    }

    resp = requests.post(f"{BASE_URL}/call", headers=HEADERS, json=payload)
    if not resp.ok:
        print(f"❌ Call failed ({resp.status_code}): {resp.text}")
        resp.raise_for_status()
    call = resp.json()
    print(f"✅ Call initiated: {call['id']}")
    print(f"   Status: {call.get('status')}")
    return call


# ─────────────────────────────────────────────
# Step 5: Poll until call ends, then get results
# ─────────────────────────────────────────────

def wait_for_call(call_id, poll_interval=5, timeout=300):
    """Poll GET /call/{id} until the call ends."""
    print(f"\n⏳ Waiting for call {call_id} to complete...")
    start = time.time()

    while time.time() - start < timeout:
        resp = requests.get(f"{BASE_URL}/call/{call_id}", headers=HEADERS)
        resp.raise_for_status()
        call = resp.json()

        status = call.get("status")
        print(f"   Status: {status} ({int(time.time() - start)}s elapsed)")

        if status == "ended":
            return call

        time.sleep(poll_interval)

    raise TimeoutError(f"Call {call_id} did not end within {timeout}s")


def print_results(call):
    """Print the call results in a readable format."""
    print("\n" + "=" * 60)
    print("CALL RESULTS")
    print("=" * 60)

    # Basic info
    print(f"\nCall ID:      {call.get('id')}")
    print(f"Status:       {call.get('status')}")
    print(f"Ended Reason: {call.get('endedReason')}")
    print(f"Duration:     {call.get('startedAt')} → {call.get('endedAt')}")
    print(f"Cost:         ${call.get('cost', 'N/A')}")

    # Transcript
    transcript = call.get("artifact", {}).get("transcript", "")
    if transcript:
        print(f"\n{'─' * 40}")
        print("TRANSCRIPT")
        print(f"{'─' * 40}")
        print(transcript)

    # Analysis
    analysis = call.get("analysis", {})
    if analysis:
        print(f"\n{'─' * 40}")
        print("ANALYSIS")
        print(f"{'─' * 40}")

        if analysis.get("summary"):
            print(f"\nSummary: {analysis['summary']}")

        if analysis.get("structuredData"):
            print(f"\nStructured Data:")
            print(json.dumps(analysis["structuredData"], indent=2))

        if analysis.get("successEvaluation"):
            print(f"\nSuccess Evaluation: {analysis['successEvaluation']}")

    # Performance metrics
    perf = call.get("artifact", {}).get("performanceMetrics", {})
    if perf:
        print(f"\n{'─' * 40}")
        print("PERFORMANCE")
        print(f"{'─' * 40}")
        print(f"  Avg turn latency:   {perf.get('turnLatencyAverage', 'N/A')}ms")
        print(f"  Avg model latency:  {perf.get('modelLatencyAverage', 'N/A')}ms")
        print(f"  Avg voice latency:  {perf.get('voiceLatencyAverage', 'N/A')}ms")
        print(f"  User interruptions: {perf.get('numUserInterrupted', 'N/A')}")

    # Message count
    messages = call.get("artifact", {}).get("messages", [])
    print(f"\n  Total messages: {len(messages)}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    # Validate config
    if not API_KEY:
        print("❌ Set VAPI_API_KEY environment variable")
        return
    if not PHONE_NUMBER_A_ID or not PHONE_NUMBER_B_ID or not PHONE_NUMBER_A_NUMBER:
        print("❌ Set VAPI_PHONE_A_ID, VAPI_PHONE_B_ID, and VAPI_PHONE_A_NUMBER env variables")
        print("   Create two free phone numbers in the Vapi dashboard first.")
        return

    # Create both assistants
    scheduler = create_dental_scheduler()
    patient = create_simulated_patient()

    # Assign the dental scheduler to phone number A (inbound)
    assign_assistant_to_phone(PHONE_NUMBER_A_ID, scheduler["id"])

    # Make the call: patient (B) calls scheduler (A)
    call = make_test_call(
        patient_assistant_id=patient["id"],
        caller_phone_id=PHONE_NUMBER_B_ID,
        destination_number=PHONE_NUMBER_A_NUMBER,
    )

    # Wait for it to finish and print results
    completed_call = wait_for_call(call["id"])
    print_results(completed_call)

    # Duration penalty: fail if call exceeded 60 seconds
    started = completed_call.get("startedAt", "")
    ended = completed_call.get("endedAt", "")
    duration_seconds = None
    if started and ended:
        from datetime import datetime
        fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
        duration_seconds = (datetime.strptime(ended, fmt) - datetime.strptime(started, fmt)).total_seconds()
        duration_exceeded = duration_seconds > 60
        print(f"\n{'─' * 40}")
        print("DURATION CHECK")
        print(f"{'─' * 40}")
        print(f"  Duration: {duration_seconds:.1f}s")
        print(f"  Limit:    60s")
        print(f"  Result:   {'❌ FAIL — exceeded 1 minute' if duration_exceeded else '✅ PASS'}")

    # Save results to call_results/<call_id>/
    call_id = completed_call["id"]
    result_dir = Path("call_results") / call_id
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save raw JSON
    with open(result_dir / "call.json", "w") as f:
        json.dump(completed_call, f, indent=2)
    print(f"\n💾 Raw results saved to {result_dir}/call.json")

    # Save transcript
    transcript = completed_call.get("artifact", {}).get("transcript", "")
    if transcript:
        with open(result_dir / "transcript.txt", "w") as f:
            f.write(transcript)
        print(f"💾 Transcript saved to {result_dir}/transcript.txt")

    # Download recording
    recording_url = (
        completed_call.get("artifact", {}).get("recordingUrl")
        or completed_call.get("artifact", {}).get("recording")
    )
    if recording_url:
        print(f"⬇️  Downloading recording...")
        audio_resp = requests.get(recording_url)
        audio_resp.raise_for_status()
        content_type = audio_resp.headers.get("Content-Type", "")
        ext = ".wav" if "wav" in content_type else ".mp3"
        audio_path = result_dir / f"recording{ext}"
        with open(audio_path, "wb") as f:
            f.write(audio_resp.content)
        print(f"💾 Recording saved to {audio_path}")
    else:
        print("⚠️  No recording URL available")

    # Compute final verdict
    analysis = completed_call.get("analysis", {})
    vapi_score = analysis.get("successEvaluation")
    duration_exceeded = duration_seconds is not None and duration_seconds > 60
    final_pass = not duration_exceeded and vapi_score is not None and str(vapi_score) not in ("false", "0")

    verdict = {
        "call_id": completed_call["id"],
        "duration_seconds": duration_seconds,
        "duration_pass": not duration_exceeded,
        "vapi_score": vapi_score,
        "structured_data": analysis.get("structuredData"),
        "final_pass": final_pass,
    }
    with open(result_dir / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    print(f"\n{'=' * 40}")
    print(f"  FINAL VERDICT: {'✅ PASS' if final_pass else '❌ FAIL'}")
    duration_str = f"{duration_seconds:.1f}s" if duration_seconds is not None else "N/A"
    print(f"  Vapi Score: {vapi_score}/10 | Duration: {duration_str} {'(over limit)' if duration_exceeded else ''}")
    print(f"{'=' * 40}")

    # Print assistant IDs for later use
    print(f"\n📋 Save these IDs:")
    print(f"   Dental Scheduler ID: {scheduler['id']}")
    print(f"   Simulated Patient ID: {patient['id']}")
    print(f"   Results directory: {result_dir}")


if __name__ == "__main__":
    main()
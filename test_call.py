"""
Quick test call: triggers patient → scheduler call and prints results.
Uses existing assistants — no creation, no optimization.

Usage:
    python test_quick.py
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

VAPI_API_KEY = os.environ["VAPI_API_KEY"]
BASE = "https://api.vapi.ai"
HEADERS = {"Authorization": f"Bearer {VAPI_API_KEY}", "Content-Type": "application/json"}

PATIENT_ASSISTANT_ID = os.environ["PATIENT_ASSISTANT_ID"]
PATIENT_PHONE_ID = os.environ["VAPI_PHONE_B_ID"]
SCHEDULER_PHONE = os.environ["VAPI_PHONE_A_NUMBER"]


def main():
    # Initiate call
    print("📞 Starting test call...")
    resp = requests.post(f"{BASE}/call", headers=HEADERS, json={
        "assistantId": PATIENT_ASSISTANT_ID,
        "phoneNumberId": PATIENT_PHONE_ID,
        "customer": {"number": SCHEDULER_PHONE},
    })
    resp.raise_for_status()
    call_id = resp.json()["id"]
    print(f"   Call ID: {call_id}")

    # Poll
    print("⏳ Waiting for call to end...")
    while True:
        time.sleep(5)
        r = requests.get(f"{BASE}/call/{call_id}", headers=HEADERS)
        r.raise_for_status()
        call = r.json()
        if call.get("status") == "ended":
            break
        print(f"   ...{call.get('status')}")

    # Results
    analysis = call.get("analysis", {})
    sd = analysis.get("structuredData", {})
    transcript = call.get("artifact", {}).get("transcript", call.get("transcript", ""))

    # Duration
    try:
        t1 = datetime.fromisoformat(call["startedAt"].replace("Z", "+00:00"))
        t2 = datetime.fromisoformat(call["endedAt"].replace("Z", "+00:00"))
        duration = (t2 - t1).total_seconds()
    except Exception:
        duration = 0

    checklist = [
        sd.get("schedulerGreetedProperly", False),
        sd.get("schedulerCollectedName", False),
        sd.get("schedulerOfferedTimes", False),
        sd.get("schedulerProvidedPricing", False),
        sd.get("schedulerConfirmedAppointment", False),
        sd.get("appointmentBooked", False),
    ]

    print(f"\n{'=' * 50}")
    print("RESULTS")
    print(f"{'=' * 50}")
    print(f"\nTranscript:\n{transcript}")
    print(f"\n{'─' * 50}")
    print(f"  Checklist:  {sum(checklist)}/6  {checklist}")
    print(f"  Vapi Score: {analysis.get('successEvaluation', 'N/A')}")
    print(f"  Booked:     {sd.get('appointmentBooked', False)}")
    print(f"  Duration:   {duration:.0f}s")
    print(f"  Ended:      {call.get('endedReason', 'N/A')}")
    print(f"  Cost:       ${call.get('cost', 0):.4f}")
    print(f"{'=' * 50}")

    # Save to call_results/<call_id>/
    result_dir = Path("call_results") / call_id
    result_dir.mkdir(parents=True, exist_ok=True)

    # Raw API response
    with open(result_dir / "call.json", "w") as f:
        json.dump(call, f, indent=2)
    print(f"\n💾 Raw results saved to {result_dir}/call.json")

    # Transcript
    if transcript:
        with open(result_dir / "transcript.txt", "w") as f:
            f.write(transcript)
        print(f"💾 Transcript saved to {result_dir}/transcript.txt")

    # Download recording
    recording_url = (
        call.get("artifact", {}).get("recordingUrl")
        or call.get("artifact", {}).get("recording")
        or call.get("recordingUrl")
    )
    if recording_url:
        print("⬇️  Downloading recording...")
        audio_resp = requests.get(recording_url)
        audio_resp.raise_for_status()
        content_type = audio_resp.headers.get("Content-Type", "")
        ext = ".wav" if "wav" in content_type else ".mp3"
        with open(result_dir / f"recording{ext}", "wb") as f:
            f.write(audio_resp.content)
        print(f"💾 Recording saved to {result_dir}/recording{ext}")
    else:
        print("⚠️  No recording URL available")

    # Verdict
    duration_exceeded = duration > 60
    with open(result_dir / "verdict.json", "w") as f:
        json.dump({
            "call_id": call_id,
            "checklist": checklist,
            "checklist_score": f"{sum(checklist)}/6",
            "vapi_score": analysis.get("successEvaluation"),
            "structured_data": sd,
            "duration_seconds": duration,
            "duration_pass": not duration_exceeded,
            "ended_reason": call.get("endedReason"),
            "cost": call.get("cost", 0),
            "final_pass": not duration_exceeded and sd.get("appointmentBooked", False),
        }, f, indent=2)
    print(f"💾 Verdict saved to {result_dir}/verdict.json")


if __name__ == "__main__":
    main()
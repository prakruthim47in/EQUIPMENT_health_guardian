# Equipment Health Guardian

A predictive maintenance dashboard built for the **Student Track: Agentic
Data Analysis** (ADK). Simulated industrial sensors (vibration, temperature,
RPM) across a small machine fleet feed an IsolationForest anomaly detector;
when something looks wrong, an ADK agent calls its own tool to fetch the
machine's data and produces a plain-English diagnosis + recommended action.

## Quick start (5 minutes)

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Get a free Gemini API key: https://aistudio.google.com/apikey
3. Run the app:
   ```
   streamlit run app.py
   ```
4. Paste your API key into the sidebar. Pick a machine + fault type in the
   sidebar to simulate a problem, then click "Generate diagnosis" and try
   the chat box — watch the agent call its tool before answering.

### Optional: route through Vertex AI instead of a plain API key
If you've redeemed your GCP trial credits and want the submission to show
Vertex AI usage, set these environment variables before running Streamlit
(no code changes needed):
```
export GOOGLE_GENAI_USE_VERTEXAI=1
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us-central1
```

## How it's built (for your submission writeup)

- `simulator.py` — generates realistic multi-machine sensor time series with
  injectable fault patterns (bearing wear, overheating, imbalance).
- `detector.py` — IsolationForest anomaly detection on rolling-window
  statistical features, trained per machine.
- `ai_assistant.py` — the **ADK agent**. It's given one tool,
  `get_machine_sensor_data`, which it decides to call on its own whenever a
  technician asks about a machine — this tool-use decision loop is what
  makes it "agentic" rather than a single wrapped prompt.
- `app.py` — Streamlit dashboard tying it all together.

## A note on which track/codelab this matches

Based on the program email, this project is built for the **Student Track:
Agentic Data Analysis**, whose mandatory codelab is the **ADK Crash Course**
— which is exactly what `ai_assistant.py` implements (an `Agent` with a
tool, run via ADK's runner). The Professional Tracks (BigQuery conversational
agents / BigQuery AI-assisted data science) are separate tracks aimed at a
different codelab — you shouldn't need those unless your program instructions
say otherwise. Double-check your actual portal/email if anything here reads
ambiguously — screenshots don't always show the full instructions.

## Things to customize before submission (do these — judges notice generic demos)

- [ ] Rename machines to match a specific real-world scenario you're pitching
      (e.g., a named factory, a specific industry — textile, steel, food
      processing — pick one and commit to it in your pitch).
- [ ] Add a one-line "business case" to the README/pitch: estimated downtime
      cost avoided, or safety incident prevented, if this caught the fault early.
- [ ] If you have any real sensor datasets available (Kaggle's NASA bearing
      dataset, CWRU bearing dataset, etc.) and time permits, swap in real data
      for extra credibility — but don't risk your working demo over this if
      time is short. A believable synthetic demo that works is better than
      a real dataset demo that's half-broken.
- [ ] Complete the ADK Crash Course codelab and take the submission screenshot
      it asks for (ADK web UI at localhost:8000 + terminal) — that's your
      mandatory milestone, separate from the hackathon demo itself.

## Recording your demo video (3 min)

1. (0:00–0:20) State the problem: unplanned industrial downtime is expensive;
   most small/mid factories can't afford enterprise predictive-maintenance suites.
2. (0:20–1:30) Show the healthy dashboard, then trigger a fault in the sidebar
   and show the alert appear.
3. (1:30–2:30) Click "Generate diagnosis," show the AI's explanation, then ask
   it a follow-up question live in the chat box.
4. (2:30–3:00) Close with impact + what you'd add with more time (real sensor
   integration, SMS/email alerts, multi-factory fleet view).

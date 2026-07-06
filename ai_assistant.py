"""
Multi-agent layer built with Google's Agent Development Kit (ADK).

This is deliberately a small MULTI-AGENT system, not a single agent, to
match the "Agentic AI (Design to Deployment)" track's requirement of
"multi-agent AI ecosystems that independently execute tasks... and
automate multi-step workflows":

    maintenance_orchestrator (root agent)
        |-- diagnostic_agent   -> calls get_machine_sensor_data
        |-- action_agent       -> calls estimate_downtime_and_priority

The orchestrator delegates: first to the diagnostic agent to understand
what's wrong, then to the action agent to turn that into a prioritized,
cost-quantified action plan. That's the "multi-step workflow" the track
is asking for.

Setup:
  Set the GOOGLE_API_KEY environment variable with a free key from
  https://aistudio.google.com/apikey (or export it before running streamlit).

Optional (only if your track requires it — check the Tech Stack Rule):
  Route through Vertex AI instead of a plain API key by setting:
    GOOGLE_GENAI_USE_VERTEXAI=1
    GOOGLE_CLOUD_PROJECT=<your-project-id>
    GOOGLE_CLOUD_LOCATION=us-central1
  No code changes needed — ADK picks this up automatically.
"""
import asyncio
import pandas as pd
from google.adk import Agent
from google.adk.runners import InMemoryRunner

MODEL_NAME = "gemini-2.5-flash"

# Holds the current fleet dataframe so tools can read it.
# Set this once per Streamlit rerun via set_fleet_data(df).
_fleet_df: pd.DataFrame | None = None
_runner: InMemoryRunner | None = None

# Rough cost-per-hour-of-downtime by urgency, used to make the action
# agent's output concrete rather than generic advice.
_DOWNTIME_COST_PER_HOUR = {"Low": 150, "Medium": 600, "High": 2200, "Critical": 6000}


def set_fleet_data(df: pd.DataFrame) -> None:
    """Call this once per app refresh so agent tools see current data."""
    global _fleet_df
    _fleet_df = df


# ---------- Tool 1: used by the diagnostic sub-agent ----------
def get_machine_sensor_data(machine_name: str) -> dict:
    """Returns the latest sensor readings and recent history for a machine.

    Args:
        machine_name: The name of the machine to look up, e.g. "Motor-A1".
    """
    if _fleet_df is None:
        return {"error": "No sensor data loaded yet."}

    machine_df = _fleet_df[_fleet_df["machine"] == machine_name]
    if machine_df.empty:
        return {"error": f"No data found for machine '{machine_name}'."}

    machine_df = machine_df.sort_values("t")
    latest = machine_df.iloc[-1]
    recent = machine_df.tail(10)[
        ["t", "vibration_mm_s", "temperature_c", "rpm", "is_anomaly"]
    ].to_dict(orient="records")

    return {
        "machine": machine_name,
        "latest_vibration_mm_s": float(latest["vibration_mm_s"]),
        "latest_temperature_c": float(latest["temperature_c"]),
        "latest_rpm": float(latest["rpm"]),
        "is_currently_anomalous": bool(latest["is_anomaly"]),
        "recent_history": recent,
    }


# ---------- Tool 2: used by the action-planning sub-agent ----------
def estimate_downtime_and_priority(machine_name: str, urgency: str, estimated_repair_hours: float) -> dict:
    """Estimates downtime cost and gives a maintenance priority ranking.

    Args:
        machine_name: The machine being assessed.
        urgency: One of "Low", "Medium", "High", "Critical".
        estimated_repair_hours: Estimated hours of downtime to fix the issue.
    """
    rate = _DOWNTIME_COST_PER_HOUR.get(urgency, 300)
    estimated_cost = round(rate * max(estimated_repair_hours, 0.5), 2)
    priority_rank = {"Critical": 1, "High": 2, "Medium": 3, "Low": 4}.get(urgency, 3)
    return {
        "machine": machine_name,
        "urgency": urgency,
        "estimated_downtime_cost_usd": estimated_cost,
        "priority_rank": priority_rank,  # 1 = fix first
    }


def _build_root_agent() -> Agent:
    diagnostic_agent = Agent(
        model=MODEL_NAME,
        name="diagnostic_agent",
        description="Fetches sensor data for a machine and diagnoses the likely fault.",
        instruction=(
            "Call get_machine_sensor_data for the machine in question, then explain "
            "in plain English what is likely going wrong and how urgent it is "
            "(Low / Medium / High / Critical)."
        ),
        tools=[get_machine_sensor_data],
    )

    action_agent = Agent(
        model=MODEL_NAME,
        name="action_planner_agent",
        description="Turns a diagnosis into a prioritized, cost-quantified maintenance action plan.",
        instruction=(
            "Given a machine, its urgency level, and a rough repair-time estimate in hours, "
            "call estimate_downtime_and_priority to quantify the impact, then produce a short, "
            "concrete action plan a technician can act on immediately."
        ),
        tools=[estimate_downtime_and_priority],
    )

    return Agent(
        model=MODEL_NAME,
        name="maintenance_orchestrator",
        description="Coordinates diagnosis and action planning for factory machines.",
        instruction=(
            "You coordinate two specialists to help a maintenance technician. "
            "First delegate to diagnostic_agent to understand what's wrong with the machine "
            "and its urgency. Then delegate to action_planner_agent, passing along the "
            "urgency and a reasonable repair-time estimate in hours, to get a cost-quantified "
            "action plan. Combine both results into one clear final answer under 120 words: "
            "diagnosis, urgency, estimated cost, and the recommended action."
        ),
        sub_agents=[diagnostic_agent, action_agent],
    )


def _get_runner() -> InMemoryRunner:
    global _runner
    if _runner is None:
        _runner = InMemoryRunner(agent=_build_root_agent(), app_name="equipment_guardian")
    return _runner


async def _ask_async(question: str) -> str:
    runner = _get_runner()
    events = await runner.run_debug(question, quiet=True)
    texts = []
    for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    texts.append(part.text)
    return "\n".join(texts) if texts else "The agent didn't return a response."


def ask_agent(machine_name: str, question: str) -> str:
    """Send a question about a specific machine to the multi-agent system.

    The orchestrator delegates to the diagnostic and action-planning
    sub-agents itself — you don't need to call them individually.
    """
    full_question = (
        f"The technician is asking about machine '{machine_name}'. "
        f"Their question: {question}"
    )
    return asyncio.run(_ask_async(full_question))

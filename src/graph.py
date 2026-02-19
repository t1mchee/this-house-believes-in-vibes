"""
LangGraph pipeline definition.

Wires together all phases into a single executable graph:
  Phase 0 (ingest) → Phase 1 (prepare) → Phase 2 (debate) → Phase 3 (division) → Phase 4 (refine?)
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

from langgraph.graph import END, StateGraph
from rich.console import Console
from typing_extensions import TypedDict

from src.corpus.ingest import retrieve_for_topics
from src.debate.speech import (
    build_definitions_context,
    extract_contestation,
    extract_definitions,
    format_transcript,
    generate_pois,
    generate_speech,
)
from src.models import (
    DebateRun,
    DefinitionsContestation,
    DefinitionsFrame,
    DivisionResult,
    POI,
    Side,
    SpeakerData,
    SpeakerProfile,
    SpeechOutput,
    StyleProfile,
)
from src.persona.builder import prepare_all_speakers

# Suppress noisy Pydantic serialization warnings from LangGraph internals
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class DebateGraphState(TypedDict):
    """State that flows through the LangGraph pipeline."""

    # Configuration (set once)
    motion: str
    prop_speakers: list[SpeakerProfile]
    opp_speakers: list[SpeakerProfile]

    # Phase 0 outputs
    styles: dict[str, StyleProfile]  # speaker_id -> style profile

    # Ensemble variation
    strategy_directives: dict[str, str]  # speaker_id -> strategic angle (optional)
    passage_overrides: Optional[dict[str, list[str]]]  # speaker_id -> pre-selected passages

    # Phase 1 outputs
    speaker_data: dict[str, SpeakerData]  # speaker_id -> full persona + prep

    # Phase 2 outputs (accumulated)
    speeches: list[SpeechOutput]
    pois: list[POI]
    current_speech_index: int  # 0-5, tracks which speech we're on

    # Definitions framework (extracted after speeches 1 and 2)
    definitions: Optional[DefinitionsFrame]
    contestation: Optional[DefinitionsContestation]
    definitions_context: str  # formatted text block for injection

    # Phase 3 outputs
    division: Optional[DivisionResult]
    verdict_raw: str  # raw LLM verdict text

    # Phase 4 — refinement
    iteration: int
    history: list[DebateRun]
    should_terminate: bool


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def prepare_node(state: DebateGraphState) -> dict[str, Any]:
    """Phase 1: Prepare all speakers in parallel."""
    all_speakers = state["prop_speakers"] + state["opp_speakers"]

    speaker_data = await prepare_all_speakers(
        speakers=all_speakers,
        styles=state["styles"],
        motion=state["motion"],
        strategy_directives=state.get("strategy_directives"),
        passage_overrides=state.get("passage_overrides"),
    )

    return {
        "speaker_data": speaker_data,
        "speeches": [],
        "pois": [],
        "current_speech_index": 0,
        "definitions": None,
        "contestation": None,
        "definitions_context": "",
    }


async def speech_node(state: DebateGraphState) -> dict[str, Any]:
    """Generate one speech + its POIs, then advance the index."""
    idx = state["current_speech_index"]

    # Interleave: Prop1(0), Opp1(1), Prop2(2), Opp2(3), Prop3(4), Opp3(5)
    speaking_order = [
        state["prop_speakers"][0],
        state["opp_speakers"][0],
        state["prop_speakers"][1],
        state["opp_speakers"][1],
        state["prop_speakers"][2],
        state["opp_speakers"][2],
    ]

    current_speaker = speaking_order[idx]
    sd = state["speaker_data"][current_speaker.id]

    # Optionally refresh RAG based on debate topics so far
    if idx > 0:
        topics = []
        for s in state["speeches"]:
            for arg in s.arguments:
                topics.append(arg.claim)
        additional = await retrieve_for_topics(current_speaker, topics[:5], k_per_query=3)
        sd.retrieved_passages = list(set(sd.retrieved_passages + additional))

    # Generate the speech — pass definitions context for speeches 2+
    speech = await generate_speech(
        speaker_data=sd,
        position_in_order=idx + 1,
        motion=state["motion"],
        prior_speeches=state["speeches"],
        prior_pois=state["pois"],
        definitions_context=state.get("definitions_context", ""),
    )

    # Generate POIs for this speech
    if current_speaker.side == Side.PROPOSITION:
        opposing_data = [state["speaker_data"][s.id] for s in state["opp_speakers"]]
    else:
        opposing_data = [state["speaker_data"][s.id] for s in state["prop_speakers"]]

    new_pois = await generate_pois(
        speech=speech,
        opposing_speakers=opposing_data,
        receiving_speaker=sd,
        all_pois_so_far=state["pois"],
    )

    updated_speeches = state["speeches"] + [speech]

    # --- Extract definitions after key speeches ---
    result: dict[str, Any] = {
        "speeches": updated_speeches,
        "pois": state["pois"] + new_pois,
        "current_speech_index": idx + 1,
    }

    if idx == 0:
        # After Prop 1: extract the definitional framework
        try:
            definitions = await extract_definitions(speech, state["motion"])
            defs_context = build_definitions_context(definitions)
            result["definitions"] = definitions
            result["definitions_context"] = defs_context
        except Exception as e:
            print(f"  ⚠️  Definitions extraction failed: {e}")

    elif idx == 1 and state.get("definitions"):
        # After Opp 1: extract their response to the definitions
        try:
            contestation = await extract_contestation(
                speech, state["definitions"], state["motion"]
            )
            result["contestation"] = contestation
            # Rebuild context with the contestation included
            result["definitions_context"] = build_definitions_context(
                state["definitions"], contestation
            )
        except Exception as e:
            print(f"  ⚠️  Contestation extraction failed: {e}")

    return result


def should_continue_debate(state: DebateGraphState) -> str:
    """Route: more speeches or move to division?"""
    if state["current_speech_index"] < 6:
        return "next_speech"
    return "division"


async def division_node(state: DebateGraphState) -> dict[str, Any]:
    """Phase 3: Three-layer judging — rubric, panel, argument audit."""
    from src.debate.judge import run_division

    console = Console(width=120)
    console.print("\n[bold yellow]══ PHASE 3: THREE-LAYER JUDGING ══[/bold yellow]\n")
    console.print(
        "  Running [cyan]Rubric Scoring[/cyan] + "
        "[cyan]Judge Panel[/cyan] + "
        "[cyan]Argument Audit[/cyan] in parallel…\n"
    )

    division, raw_verdict = await run_division(
        speeches=state["speeches"],
        pois=state["pois"],
        motion=state["motion"],
        definitions_context=state.get("definitions_context", ""),
    )

    console.print(f"  ✅ Verdict: [bold]{division.winner.value.upper()}[/bold] "
                  f"({division.margin}, {division.ayes}-{division.noes})\n")

    current_run = DebateRun(
        iteration=state["iteration"],
        speeches=state["speeches"],
        pois=state["pois"],
        division=division,
    )

    return {
        "division": division,
        "verdict_raw": raw_verdict,
        "history": state["history"] + [current_run],
    }


def should_refine(state: DebateGraphState) -> str:
    """Decide whether to run another refinement iteration."""
    if state["iteration"] >= 3:
        return "end"

    # TODO: implement convergence checks (verdict stability, argument novelty)
    return "end"  # Default to single run for now


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_debate_graph() -> StateGraph:
    """Construct and compile the debate simulation graph."""

    graph = StateGraph(DebateGraphState)

    # Nodes
    graph.add_node("prepare", prepare_node)
    graph.add_node("speech", speech_node)
    graph.add_node("division", division_node)

    # Edges
    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "speech")

    # Speech loop: after each speech, check if we need more
    graph.add_conditional_edges(
        "speech",
        should_continue_debate,
        {
            "next_speech": "speech",
            "division": "division",
        },
    )

    # After division, check for refinement
    graph.add_conditional_edges(
        "division",
        should_refine,
        {
            "refine": "prepare",  # loop back
            "end": END,
        },
    )

    return graph.compile()

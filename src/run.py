"""
Entry point: Cambridge Union Lent Term 2026, Fifth Debate

Motion: "This House Believes AI Should Be Allowed To Make Decisions About Human Life"

Speaking order:
  Prop 1: Dr Henry Shevlin
  Opp 1:  Dr Fazl Barez
  Prop 2: Student Speaker (Proposition)
  Opp 2:  Allison Gardner MP
  Prop 3: Student Speaker (Proposition)
  Opp 3:  Demetrius Floudas

Usage:
    cd project_debater_2
    source .venv/bin/activate
    python -m src.run
"""

from __future__ import annotations

import asyncio
import json
import warnings
from datetime import datetime
from pathlib import Path

from src.corpus.ingest import ingest_speaker_corpus, extract_style_profile, load_documents_from_directory
from src.graph import DebateGraphState, build_debate_graph
from src.models import DivisionResult, Side, SpeakerProfile, StyleProfile

# Suppress noisy Pydantic serialization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


# ---------------------------------------------------------------------------
# Debate configuration
# ---------------------------------------------------------------------------

MOTION = (
    "This House Believes AI Should Be Allowed To Make "
    "Decisions About Human Life"
)

PROP_SPEAKERS = [
    SpeakerProfile(
        id="henry_shevlin",
        name="Dr Henry Shevlin",
        side=Side.PROPOSITION,
        speaking_position=1,
        bio=(
            "Dr Henry Shevlin is an AI ethicist and philosopher of cognitive "
            "science, and Associate Director of the Leverhulme Centre for the "
            "Future of Intelligence at the University of Cambridge. His research "
            "focuses on AI consciousness, moral status, and the ethics of our "
            "relationships with machines. He is a widely published philosopher "
            "with 20+ publications in first-rate academic journals."
        ),
        corpus_collection="henry_shevlin_corpus",
    ),
    SpeakerProfile(
        id="student_prop_2",
        name="Student Speaker (Prop 2)",
        side=Side.PROPOSITION,
        speaking_position=2,
        bio=(
            "A Cambridge student speaking for the Proposition. This speaker "
            "focuses on the PHILOSOPHICAL and TECHNICAL case: comparative "
            "risk analysis (human error vs AI error), the accountability "
            "dividend of auditable code, and existing governance frameworks "
            "(EU AI Act, FDA-cleared devices, ISO 42001). Their style is "
            "evidence-dense, fast-paced, and forensic — dismantling "
            "opposition claims with data."
        ),
        corpus_collection="student_speaker_corpus",
    ),
    SpeakerProfile(
        id="student_prop_3",
        name="Student Speaker (Prop 3)",
        side=Side.PROPOSITION,
        speaking_position=3,
        bio=(
            "A Cambridge student delivering the FINAL Proposition speech. "
            "This speaker focuses on the MORAL and GEOPOLITICAL case: "
            "distributive justice (global health equity, low-income country "
            "access), the arms-race dynamics of AI non-deployment, and the "
            "democratic imperative to govern rather than prohibit. Their "
            "style is impassioned and persuasive — building to a powerful "
            "peroration that drives the audience to vote. Must advance "
            "arguments NOT already made by earlier Proposition speakers."
        ),
        corpus_collection="student_speaker_corpus",
    ),
]

OPP_SPEAKERS = [
    SpeakerProfile(
        id="fazl_barez",
        name="Dr Fazl Barez",
        side=Side.OPPOSITION,
        speaking_position=1,
        bio=(
            "Dr Fazl Barez is a Senior Research Fellow at the University of "
            "Oxford, leading research on Technical AI Safety, Interpretability, "
            "and Governance. He is affiliated with Cambridge's Centre for the "
            "Study of Existential Risk and has worked with Anthropic's Alignment "
            "team on research investigating deception in language models."
        ),
        corpus_collection="fazl_barez_corpus",
    ),
    SpeakerProfile(
        id="allison_gardner",
        name="Allison Gardner MP",
        side=Side.OPPOSITION,
        speaking_position=2,
        bio=(
            "Allison Gardner is the Labour MP for Stoke-on-Trent South. She is "
            "an expert in AI and Data Ethics with interests in health technology, "
            "algorithmic bias, diversity and inclusion. She is a co-founder and "
            "director of Women Leading in AI, and previously worked with the NHS "
            "as a Senior Scientific Adviser for Artificial Intelligence."
        ),
        corpus_collection="allison_gardner_corpus",
    ),
    SpeakerProfile(
        id="demetrius_floudas",
        name="Demetrius Floudas",
        side=Side.OPPOSITION,
        speaking_position=3,
        bio=(
            "Demetrius Floudas is an AI policy strategist, geopolitical adviser "
            "and lawyer specialising in AI Governance. He is Visiting Scholar at "
            "the Leverhulme Centre for the Future of Intelligence and Senior "
            "Adviser to the Cambridge Existential Risk Initiative. He treats "
            "advanced AI as a civilisational-level risk and advocates for an "
            "international AI Control & Non-Proliferation Treaty."
        ),
        corpus_collection="demetrius_floudas_corpus",
    ),
]

# Map speaker IDs to their data directories
SPEAKER_DATA_DIRS = {
    "henry_shevlin": "data/speakers/Henry Shevlin",
    "fazl_barez": "data/speakers/Fazl Barez",
    "allison_gardner": "data/speakers/Allison Gardner",
    "demetrius_floudas": "data/speakers/Demetrius Floudas",
    "student_prop_2": "data/speakers/Student Speaker",
    "student_prop_3": "data/speakers/Student Speaker",  # shared corpus
}


async def run_phase_0(
    all_speakers: list[SpeakerProfile],
    console,
) -> dict[str, StyleProfile]:
    """
    Phase 0: Ingest each speaker's corpus into ChromaDB and extract style profiles.

    Returns a dict mapping speaker_id -> StyleProfile.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    styles: dict[str, StyleProfile] = {}
    ingested_collections: set[str] = set()  # avoid double-ingesting shared corpora

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for speaker in all_speakers:
            data_dir = Path(SPEAKER_DATA_DIRS[speaker.id])

            # ---- Ingest into ChromaDB (skip if collection already done) ----
            if speaker.corpus_collection not in ingested_collections:
                task = progress.add_task(
                    f"Ingesting {speaker.name} ({data_dir.name})…", total=None
                )
                n_chunks = await ingest_speaker_corpus(speaker, data_dir)
                ingested_collections.add(speaker.corpus_collection)
                progress.update(task, completed=True)
                console.print(
                    f"  ✅ [bold]{speaker.name}[/bold]: {n_chunks} chunks → "
                    f"[cyan]{speaker.corpus_collection}[/cyan]"
                )
            else:
                console.print(
                    f"  ♻️  [bold]{speaker.name}[/bold]: reusing "
                    f"[cyan]{speaker.corpus_collection}[/cyan]"
                )

            # ---- Extract style profile ----
            if speaker.id not in styles:
                task = progress.add_task(
                    f"Extracting style for {speaker.name}…", total=None
                )
                # Build a corpus sample from the first ~8 000 chars of their docs
                docs = load_documents_from_directory(data_dir)
                corpus_sample = "\n\n---\n\n".join(
                    d["text"][:2000] for d in docs[:6]
                )[:8000]

                style = await extract_style_profile(speaker, corpus_sample)
                styles[speaker.id] = style
                progress.update(task, completed=True)
                console.print(
                    f"  🎨 Style → register: [italic]{style.speech_register}[/italic]"
                )

    # Student speakers share corpus but need distinct style entries
    # (both point to same data, but each gets their own key)
    for speaker in all_speakers:
        if speaker.id not in styles:
            # Inherit from the sibling who was already extracted
            donor_id = [
                s.id for s in all_speakers
                if s.corpus_collection == speaker.corpus_collection and s.id in styles
            ][0]
            styles[speaker.id] = styles[donor_id]
            console.print(
                f"  ♻️  [bold]{speaker.name}[/bold]: inheriting style from "
                f"{donor_id}"
            )

    return styles


async def main():
    """Run a full debate simulation."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console(width=120)

    # Header
    console.print()
    console.print(Panel(
        f"[bold]{MOTION}[/bold]\n\n"
        "[dim]Cambridge Union · Lent Term 2026 · The Fifth Debate · "
        "Thursday 19th February, 8pm[/dim]",
        title="🏛️  MOTION FOR DEBATE",
        border_style="red",
    ))

    # Speaker lineup
    table = Table(title="Speaking Order", show_header=True)
    table.add_column("Order", style="bold", width=6)
    table.add_column("Side", width=12)
    table.add_column("Speaker", width=30)
    table.add_column("Role", width=20)

    speaking_order = [
        PROP_SPEAKERS[0], OPP_SPEAKERS[0],
        PROP_SPEAKERS[1], OPP_SPEAKERS[1],
        PROP_SPEAKERS[2], OPP_SPEAKERS[2],
    ]
    for i, s in enumerate(speaking_order):
        side_label = "[green]PROPOSITION[/green]" if s.side == Side.PROPOSITION else "[red]OPPOSITION[/red]"
        role = f"Speaker {s.speaking_position}/3 on side"
        table.add_row(str(i + 1), side_label, s.name, role)

    console.print(table)

    # ---------------------------------------------------------------
    # PHASE 0: Corpus ingestion & style extraction
    # ---------------------------------------------------------------
    console.print("\n[bold blue]══ PHASE 0: CORPUS INGESTION ══[/bold blue]\n")
    all_speakers = PROP_SPEAKERS + OPP_SPEAKERS
    styles = await run_phase_0(all_speakers, console)
    console.print(f"\n  📚 Ingested & styled {len(styles)} speakers.\n")

    # ---------------------------------------------------------------
    # PHASES 1-3: Build graph & run debate
    # ---------------------------------------------------------------
    console.print("[bold blue]══ PHASE 1→3: DEBATE SIMULATION ══[/bold blue]\n")

    graph = build_debate_graph()

    initial_state: DebateGraphState = {
        "motion": MOTION,
        "prop_speakers": PROP_SPEAKERS,
        "opp_speakers": OPP_SPEAKERS,
        "styles": styles,
        "strategy_directives": {},  # No strategic emphasis for single runs
        "passage_overrides": None,  # Use default RAG retrieval for single runs
        "speaker_data": {},
        "speeches": [],
        "pois": [],
        "current_speech_index": 0,
        "definitions": None,
        "contestation": None,
        "definitions_context": "",
        "division": None,
        "verdict_raw": "",
        "iteration": 1,
        "history": [],
        "should_terminate": False,
    }

    console.print("[dim]Preparing speakers & generating debate…[/dim]\n")
    result = await graph.ainvoke(initial_state)

    # ---------------------------------------------------------------
    # Output: Debate Transcript
    # ---------------------------------------------------------------
    console.print("\n[bold blue]══ DEBATE TRANSCRIPT ══[/bold blue]\n")

    for speech in result["speeches"]:
        side = "[green]PROP[/green]" if speech.side == Side.PROPOSITION else "[red]OPP[/red]"
        console.print(Panel(
            speech.full_text,
            title=f"[bold]{speech.speaker_name}[/bold] ({side})",
            subtitle=f"{speech.word_count} words · Tone: {speech.tone}",
            border_style="green" if speech.side == Side.PROPOSITION else "red",
        ))

        # Print any POIs during this speech
        speech_pois = [p for p in result["pois"] if p.to_speaker == speech.speaker_name]
        for poi in speech_pois:
            status = "✅ ACCEPTED" if poi.accepted else "❌ DECLINED"
            console.print(f"  [yellow]📢 POI from {poi.from_speaker}[/yellow] [{status}]")
            console.print(f'     "{poi.text}"')
            if poi.accepted and poi.response:
                console.print(f"     → {poi.response}")

    # ---------------------------------------------------------------
    # Output: The Division (Three-Layer Verdict)
    # ---------------------------------------------------------------
    console.print("\n[bold green]═══════════════════════════════════════[/bold green]")
    console.print("[bold green]           THE DIVISION[/bold green]")
    console.print("[bold green]═══════════════════════════════════════[/bold green]\n")

    division = result.get("division")
    verdict_raw = result.get("verdict_raw", "")

    if division:
        winner_label = (
            "[green]PROPOSITION (AYE)[/green]"
            if division.winner == Side.PROPOSITION
            else "[red]OPPOSITION (NO)[/red]"
        )
        console.print(f"  Result: {winner_label} by a [bold]{division.margin}[/bold] margin\n")

        # ── Layer 1: Rubric Scorecard ──
        if division.rubric:
            console.print("  [bold cyan]── LAYER 1: ANALYTICAL RUBRIC ──[/bold cyan]")
            rubric_table = Table(show_header=True, box=None, padding=(0, 1))
            rubric_table.add_column("Speaker", width=28)
            rubric_table.add_column("Side", width=5)
            rubric_table.add_column("Arg", width=5, justify="center")
            rubric_table.add_column("Reb", width=5, justify="center")
            rubric_table.add_column("Evd", width=5, justify="center")
            rubric_table.add_column("Rht", width=5, justify="center")
            rubric_table.add_column("Per", width=5, justify="center")
            rubric_table.add_column("OVR", width=5, justify="center", style="bold")
            rubric_table.add_column("Rationale", width=50)

            for s in division.rubric.scores:
                side_str = "[green]P[/green]" if s.side == Side.PROPOSITION else "[red]O[/red]"
                rubric_table.add_row(
                    s.speaker_name, side_str,
                    f"{s.argument_strength:.0f}", f"{s.rebuttal_quality:.0f}",
                    f"{s.evidence_grounding:.0f}", f"{s.rhetorical_effectiveness:.0f}",
                    f"{s.persona_fidelity:.0f}", f"{s.overall:.0f}",
                    s.rationale[:50] + ("…" if len(s.rationale) > 50 else ""),
                )
            console.print(rubric_table)

            prop_label = "[green]PROP[/green]" if division.rubric.rubric_winner == Side.PROPOSITION else "PROP"
            opp_label = "[red]OPP[/red]" if division.rubric.rubric_winner == Side.OPPOSITION else "OPP"
            console.print(
                f"\n  {prop_label}: {division.rubric.prop_total:.1f}  |  "
                f"{opp_label}: {division.rubric.opp_total:.1f}  →  "
                f"[bold]{division.rubric.rubric_winner.value.upper()}[/bold]\n"
            )

        # ── Layer 2b: Engagement Verdict (PRIMARY) ──
        if division.engagement:
            eng = division.engagement
            console.print("  [bold cyan]── LAYER 2b: ENGAGEMENT-FOCUSED VERDICT (PRIMARY) ──[/bold cyan]")
            prop_label = "[green]PROP[/green]" if eng.winner == Side.PROPOSITION else "PROP"
            opp_label = "[red]OPP[/red]" if eng.winner == Side.OPPOSITION else "OPP"
            pass_str = "[green]✓ Both passes agree[/green]" if eng.pass_agreement else "[yellow]⚠ Passes DISAGREE[/yellow]"
            console.print(
                f"  Votes: {prop_label} {eng.prop_votes} – {opp_label} {eng.opp_votes}  |  "
                f"{pass_str}  |  "
                f"Confidence: {eng.mean_confidence:.2f}  →  "
                f"[bold]{eng.winner.value.upper()}[/bold] ({eng.margin})\n"
            )
            for i, v in enumerate(eng.votes, 1):
                pass_label = "Pass 1" if i <= 3 else "Pass 2"
                console.print(f"  Judge {i} ({pass_label}): {v.better_team} (conf: {v.confidence:.2f})")
                console.print(f"    Key reason: {v.key_reason[:100]}{'…' if len(v.key_reason) > 100 else ''}")
            console.print("")

        # ── Layer 2a: Annotation-Based Verdict ──
        if division.annotation:
            ann = division.annotation
            console.print("  [bold cyan]── LAYER 2a: ANNOTATION-BASED MECHANICAL VERDICT ──[/bold cyan]")
            console.print(
                f"  Claims: [green]Prop {ann.prop_total_claims}[/green] / "
                f"[red]Opp {ann.opp_total_claims}[/red]  |  "
                f"Rebuttals mapped: {len(ann.rebuttals)}\n"
            )

            # Claims table
            claims_table = Table(show_header=True, box=None, padding=(0, 1))
            claims_table.add_column("ID", width=12)
            claims_table.add_column("Speaker", width=25)
            claims_table.add_column("Type", width=15)
            claims_table.add_column("Status", width=12)
            claims_table.add_column("Claim", width=55)

            for c in ann.claims:
                side_style = "green" if c.side == Side.PROPOSITION else "red"
                demolished = any(
                    r.target_claim_id == c.claim_id
                    and r.addresses_specific_logic and r.undermines_original
                    for r in ann.rebuttals
                )
                status = "[red]✗ DEMOLISHED[/red]" if demolished else "[green]✓ SURVIVES[/green]"
                claims_table.add_row(
                    c.claim_id,
                    f"[{side_style}]{c.speaker_name}[/{side_style}]",
                    f"{c.claim_type} ({c.specificity[:4]})",
                    status,
                    c.claim_text[:55] + ("…" if len(c.claim_text) > 55 else ""),
                )
            console.print(claims_table)

            # Score breakdown
            console.print(f"\n  [bold]Score breakdown:[/bold]")
            for line in ann.score_breakdown.split("\n"):
                if line.strip():
                    console.print(f"    {line}")

            prop_label = "[green]PROP[/green]" if ann.winner == Side.PROPOSITION else "PROP"
            opp_label = "[red]OPP[/red]" if ann.winner == Side.OPPOSITION else "OPP"
            console.print(
                f"\n  {prop_label}: {ann.prop_score:.1f}  vs  "
                f"{opp_label}: {ann.opp_score:.1f}  →  "
                f"[bold]{ann.winner.value.upper()}[/bold] ({ann.margin})\n"
            )

        # ── Layer 3: Argument Audit ──
        if division.argument_audit:
            audit = division.argument_audit
            console.print("  [bold cyan]── LAYER 3: ARGUMENT GRAPH AUDIT ──[/bold cyan]")
            console.print(
                f"  Prop claims surviving: [green]{audit.prop_claims_surviving}[/green]  |  "
                f"Opp claims surviving: [red]{audit.opp_claims_surviving}[/red]  →  "
                f"[bold]{audit.structural_winner.value.upper()}[/bold]"
            )
            if audit.key_uncontested_claims:
                console.print("  [bold]Uncontested:[/bold]")
                for c in audit.key_uncontested_claims:
                    console.print(f"    ✓ {c}")
            if audit.key_demolished_claims:
                console.print("  [bold]Demolished:[/bold]")
                for c in audit.key_demolished_claims:
                    console.print(f"    ✗ {c}")
            console.print(f"\n  {audit.structural_summary}\n")

        # ── Synthesis ──
        console.print("  [bold cyan]── SYNTHESIS ──[/bold cyan]")
        console.print(f"  {division.summary}\n")

    else:
        # Fall back to raw verdict text
        console.print("[dim]Structured verdict failed — raw verdict below:[/dim]\n")
        console.print(verdict_raw)

    # ---------------------------------------------------------------
    # Save transcript to file (text + structured JSON)
    # ---------------------------------------------------------------
    txt_path = save_transcript(result, console)

    # Save structured JSON alongside the text transcript
    json_path = txt_path.with_suffix(".json")
    _save_single_run_json(result, json_path)
    console.print(f"  💾 Structured data saved to [bold cyan]{json_path}[/bold cyan]\n")


def _save_single_run_json(result: dict, filepath: Path) -> None:
    """Save the full structured data from a single debate run as JSON.

    This makes per-speaker rubric scores, judge votes, and argument audit
    data available for visualization without parsing the text transcript.
    """
    division: DivisionResult | None = result.get("division")
    division_dict = json.loads(division.model_dump_json()) if division else None

    per_speaker = None
    if division and division.rubric:
        per_speaker = [
            {
                "speaker_name": s.speaker_name,
                "side": s.side.value,
                "argument_strength": s.argument_strength,
                "rebuttal_quality": s.rebuttal_quality,
                "evidence_grounding": s.evidence_grounding,
                "rhetorical_effectiveness": s.rhetorical_effectiveness,
                "persona_fidelity": s.persona_fidelity,
                "overall": s.overall,
            }
            for s in division.rubric.scores
        ]

    data = {
        "motion": MOTION,
        "timestamp": datetime.now().isoformat(),
        "speeches": [
            {
                "speaker_name": s.speaker_name,
                "side": s.side.value,
                "word_count": s.word_count,
                "tone": s.tone,
                "arguments": [
                    {"claim": a.claim, "is_rebuttal": a.is_rebuttal}
                    for a in s.arguments
                ],
            }
            for s in result["speeches"]
        ],
        "pois": [
            {
                "from": p.from_speaker,
                "to": p.to_speaker,
                "accepted": p.accepted,
                "text": p.text,
            }
            for p in result["pois"]
        ],
        "per_speaker_scores": per_speaker,
        "division": division_dict,
    }
    filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_transcript(result: dict, console, filepath: Path | str | None = None) -> Path:
    """Save the full debate transcript and verdict to a file.

    Args:
        result: The graph output dict.
        console: Rich console for printing.
        filepath: Optional explicit path. Auto-generates if None.

    Returns:
        The path to the saved file.
    """
    if filepath is None:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"debate_{timestamp}.txt"
    filepath = Path(filepath)

    lines = []
    lines.append("=" * 80)
    lines.append("CAMBRIDGE UNION EXHIBITION DEBATE")
    lines.append(f"Motion: {MOTION}")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 80)
    lines.append("")

    # Include definitions framework if extracted
    defs_ctx = result.get("definitions_context", "")
    if defs_ctx:
        lines.append(defs_ctx)
        lines.append("")

    for speech in result["speeches"]:
        side_label = "PROPOSITION" if speech.side == Side.PROPOSITION else "OPPOSITION"
        lines.append(f"{'─' * 80}")
        lines.append(f"{speech.speaker_name} ({side_label})")
        lines.append(f"Position: {speech.speaking_position}/6 · "
                      f"{speech.word_count} words · Tone: {speech.tone}")
        lines.append(f"{'─' * 80}")
        lines.append(speech.full_text)
        lines.append("")

        # POIs
        speech_pois = [p for p in result["pois"] if p.to_speaker == speech.speaker_name]
        for poi in speech_pois:
            status = "ACCEPTED" if poi.accepted else "DECLINED"
            lines.append(f"  [POI from {poi.from_speaker} — {status}]")
            lines.append(f'  "{poi.text}"')
            if poi.accepted and poi.response:
                lines.append(f"  → {poi.response}")
            lines.append("")

    lines.append("")
    lines.append("=" * 80)
    lines.append("THE DIVISION")
    lines.append("=" * 80)
    lines.append("")

    division = result.get("division")
    verdict_raw = result.get("verdict_raw", "")

    if division:
        winner = "PROPOSITION (AYE)" if division.winner == Side.PROPOSITION else "OPPOSITION (NO)"
        lines.append(f"Result: {winner} by a {division.margin} margin")
        lines.append(f"Summary: {division.summary}")
        lines.append("")

        # Rubric scores
        if division.rubric:
            lines.append("LAYER 1: ANALYTICAL RUBRIC")
            lines.append("-" * 40)
            for s in division.rubric.scores:
                side_str = "PROP" if s.side == Side.PROPOSITION else "OPP"
                lines.append(
                    f"  {s.speaker_name} ({side_str}): "
                    f"Arg={s.argument_strength:.0f} Reb={s.rebuttal_quality:.0f} "
                    f"Evd={s.evidence_grounding:.0f} Rht={s.rhetorical_effectiveness:.0f} "
                    f"Per={s.persona_fidelity:.0f} → OVR={s.overall:.0f}/10"
                )
                lines.append(f"    {s.rationale}")
            lines.append(f"  Prop Total: {division.rubric.prop_total:.1f} | "
                         f"Opp Total: {division.rubric.opp_total:.1f} → "
                         f"{division.rubric.rubric_winner.value.upper()}")
            lines.append("")

    lines.append("")
    lines.append("=" * 80)
    lines.append("FULL VERDICT ANALYSIS")
    lines.append("=" * 80)
    lines.append(verdict_raw)

    filepath.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"\n  💾 Transcript saved to [bold cyan]{filepath}[/bold cyan]\n")
    return filepath


if __name__ == "__main__":
    asyncio.run(main())

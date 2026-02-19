"""
Phase 1: Persona Construction & Speech Preparation

Builds per-speaker persona prompts from their profile + style + RAG passages,
then generates independent preparation notes for the debate.
"""

from __future__ import annotations

import asyncio
import random

import src.config as cfg
from src.corpus.ingest import retrieve_relevant_passages
from src.models import Side, SpeakerData, SpeakerProfile, StyleProfile


# ---------------------------------------------------------------------------
# Argument emphasis lenses — randomly selected per speaker per run
# ---------------------------------------------------------------------------

ARGUMENT_EMPHASIS_LENSES = [
    (
        "Lead with your most CONCRETE, evidence-backed argument. Open with "
        "specific data, real case studies, or named examples. Build your "
        "principled case on top of this empirical foundation."
    ),
    (
        "Lead with your strongest PRINCIPLED argument — the ethical, "
        "philosophical, or rights-based case. Establish the moral framework "
        "first, then bring in evidence to support it."
    ),
    (
        "Prioritise REBUTTAL. Spend significant time dismantling the "
        "opposing side's most likely arguments before building your own "
        "case. Make their position look untenable, then present yours "
        "as the obvious alternative."
    ),
    (
        "Think about what the AUDIENCE — a room of Cambridge students — "
        "cares about most. Frame every argument in terms of practical, "
        "near-term impact on people's lives. Abstract debates about "
        "principles should be secondary to tangible consequences."
    ),
    (
        "Find the most COUNTERINTUITIVE or unexpected argument in your "
        "preparation material. Lead with it to surprise the audience and "
        "reframe the debate. Challenge assumptions that both sides usually "
        "take for granted."
    ),
    (
        "Structure your speech around a NARRATIVE ARC. Don't just list "
        "arguments — tell a story. Start with a vivid example or scenario, "
        "build through your reasoning, and arrive at a powerful conclusion "
        "that ties back to your opening."
    ),
    (
        "Focus on SYSTEMIC arguments. Don't argue about individual cases — "
        "argue about structures, incentives, and institutions. What systems "
        "are in place? What systems are needed? Why does the system-level "
        "view favour your side?"
    ),
    (
        "Play to your UNIQUE EXPERTISE. What can you say about this topic "
        "that NO ONE ELSE in this debate can? Lean into what makes your "
        "perspective distinctive. Avoid arguments that any generalist could "
        "make."
    ),
]


# ---------------------------------------------------------------------------
# Persona prompt builder
# ---------------------------------------------------------------------------

def build_persona_prompt(
    speaker: SpeakerProfile,
    style: StyleProfile,
    retrieved_passages: list[str],
) -> str:
    """
    Build the full system prompt that makes an LLM behave as this speaker.
    """
    passages_text = "\n\n".join(
        f"[{i+1}] {p}" for i, p in enumerate(retrieved_passages)
    )

    return f"""You are {speaker.name}, speaking at the Cambridge Union.

IDENTITY
{speaker.bio}

YOUR POSITIONS (from your own writings and speeches):
{passages_text}

YOUR STYLE
Register: {style.speech_register}
Opening patterns: {', '.join(style.opening_patterns)}
Characteristic devices: {', '.join(style.rhetorical_devices)}
Disagreement style: {style.disagreement_style}
Signature phrases: {', '.join(style.signature_phrases) if style.signature_phrases else 'None identified'}
Closing patterns: {', '.join(style.closing_patterns)}

THE SETTING
This is a Cambridge Union exhibition debate — a formal but lively setting.
The audience is largely students and academics. Wit, clarity, and
conviction matter. You are speaking to persuade a live audience who will
vote with their feet at the end.

BEHAVIOURAL RULES
- You are {speaker.name}. Stay in character throughout.
- Ground your arguments in your documented positions and knowledge.
  Do not fabricate views you do not hold.
- You may reference what previous speakers said (if you've heard them),
  but your core arguments should be your own.
- If you have no documented position on a specific sub-point, draw on
  your broader worldview to reason about it — as {speaker.name} would —
  rather than inventing a position from nothing.
- You may accept or decline Points of Information. If you accept,
  respond briefly and sharply before continuing your speech.
- Think of yourself as "{speaker.name} would argue that..."
  (third-person framing to maintain consistency)

CRITICAL — FACTUAL INTEGRITY
- Do NOT invent personal anecdotes, fictional friends, fictional patients,
  or made-up stories. No "A friend of mine…" or "My aunt…" unless it is
  documented in your corpus.
- Use ONLY real, verifiable examples: documented case studies, published
  research, named institutions, and well-known public events.
- If you cite a statistic, it must be a real one from your corpus or
  widely known public knowledge. Do NOT hallucinate numbers.
- When in doubt, argue from principle rather than fabricating evidence."""


# ---------------------------------------------------------------------------
# Speech preparation (independent per speaker)
# ---------------------------------------------------------------------------

async def prepare_speaker(
    speaker: SpeakerProfile,
    style: StyleProfile,
    motion: str,
    all_speaker_names: list[str],
    strategy_directive: str = "",
    passages: list[str] | None = None,
) -> SpeakerData:
    """
    Prepare a single speaker for the debate. This runs independently for each
    speaker — they do NOT see each other's preparation.

    Args:
        speaker: The speaker's static profile.
        style: Extracted rhetorical style.
        motion: The debate motion.
        all_speaker_names: Names of all 6 speakers (for awareness, not coordination).
        strategy_directive: Optional strategic emphasis for this variation.
        passages: Pre-selected RAG passages.  If provided, skips retrieval
            (used by the ensemble to inject retrieval variation).

    Returns:
        SpeakerData with persona prompt, prep notes, and retrieved passages.
    """
    # 1. RAG retrieval — use provided passages or retrieve normally
    if passages is not None:
        retrieved = list(passages)  # copy so we can shuffle safely
    else:
        retrieved = await retrieve_relevant_passages(speaker, motion, k=10)

    # 1b. Shuffle passage order — LLMs are position-sensitive, so this
    # naturally varies which arguments get emphasised first.
    random.shuffle(retrieved)

    # 2. Build persona prompt
    persona_prompt = build_persona_prompt(speaker, style, retrieved)

    # 3. Select a random argument emphasis lens for this run
    emphasis_lens = random.choice(ARGUMENT_EMPHASIS_LENSES)

    # 4. Generate preparation notes
    side_label = "support of" if speaker.side == Side.PROPOSITION else "opposition to"
    others = [n for n in all_speaker_names if n != speaker.name]

    strategy_block = ""
    if strategy_directive:
        strategy_block = f"""
STRATEGIC EMPHASIS (from your debate coach):
{strategy_directive}
This should be the backbone of your speech. Build your case around this angle,
though you may of course weave in supporting points from other angles.
"""

    prep_message = f"""You are preparing to speak at the Cambridge Union
in {side_label} the motion: "{motion}"

You are speaker {speaker.speaking_position} of 3 on your side.
The other speakers in the debate are: {', '.join(others)}
{strategy_block}
ARGUMENT APPROACH FOR THIS SPEECH:
{emphasis_lens}

Prepare your core arguments and key points. You have approximately 7 minutes
(~1,300 words).

You do NOT know what your teammates will argue. You have not coordinated
with anyone.

Output your preparation notes — the arguments you intend to make, the
evidence you'll draw on, and your planned structure. These are YOUR notes;
you may adapt during the debate based on what earlier speakers say."""

    response = await cfg.SPEAKER_LLM.ainvoke([
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": prep_message},
    ])

    return SpeakerData(
        profile=speaker,
        style=style,
        persona_prompt=persona_prompt,
        prep_notes=response.content,
        retrieved_passages=retrieved,
    )


async def prepare_all_speakers(
    speakers: list[SpeakerProfile],
    styles: dict[str, StyleProfile],
    motion: str,
    strategy_directives: dict[str, str] | None = None,
    passage_overrides: dict[str, list[str]] | None = None,
) -> dict[str, SpeakerData]:
    """
    Prepare all 6 speakers in parallel (they're independent).

    Args:
        speakers: All speaker profiles.
        styles: Speaker ID -> StyleProfile mapping.
        motion: The debate motion.
        strategy_directives: Optional speaker_id -> strategic emphasis mapping.
        passage_overrides: Optional speaker_id -> pre-selected passages mapping.
            If a speaker's ID is present, those passages are used instead of
            live RAG retrieval (used by the ensemble for retrieval variation).

    Returns:
        Dict mapping speaker_id -> SpeakerData
    """
    all_names = [s.name for s in speakers]
    directives = strategy_directives or {}
    passages_map = passage_overrides or {}

    tasks = [
        prepare_speaker(
            speaker=speaker,
            style=styles[speaker.id],
            motion=motion,
            all_speaker_names=all_names,
            strategy_directive=directives.get(speaker.id, ""),
            passages=passages_map.get(speaker.id),
        )
        for speaker in speakers
    ]

    results = await asyncio.gather(*tasks)

    return {sd.profile.id: sd for sd in results}


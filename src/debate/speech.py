"""
Phase 2: Speech Generation & Points of Information

Generates sequential speeches and mid-speech POI interjections.
Each speech sees the full transcript of all prior speeches.
"""

from __future__ import annotations

import json
import random

from pydantic import BaseModel, Field

import src.config as cfg
from src.models import (
    ArgumentPoint,
    DefinitionsContestation,
    DefinitionsFrame,
    POI,
    Side,
    SpeakerData,
    SpeechOutput,
)


# ---------------------------------------------------------------------------
# Transcript formatting
# ---------------------------------------------------------------------------

def format_transcript(speeches: list[SpeechOutput], pois: list[POI]) -> str:
    """Format the debate transcript so far for inclusion in prompts."""
    if not speeches:
        return "(No speeches yet.)"

    parts = []
    for speech in speeches:
        side_label = "PROPOSITION" if speech.side == Side.PROPOSITION else "OPPOSITION"
        parts.append(f"--- {speech.speaker_name} ({side_label}) ---")
        parts.append(speech.full_text)

        # Include any POIs that occurred during this speech
        speech_pois = [p for p in pois if p.to_speaker == speech.speaker_name]
        for poi in speech_pois:
            status = "ACCEPTED" if poi.accepted else "DECLINED"
            parts.append(f"\n  [POI from {poi.from_speaker} — {status}]")
            parts.append(f'  "{poi.text}"')
            if poi.accepted and poi.response:
                parts.append(f"  Response: {poi.response}")

        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Definitions & framing extraction
# ---------------------------------------------------------------------------

async def extract_definitions(first_speech: SpeechOutput, motion: str) -> DefinitionsFrame:
    """Extract the definitional framework from the first Proposition speech."""

    prompt = f"""You are analysing the opening speech of a Cambridge Union debate.

Motion: "{motion}"

The first Proposition speaker ({first_speech.speaker_name}) gave this speech:
---
{first_speech.full_text}
---

Extract the DEFINITIONAL FRAMEWORK this speaker has set for the debate:

1. KEY TERMS: Identify 2-4 key terms from the motion and how the speaker
   defines (explicitly or implicitly) each one.  Pay special attention to
   words that could be interpreted multiple ways.

2. SCOPE: What is IN scope for this debate according to the speaker?
   (e.g. "AI systems making recommendations under governance frameworks"
   vs "fully autonomous AI with no oversight")

3. EXCLUSIONS: What does the speaker explicitly put OUT of scope, if anything?

4. PROPOSITION FRAMING: In 1-2 sentences, how does this speaker frame the
   central question the audience should answer?

Be precise.  If the speaker did not explicitly define a term, note how
they implicitly interpreted it based on their arguments."""

    defs = await cfg.ANALYSIS_LLM.with_structured_output(DefinitionsFrame).ainvoke(prompt)
    return defs


async def extract_contestation(
    second_speech: SpeechOutput,
    definitions: DefinitionsFrame,
    motion: str,
) -> DefinitionsContestation:
    """Extract the Opposition's response to the Proposition's definitions."""

    terms_text = "\n".join(
        f"  - {t.term}: {t.definition}" for t in definitions.key_terms
    )

    prompt = f"""You are analysing the first Opposition response in a Cambridge Union debate.

Motion: "{motion}"

The Proposition defined terms as follows:
{terms_text}
Scope: {definitions.scope}
Framing: {definitions.proposition_framing}

The first Opposition speaker ({second_speech.speaker_name}) responded:
---
{second_speech.full_text}
---

Analyse the Opposition's response to the definitions:

1. Do they broadly ACCEPT the Proposition's definitions?
2. Which terms (if any) do they CONTEST or REDEFINE?  How?
3. Do they offer a COUNTER-FRAMING of the central question?
4. What is the AGREED GROUND — what do both sides accept?

If the Opposition did not explicitly contest definitions, note whether
they implicitly operated under different assumptions."""

    contestation = await cfg.ANALYSIS_LLM.with_structured_output(
        DefinitionsContestation
    ).ainvoke(prompt)
    return contestation


def build_definitions_context(
    definitions: DefinitionsFrame,
    contestation: DefinitionsContestation | None = None,
) -> str:
    """Build a definitions context block for injection into speech/judge prompts."""

    parts = ["DEFINITIONAL FRAMEWORK FOR THIS DEBATE"]
    parts.append("=" * 45)
    parts.append("")
    parts.append("The Proposition has defined the key terms as follows:")
    for t in definitions.key_terms:
        parts.append(f"  • {t.term}: {t.definition}")
    parts.append(f"\nScope: {definitions.scope}")
    if definitions.exclusions:
        parts.append(f"Exclusions: {definitions.exclusions}")
    parts.append(f"Proposition's framing: {definitions.proposition_framing}")

    if contestation:
        parts.append("")
        if contestation.accepts_definitions:
            parts.append("The Opposition ACCEPTS these definitions.")
        else:
            parts.append("The Opposition CONTESTS some definitions:")
            for t in contestation.contested_terms:
                parts.append(f"  • {t.term}: {t.definition}")
        if contestation.counter_framing:
            parts.append(f"Opposition's counter-framing: {contestation.counter_framing}")
        if contestation.agreed_ground:
            parts.append(f"Agreed ground: {contestation.agreed_ground}")

    parts.append("")
    parts.append(
        "ALL speakers should argue within this framework. If you disagree "
        "with how a term has been defined, you may contest it explicitly, "
        "but do NOT silently operate under different definitions."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Structured extraction schema (lightweight — for post-processing only)
# ---------------------------------------------------------------------------

class SpeechMetadata(BaseModel):
    """Extracted metadata from a generated speech (not the speech itself)."""

    opening: str = Field(description="The opening lines of the speech (first 2-3 sentences)")
    closing: str = Field(description="The closing lines of the speech (last 2-3 sentences)")
    arguments: list[ArgumentPoint] = Field(description="The 2-4 main argument points made")
    tone: str = Field(description="The overall tone, e.g. 'measured but firm', 'impassioned', 'forensic'")
    key_rhetorical_moves: list[str] = Field(
        default_factory=list, description="Notable rhetorical devices used"
    )


# ---------------------------------------------------------------------------
# Speech generation (two-step: generate then extract)
# ---------------------------------------------------------------------------

async def generate_speech(
    speaker_data: SpeakerData,
    position_in_order: int,
    motion: str,
    prior_speeches: list[SpeechOutput],
    prior_pois: list[POI],
    definitions_context: str = "",
) -> SpeechOutput:
    """
    Generate a single debate speech using a two-step process:
      1. Generate the full speech as free text (o3, no structural constraints)
      2. Extract structured metadata (arguments, tone) via gpt-4o

    This ensures speeches hit the word target without being compressed by JSON.
    """
    speaker = speaker_data.profile

    length_instruction = (
        f"Your speech MUST be approximately {cfg.SPEECH_WORD_TARGET} words "
        f"(~7 minutes at speaking pace). This is a HARD requirement. "
        f"Write a FULL speech — do not abbreviate, summarise, or cut short."
    )

    if position_in_order == 1:
        task = f"""You are the first speaker for the Proposition.
Open the case for the motion: "{motion}"

Your role:
- Define and frame the motion
- Present your strongest 2-3 arguments
- Set the tone for your side
- You are speaking first; there is nothing to rebut yet.

{length_instruction}

Deliver your speech as {speaker.name} would — in their voice,
with their characteristic style and argumentation.

Write ONLY the speech text. No metadata, no stage directions, no JSON."""

    elif position_in_order == 2:
        defs_block = ""
        if definitions_context:
            defs_block = f"""
The Proposition has set the following definitional framework:
{definitions_context}

You MUST engage with these definitions. You may:
  (a) Accept them and argue within that framework, OR
  (b) Explicitly contest specific definitions and explain WHY your
      interpretation is better — do NOT silently redefine terms.
"""

        task = f"""You are the first speaker for the Opposition against
the motion: "{motion}"

You have just heard the following speech from {prior_speeches[0].speaker_name}:
---
{prior_speeches[0].full_text}
---
{defs_block}
Your role:
- Engage with the Proposition's definitions — accept or contest them explicitly
- Respond to the strongest points made by the Proposition
- Present your own 2-3 core arguments against the motion
- You must both rebut AND build your own case

{length_instruction}

Deliver your speech as {speaker.name} would.

Write ONLY the speech text. No metadata, no stage directions, no JSON."""

    else:
        side_label = "Proposition" if speaker.side == Side.PROPOSITION else "Opposition"
        transcript = format_transcript(prior_speeches, prior_pois)
        is_final = position_in_order == 6

        defs_block = ""
        if definitions_context:
            defs_block = f"""
{definitions_context}

You MUST argue within this agreed framework. If you believe a definition
has been set up unfairly, contest it explicitly — but do NOT silently
operate under different assumptions.
"""

        task = f"""You are speaker {speaker.speaking_position} of 3 for the {side_label}.

The motion is: "{motion}"
{defs_block}
The debate so far:
{transcript}

Your preparation notes (written before hearing the other speeches):
{speaker_data.prep_notes}

Your role:
- Engage with what has been said — rebut key opposing arguments
- Advance NEW arguments that haven't been made yet by your side
  (adapt your prep notes: drop points already covered by teammates,
   strengthen points that are under attack)
- {'This is the FINAL speech of the debate. Drive home the most compelling case for your side. Build to a powerful peroration. Make it count.' if is_final else 'Build on what your side has established while advancing the debate.'}

CRITICAL — DO NOT REPEAT YOUR TEAMMATES' ARGUMENTS:
- Re-read the speeches from your own side above. If a teammate has
  ALREADY made an argument, do NOT make it again — even in different
  words.  Saying the same thing a second time wastes your speech time
  and bores the audience.
- Instead, ADVANCE the debate: bring arguments your teammates did NOT
  make, provide evidence they did NOT cite, or develop angles they only
  touched on briefly.
- You may briefly REFERENCE a teammate's point ("As my colleague noted…")
  but then immediately move to your OWN distinct contribution.
- If all your prepared arguments have already been covered, find fresh
  angles: a counter-example, a different moral framework, an empirical
  case study, or a direct rebuttal of a specific opposing claim.

You did NOT coordinate with your teammates beforehand. If they made a
point you disagree with or would frame differently, you may do so —
this is natural in exhibition debate.

{length_instruction}

Deliver your speech as {speaker.name} would.

Write ONLY the speech text. No metadata, no stage directions, no JSON."""

    # --- Step 1: Generate the full speech as free text ---
    response = await cfg.SPEAKER_LLM.ainvoke([
        {"role": "system", "content": speaker_data.persona_prompt},
        {"role": "user", "content": task},
    ])
    full_text = response.content.strip()

    # --- Step 2: Extract structured metadata ---
    extraction_prompt = f"""Analyse this debate speech and extract its structure.

Speaker: {speaker.name}
Side: {"Proposition" if speaker.side == Side.PROPOSITION else "Opposition"}

Speech:
---
{full_text}
---

Extract:
- The opening lines (first 2-3 sentences that hook the audience)
- The closing lines (the peroration / final appeal)
- The 2-4 main argument points (each with claim, reasoning, evidence if cited,
  and whether it's a rebuttal of a specific speaker)
- The overall tone
- Key rhetorical devices used"""

    metadata = await cfg.ANALYSIS_LLM.with_structured_output(SpeechMetadata).ainvoke(
        extraction_prompt
    )

    return SpeechOutput(
        speaker_id=speaker.id,
        speaker_name=speaker.name,
        side=speaker.side,
        speaking_position=position_in_order,
        opening=metadata.opening,
        arguments=metadata.arguments,
        closing=metadata.closing,
        full_text=full_text,
        tone=metadata.tone,
        key_rhetorical_moves=metadata.key_rhetorical_moves,
        word_count=len(full_text.split()),
    )


# ---------------------------------------------------------------------------
# Points of Information
# ---------------------------------------------------------------------------

class POIOffer(BaseModel):
    """Schema for the POI generation model's output."""
    offers_poi: bool = Field(description="Whether an opposing speaker rises on a POI")
    from_speaker: str = Field(default="", description="Name of the speaker offering the POI")
    text: str = Field(default="", description="The POI challenge text (1-2 sentences)")


async def generate_pois(
    speech: SpeechOutput,
    opposing_speakers: list[SpeakerData],
    receiving_speaker: SpeakerData,
    all_pois_so_far: list[POI],
) -> list[POI]:
    """
    For each non-protected argument point, decide whether an opposing
    speaker rises on a POI.

    If accepted, the RECEIVING speaker generates their own response
    using their persona.

    Protected time: first and last argument points are off-limits.
    Max 2 POIs per speech to maintain realism.
    """
    pois = []
    max_pois_per_speech = 2

    for i, argument in enumerate(speech.arguments):
        if len(pois) >= max_pois_per_speech:
            break

        # Skip protected time (first and last argument)
        if i == 0 or i == len(speech.arguments) - 1:
            continue

        opponent_names = [sd.profile.name for sd in opposing_speakers]

        poi_prompt = f"""During a Cambridge Union debate, the current speaker
({speech.speaker_name}) just made this argument:

"{argument.claim} — {argument.reasoning}"

The opposing speakers are: {', '.join(opponent_names)}

Should any opposing speaker rise on a Point of Information?

Rules:
- POIs should be brief (1-2 sentences), pointed, and designed to
  wrong-foot the speaker.
- Not every argument warrants a POI. Only rise if there's a genuinely
  sharp, targeted intervention to make.
- At most one speaker should rise per argument.
- Roughly 30-40% of arguments attract a POI attempt — most do NOT.

If YES: set offers_poi to true, provide from_speaker and text.
If NO: set offers_poi to false."""

        try:
            poi_offer = await cfg.POI_LLM.with_structured_output(POIOffer).ainvoke(poi_prompt)

            if not poi_offer.offers_poi:
                continue

            # --- Acceptance gating ---
            # Base probability + persona adjustment
            acceptance_prob = 0.45  # Base: speakers decline more than half

            # Adjust based on speech position (later speakers accept fewer)
            if speech.speaking_position >= 5:
                acceptance_prob -= 0.15  # Final speakers are more guarded

            accepted = random.random() < acceptance_prob

            # --- Generate response if accepted ---
            poi_response_text = None
            if accepted:
                response_prompt = f"""You are {speech.speaker_name}, mid-speech at the Cambridge Union.

You have just been interrupted by a Point of Information from {poi_offer.from_speaker}:
"{poi_offer.text}"

You ACCEPTED the POI. Give a sharp, brief response (1-3 sentences) that
either deflects, rebuts, or turns the point to your advantage, then
indicate you're resuming your speech.

Respond in character as {speech.speaker_name}. Write ONLY the response."""

                poi_resp = await cfg.POI_LLM.ainvoke([
                    {"role": "system", "content": receiving_speaker.persona_prompt},
                    {"role": "user", "content": response_prompt},
                ])
                poi_response_text = poi_resp.content.strip()

            pois.append(POI(
                from_speaker=poi_offer.from_speaker,
                to_speaker=speech.speaker_name,
                text=poi_offer.text,
                accepted=accepted,
                response=poi_response_text,
                after_argument_index=i,
            ))

        except Exception:
            # If anything fails, skip this POI opportunity
            continue

    return pois

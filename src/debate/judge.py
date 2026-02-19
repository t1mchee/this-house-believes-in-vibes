"""
Three-layer debate judging system.

  Layer 1 — ANALYTICAL RUBRIC SCORING
    One gpt-4o call per speech (6 calls).  Each speech is scored on a
    5-dimension rubric: argument strength, rebuttal quality, evidence
    grounding, rhetorical effectiveness, and persona fidelity.

  Layer 2 — ANNOTATION-BASED MECHANICAL VERDICT (primary verdict)
    Two gpt-4o calls:
      Call 1: Extract and classify every substantive claim.
      Call 2: Map every rebuttal, assess engagement and effectiveness
              via three binary (yes/no) questions per rebuttal.
    The verdict is computed arithmetically from the annotations —
    the LLM never decides who won.

  Layer 3 — ARGUMENT GRAPH AUDIT
    One gpt-4o call that provides a high-level structural summary:
    surviving claims, uncontested claims, demolished claims.

Rate limiting:
  The layers execute SEQUENTIALLY (1 → 2 → 3) and calls within each
  layer are throttled via an asyncio.Semaphore to stay within the
  OpenAI TPM budget.  A small inter-call delay prevents burst-induced
  429 errors.
"""

from __future__ import annotations

import asyncio

import src.config as cfg
from src.debate.speech import format_transcript
from src.models import (
    AnnotationVerdict,
    ArgumentAudit,
    ClaimAnnotation,
    ClaimExtractionResult,
    ClaimNode,
    DivisionResult,
    EngagementVerdict,
    EngagementVote,
    JudgeVote,
    PanelVerdict,
    POI,
    RecalibrationResult,
    RebuttalAnnotation,
    RebuttalMappingResult,
    RubricScorecard,
    Side,
    SpeechOutput,
    SpeechScore,
)

# ---------------------------------------------------------------------------
# Rate-limiting helpers
# ---------------------------------------------------------------------------

# Max concurrent judge LLM calls — keeps us within TPM limits.
# Each call sends the full transcript (~12K tokens).  With 30K TPM
# on gpt-4o, 2 concurrent calls is safe.
_MAX_CONCURRENT_JUDGE_CALLS = 2
_INTER_CALL_DELAY = 1.0  # seconds between releasing semaphore slots


async def _throttled_invoke(coro, semaphore: asyncio.Semaphore):
    """Run a coroutine under a semaphore with a small post-call delay."""
    async with semaphore:
        result = await coro
        await asyncio.sleep(_INTER_CALL_DELAY)
        return result


# ---------------------------------------------------------------------------
# Layer 1: Analytical Rubric Scoring
# ---------------------------------------------------------------------------

async def _score_single_speech(
    speech: SpeechOutput,
    transcript: str,
    motion: str,
    definitions_context: str = "",
) -> SpeechScore:
    """Score one speech on a 5-dimension rubric."""

    side_label = "Proposition" if speech.side == Side.PROPOSITION else "Opposition"

    defs_block = ""
    if definitions_context:
        defs_block = f"""
{definitions_context}

When scoring, consider whether the speaker argues within the agreed
definitional framework or effectively contests it.  Speakers who silently
operate under different definitions than those established should be
marked down on ARGUMENT STRENGTH.
"""

    prompt = f"""You are an expert debate adjudicator scoring a single speech
in a Cambridge Union exhibition debate.

CRITICAL: You are evaluating ARGUMENTATIVE SKILL, not whether you agree
with the speaker's position.  Both sides of any motion can be argued well
or poorly.  A speaker defending a position you find uncomfortable can
still deliver an outstanding speech.  Score the CRAFT, not the CONTENT.

Motion: "{motion}"
{defs_block}
Full debate transcript (for context — you need this to evaluate rebuttals
and how the speech fits into the overall debate):
{transcript}

────────────────────────────────
Score THIS speech:
Speaker: {speech.speaker_name} ({side_label})
Speech:
{speech.full_text}
────────────────────────────────

Score on five dimensions using the FULL 1–10 scale.  Here are anchors:
  1–2: Poor — incoherent, irrelevant, or no real arguments
  3–4: Below average — generic, vague, or poorly structured
  5:   Average — competent but unremarkable; makes basic points
  6:   Solid — clear, well-organised, but no distinctive insight
  7:   Strong — sharp arguments, good engagement, persuasive moments
  8:   Excellent — impressive depth, strong evidence, memorable rhetoric
  9:   Outstanding — top-tier debating; exceptional on multiple dimensions
  10:  World-class — near-perfect execution across all dimensions

Dimensions:

1. ARGUMENT STRENGTH: Are the claims logically valid and internally
   consistent?  Does the reasoning go beyond surface-level assertion?
   Do the arguments actually advance the speaker's side of the motion?

2. REBUTTAL QUALITY: How effectively does the speaker engage with the
   STRONGEST opposing arguments (not strawmen)?  Do they identify and
   attack the real pressure points?  (Score 5 for the very first speaker
   since they have nothing to rebut yet — judge their pre-emptive
   framing instead.)

3. EVIDENCE GROUNDING: Is the evidence specific, real, and well-deployed?
   Does it feel sourced from genuine expertise rather than generic
   talking points anyone could make?

4. RHETORICAL EFFECTIVENESS: How persuasive is the delivery?  Is the
   speech well-structured, clear, and compelling?  Does the speaker
   use their allotted time well and build to a strong conclusion?

5. PERSONA FIDELITY: Does this speech sound like the real person?  Is
   the vocabulary, register, and argumentative style authentic to who
   {speech.speaker_name} actually is?

Also provide:
- An OVERALL score (1–10) that is NOT a simple average.  Weight argument
  strength and rhetorical effectiveness most heavily, then rebuttal.
- A brief RATIONALE (2–3 sentences) justifying the overall score.

USE THE FULL SCALE.  If most speeches cluster at the same score, you
are not being discriminating enough.  Differentiate between speakers.
Not every speech in a debate is equally good."""

    score = await cfg.JUDGE_LLM.with_structured_output(SpeechScore).ainvoke(prompt)

    # Ensure metadata is correct (LLM may get it right, but enforce)
    score.speaker_name = speech.speaker_name
    score.side = speech.side
    return score


async def _recalibrate_scores(
    speeches: list[SpeechOutput],
    initial_scores: list[SpeechScore],
    transcript: str,
    motion: str,
    definitions_context: str = "",
) -> list[SpeechScore]:
    """
    Comparative recalibration pass.

    Sees all 6 speeches + their initial independent scores side-by-side,
    then force-ranks the speeches and reassigns scores with genuine
    differentiation.  Returns updated SpeechScore objects.
    """
    # Build the initial scores summary for the LLM
    scores_summary = []
    for s in initial_scores:
        side_label = "PROP" if s.side == Side.PROPOSITION else "OPP"
        scores_summary.append(
            f"  {s.speaker_name} ({side_label}): "
            f"Arg={s.argument_strength:.1f} Reb={s.rebuttal_quality:.1f} "
            f"Evi={s.evidence_grounding:.1f} Rhet={s.rhetorical_effectiveness:.1f} "
            f"Pers={s.persona_fidelity:.1f} → Overall={s.overall:.1f}"
        )
    scores_block = "\n".join(scores_summary)

    defs_block = ""
    if definitions_context:
        defs_block = f"\n{definitions_context}\n"

    prompt = f"""You are a chief adjudicator reviewing the scores of 6 speeches
in a Cambridge Union exhibition debate.  An initial panel scored each speech
independently, but their scores have CLUSTERED — almost every speech got the
same score.  Your job is to RECALIBRATE by comparing them directly.

Motion: "{motion}"
{defs_block}
Full debate transcript:
{transcript}

INITIAL SCORES (from independent review):
{scores_block}

YOUR TASK:

1. FORCE-RANK all 6 speeches from 1 (best) to 6 (worst).
   Every speech must have a UNIQUE rank — no ties allowed.

2. REASSIGN SCORES on each dimension (1–10) such that:
   - The spread between the BEST and WORST speaker's overall score is
     AT LEAST 2.0 points (ideally 3+).
   - You use at LEAST 3 different integer values for overall scores
     across the 6 speakers (e.g. not all 7s and 8s).
   - If a speaker genuinely did poorly on a dimension (e.g. no real
     rebuttals, generic evidence, flat delivery), score them 4-6 on
     that dimension — not 7+.
   - If a speaker was exceptional, they can keep an 8-9.
   - A score of 10 should be almost never used.

3. For each speaker, provide a 1-2 sentence COMPARATIVE rationale:
   why they are ranked above or below the adjacent speakers.

RECALIBRATION PRINCIPLES:
- The FIRST speaker cannot score highly on rebuttal (there is nothing
  to rebut).  Score them 5 and judge their proactive framing instead.
- Later speakers who FAIL to engage with preceding arguments should
  score LOW on rebuttal quality, even if their own arguments are decent.
- "Generic" arguments that anyone could make (e.g. "AI could be risky")
  deserve lower scores than specific, evidence-backed claims.
- Repeating a teammate's arguments should LOWER the score vs. advancing
  novel points.
- A side that wins structurally does not mean every speaker on that side
  was individually better.  Judge INDIVIDUAL quality.

Return the 6 recalibrated scores, ranked from best (rank=1) to worst (rank=6)."""

    try:
        result = await cfg.JUDGE_LLM.with_structured_output(
            RecalibrationResult
        ).ainvoke(prompt)

        # Map recalibrated scores back to SpeechScore objects
        recalibrated: list[SpeechScore] = []
        for ranking in result.rankings:
            # Find the original SpeechScore to preserve side info
            original = next(
                (s for s in initial_scores if s.speaker_name == ranking.speaker_name),
                None,
            )
            if not original:
                continue
            recalibrated.append(SpeechScore(
                speaker_name=ranking.speaker_name,
                side=original.side,
                argument_strength=ranking.argument_strength,
                rebuttal_quality=ranking.rebuttal_quality,
                evidence_grounding=ranking.evidence_grounding,
                rhetorical_effectiveness=ranking.rhetorical_effectiveness,
                persona_fidelity=ranking.persona_fidelity,
                overall=ranking.overall,
                rationale=ranking.rationale,
            ))

        # If recalibration returned all 6 scores, use them; otherwise fall back
        if len(recalibrated) == len(initial_scores):
            return recalibrated
        else:
            print(f"  ⚠️  Recalibration returned {len(recalibrated)}/{len(initial_scores)} scores — using originals")
            return initial_scores
    except Exception as e:
        print(f"  ⚠️  Recalibration failed ({e}) — using initial scores")
        return initial_scores


async def score_speeches(
    speeches: list[SpeechOutput],
    pois: list[POI],
    motion: str,
    definitions_context: str = "",
) -> RubricScorecard:
    """Layer 1: Score all 6 speeches on the 5-dimension rubric.

    Two phases:
      1. Independent scoring (6 throttled calls) — each speech scored alone.
      2. Comparative recalibration (1 call) — sees all scores side-by-side,
         force-ranks, and spreads scores to avoid clustering.

    Calls are throttled to stay within the TPM budget.
    """
    transcript = format_transcript(speeches, pois)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_JUDGE_CALLS)

    # Phase 1: Independent scoring
    tasks = [
        _throttled_invoke(
            _score_single_speech(speech, transcript, motion, definitions_context),
            semaphore,
        )
        for speech in speeches
    ]
    initial_scores: list[SpeechScore] = list(await asyncio.gather(*tasks))

    # Check if scores are clustered — if spread is < 1.5, recalibrate
    overalls = [s.overall for s in initial_scores]
    spread = max(overalls) - min(overalls)
    unique_ints = len(set(int(o) for o in overalls))

    if spread < 1.5 or unique_ints <= 2:
        print(f"  📊 Scores clustered (spread={spread:.1f}, {unique_ints} unique values) — recalibrating…")
        scores = await _recalibrate_scores(
            speeches, initial_scores, transcript, motion, definitions_context,
        )
    else:
        scores = initial_scores

    prop_total = sum(s.overall for s in scores if s.side == Side.PROPOSITION)
    opp_total = sum(s.overall for s in scores if s.side == Side.OPPOSITION)

    # On a tie, use the higher argument_strength + rebuttal_quality subtotals
    # as a tiebreaker (these are the most debate-relevant dimensions)
    if prop_total == opp_total:
        prop_debate_sub = sum(
            s.argument_strength + s.rebuttal_quality
            for s in scores if s.side == Side.PROPOSITION
        )
        opp_debate_sub = sum(
            s.argument_strength + s.rebuttal_quality
            for s in scores if s.side == Side.OPPOSITION
        )
        rubric_winner = Side.PROPOSITION if prop_debate_sub >= opp_debate_sub else Side.OPPOSITION
    else:
        rubric_winner = Side.PROPOSITION if prop_total > opp_total else Side.OPPOSITION

    return RubricScorecard(
        scores=scores,
        prop_total=prop_total,
        opp_total=opp_total,
        rubric_winner=rubric_winner,
    )


# ---------------------------------------------------------------------------
# Layer 2: Annotation-Based Mechanical Verdict
# ---------------------------------------------------------------------------

# Scoring weights for the mechanical tally
_WEIGHT_EVIDENCE_BACKED = 3
_WEIGHT_PRINCIPLED = 2
_WEIGHT_ASSERTION = 1
_WEIGHT_SPECIFIC_BONUS = 1  # added to any claim that is "specific"

# Rebuttal scoring — graduated system
_WEIGHT_DEMOLITION_BONUS = 2        # bonus for fully demolishing a claim (logic + undermines)
_WEIGHT_STRONG_REBUTTAL = 1.5       # direct + new info but not quite undermining
_WEIGHT_PARTIAL_REBUTTAL = 0.5      # addresses logic OR provides new info (but not both)
_CLAIM_WEAKENING_FACTOR = 0.5       # demolished claims score at 50% instead of full removal

# Last-speaker discount: new claims from the final speaker (who cannot be
# rebutted) are weighted less — mirrors the Cambridge Union convention that
# the "whip" speech should primarily summarise/rebut, not introduce new material.
_FINAL_SPEAKER_CLAIM_DISCOUNT = 0.5


async def _extract_claims(
    transcript: str,
    motion: str,
    definitions_context: str = "",
) -> ClaimExtractionResult:
    """Step 1: Extract and classify every substantive claim in the debate."""

    defs_block = ""
    if definitions_context:
        defs_block = f"""
{definitions_context}

Consider whether claims operate within or outside the established definitions.
"""

    prompt = f"""You are a debate annotator.  Your job is FACTUAL ANNOTATION,
not judgment.  You have no opinion on the motion.

Motion: "{motion}"
{defs_block}
Full debate transcript:
{transcript}

TASK: Extract every DISTINCT SUBSTANTIVE CLAIM made by each speaker.
Ignore pleasantries, procedural remarks, and pure rhetoric with no
propositional content.

For each claim, provide:

1. claim_id: A unique ID in the format "prop_N_X" or "opp_N_X" where
   N is the speaker number (1-3 on their side) and X is a letter (a, b, c…).

2. speaker_name: The speaker who made the claim.

3. side: "proposition" or "opposition".

4. claim_text: A brief, neutral summary of the claim (1-2 sentences).
   Do NOT editorialize.  Just state what the speaker claimed.

5. claim_type: EXACTLY one of:
   - "evidence_backed": The speaker cited specific evidence (data, named
     studies, specific cases, named institutions) to support this claim.
   - "principled": The speaker argued from ethical principles, logical
     reasoning, or established frameworks (but without specific empirical
     evidence).
   - "assertion": The speaker stated this as fact without supporting
     evidence or detailed reasoning.

6. specificity: EXACTLY one of:
   - "specific": The claim references particular data, cases, people,
     institutions, or examples that only someone with domain expertise
     would cite.
   - "generic": The claim could be made by anyone with general knowledge
     of the topic.

RULES:
- Extract claims from BOTH sides equally.  Do not skip claims because
  you disagree with them.
- Each claim should be a single distinct point.  If a speaker makes three
  arguments, that's three claims.
- Be thorough: aim for 3-6 claims per speaker (more for longer speeches).
- Classification must be STRICT: "evidence_backed" requires NAMED evidence.
  Saying "studies show" without naming any study is "assertion"."""

    result = await cfg.JUDGE_LLM.with_structured_output(
        ClaimExtractionResult
    ).ainvoke(prompt)
    return result


async def _map_rebuttals(
    transcript: str,
    claims: list[ClaimAnnotation],
    motion: str,
    definitions_context: str = "",
) -> RebuttalMappingResult:
    """Step 2: Map every rebuttal and assess effectiveness with yes/no questions."""

    claims_list = "\n".join(
        f"  [{c.claim_id}] {c.speaker_name} ({c.side.value}): {c.claim_text}"
        for c in claims
    )

    defs_block = ""
    if definitions_context:
        defs_block = f"""
{definitions_context}

A rebuttal that attacks a strawman version of a claim (based on different
definitions) should be classified as engagement_level="strawman".
"""

    prompt = f"""You are a debate annotator.  Your job is FACTUAL ANNOTATION,
not judgment.  You have no opinion on the motion.

Motion: "{motion}"
{defs_block}
Full debate transcript:
{transcript}

Here are all the claims extracted from the debate:
{claims_list}

TASK: For each claim that was addressed by a LATER speaker from the
OPPOSING side, create a rebuttal annotation.

CRITICAL — LOOK FOR REBUTTALS FROM BOTH SIDES:
You MUST look for rebuttals in BOTH directions:
  • Opposition speakers rebutting Proposition claims (e.g. opp_1 rebuts prop_1_a)
  • Proposition speakers rebutting Opposition claims (e.g. prop_2 rebuts opp_1_b)

Proposition rebuttals often look different from Opposition rebuttals.
Proposition speakers frequently rebut by:
  - Saying "The opposition argues X, but in fact..."
  - Saying "This point about risk ignores the evidence that..."
  - Preemptively addressing concerns raised by earlier opposition speakers
  - Providing counter-evidence to specific opposition claims
  - Reframing opposition arguments as supporting the Proposition's case

If you find rebuttals flowing in only ONE direction, you have almost
certainly missed rebuttals from the other side.  Re-read the transcript.

For each rebuttal, provide:

1. target_claim_id: The claim_id being rebutted.

2. rebutting_speaker: Who made the rebuttal.

3. rebuttal_summary: A brief, neutral summary of the rebuttal (1-2 sentences).

4. engagement_level: EXACTLY one of:
   - "direct": The rebuttal engages with the specific logic or evidence
     of the original claim.
   - "indirect": The rebuttal addresses the general theme/topic but not
     the specific claim.
   - "strawman": The rebuttal attacks a distorted version of the claim.

5. method: EXACTLY one of:
   - "counter_evidence": Provides specific evidence that contradicts the claim.
   - "logical_flaw": Identifies a logical error or inconsistency in the claim.
   - "counter_example": Provides a concrete example that undermines the claim.
   - "reassertion": Simply restates the opposing position without engaging
     the claim's logic (this is the weakest form of rebuttal).

6-8. THREE YES/NO EFFECTIVENESS QUESTIONS — answer each independently:

   6. addresses_specific_logic (true/false):
      Does this rebuttal engage with the SPECIFIC LOGICAL STRUCTURE of the
      original claim?  Not just the topic, but the actual reasoning chain?
      (Saying "AI is risky" does NOT address a claim about AI diagnostic
      accuracy in dermatology.  It must engage with the SPECIFIC claim.)

   7. provides_new_information (true/false):
      Does the rebuttal introduce NEW information — evidence, a counter-
      example, a logical analysis — that the original speaker did not
      already address?  Simply restating the opposing position = false.

   8. undermines_original (true/false):
      After this rebuttal, has the original claim been MEANINGFULLY
      WEAKENED?  Answer true if the rebuttal would cause a reasonable
      audience member to give the original claim SIGNIFICANTLY less weight.
      This does NOT require total destruction — a rebuttal that exposes
      a genuine flaw, provides strong counter-evidence, or shows the claim
      only holds under narrow conditions should be marked true.
      Answer false ONLY if the rebuttal fails to land — e.g. it is off-topic,
      attacks a strawman, or merely reasserts the opposing position.

RULES:
- Only annotate CROSS-SIDE rebuttals (Prop rebuts Opp claims, and vice versa).
- A claim can have 0, 1, or multiple rebuttals.
- Not every claim is rebutted — many go uncontested.  That's fine.
- The three yes/no questions are INDEPENDENT.  A rebuttal can provide new
  information (Q7=true) but fail to address the specific logic (Q6=false).
- Expect roughly 30-50% of rebuttals to score undermines_original=true.
  If you have 10+ rebuttals and NONE are marked true, you are being too strict.
- Apply the SAME standard to BOTH sides.  A weak Prop rebuttal of an Opp
  claim should be scored the same as a weak Opp rebuttal of a Prop claim."""

    result = await cfg.JUDGE_LLM.with_structured_output(
        RebuttalMappingResult
    ).ainvoke(prompt)
    return result


def _compute_mechanical_verdict(
    claims: list[ClaimAnnotation],
    rebuttals: list[RebuttalAnnotation],
) -> AnnotationVerdict:
    """Step 3: Pure arithmetic — no LLM involved.

    Graduated scoring system:
      - Each claim scores points based on type + specificity
      - Demolished claims (logic + undermines) score at 50% instead of full
      - Each rebuttal earns the rebutting side points based on quality:
        • Demolition (logic + undermines): bonus points
        • Strong rebuttal (direct + new info): good points
        • Partial rebuttal (logic OR new info): some points
    """

    # Build lookup: claim_id → claim
    claim_map = {c.claim_id: c for c in claims}

    # Build lookup: claim_id → list of rebuttals targeting it
    rebuttal_map: dict[str, list[RebuttalAnnotation]] = {}
    for r in rebuttals:
        rebuttal_map.setdefault(r.target_claim_id, []).append(r)

    # Determine which claims are demolished
    # A claim is "demolished" if ANY rebuttal scores true on both
    # addresses_specific_logic AND undermines_original
    demolished_ids: set[str] = set()
    for claim_id, claim_rebuttals in rebuttal_map.items():
        for r in claim_rebuttals:
            if r.addresses_specific_logic and r.undermines_original:
                demolished_ids.add(claim_id)
                break

    # Count claims per side
    prop_claims = [c for c in claims if c.side == Side.PROPOSITION]
    opp_claims = [c for c in claims if c.side == Side.OPPOSITION]

    prop_surviving = [c for c in prop_claims if c.claim_id not in demolished_ids]
    opp_surviving = [c for c in opp_claims if c.claim_id not in demolished_ids]
    prop_demolished = [c for c in prop_claims if c.claim_id in demolished_ids]
    opp_demolished = [c for c in opp_claims if c.claim_id in demolished_ids]

    # Identify the final speech slot — it's the highest-numbered slot in
    # the claim IDs (e.g. "opp_3_a" → slot "opp_3", or "opp_6_a" → "opp_6")
    # The final speaker's NEW claims get discounted because they cannot be rebutted.
    all_slots = set()
    for c in claims:
        parts = c.claim_id.split("_")
        slot_num = int(parts[1])
        all_slots.add(slot_num)
    final_slot_num = max(all_slots) if all_slots else 999

    # Score claims — demolished claims score at reduced rate, not zero
    def _raw_claim_score(c: ClaimAnnotation) -> float:
        base = {
            "evidence_backed": _WEIGHT_EVIDENCE_BACKED,
            "principled": _WEIGHT_PRINCIPLED,
            "assertion": _WEIGHT_ASSERTION,
        }.get(c.claim_type, _WEIGHT_ASSERTION)
        if c.specificity == "specific":
            base += _WEIGHT_SPECIFIC_BONUS
        return base

    def score_claims_graduated(side_claims: list[ClaimAnnotation]) -> float:
        total = 0.0
        for c in side_claims:
            raw = _raw_claim_score(c)
            if c.claim_id in demolished_ids:
                raw *= _CLAIM_WEAKENING_FACTOR  # weakened, not zeroed

            # Discount claims from the final speaker (who can't be rebutted)
            parts = c.claim_id.split("_")
            slot_num = int(parts[1])
            if slot_num == final_slot_num:
                raw *= _FINAL_SPEAKER_CLAIM_DISCOUNT

            total += raw
        return total

    prop_claim_score = score_claims_graduated(prop_claims)
    opp_claim_score = score_claims_graduated(opp_claims)

    # Score rebuttals — graduated system that rewards engagement even
    # without full demolition
    def score_rebuttals(side: Side) -> float:
        """Credit `side` for rebuttals they made against the OTHER side's claims."""
        total = 0.0
        other_side = Side.OPPOSITION if side == Side.PROPOSITION else Side.PROPOSITION
        for r in rebuttals:
            target_claim = claim_map.get(r.target_claim_id)
            if not target_claim:
                continue
            # Only count rebuttals targeting the OTHER side's claims
            if target_claim.side != other_side:
                continue

            # Graduated scoring based on rebuttal quality
            if r.addresses_specific_logic and r.undermines_original:
                # Full demolition — highest credit
                total += _WEIGHT_DEMOLITION_BONUS
            elif r.engagement_level == "direct" and r.provides_new_information:
                # Strong rebuttal — engaged directly with new evidence
                total += _WEIGHT_STRONG_REBUTTAL
            elif r.addresses_specific_logic or r.provides_new_information:
                # Partial rebuttal — at least engaged or brought new info
                total += _WEIGHT_PARTIAL_REBUTTAL
            # else: reassertion / strawman / no engagement → 0 pts
        return total

    prop_rebuttal_score = score_rebuttals(Side.PROPOSITION)
    opp_rebuttal_score = score_rebuttals(Side.OPPOSITION)

    prop_total = prop_claim_score + prop_rebuttal_score
    opp_total = opp_claim_score + opp_rebuttal_score

    # Determine winner and margin
    if prop_total == opp_total:
        # Tiebreak: side with more surviving evidence-backed claims
        prop_eb = sum(1 for c in prop_surviving if c.claim_type == "evidence_backed")
        opp_eb = sum(1 for c in opp_surviving if c.claim_type == "evidence_backed")
        winner = Side.PROPOSITION if prop_eb >= opp_eb else Side.OPPOSITION
    else:
        winner = Side.PROPOSITION if prop_total > opp_total else Side.OPPOSITION

    diff = abs(prop_total - opp_total)
    max_score = max(prop_total, opp_total) if max(prop_total, opp_total) > 0 else 1
    ratio = diff / max_score
    if ratio >= 0.4:
        margin = "landslide"
    elif ratio >= 0.2:
        margin = "clear"
    else:
        margin = "narrow"

    # Count rebuttals by direction for the breakdown
    prop_rebuttals_made = sum(
        1 for r in rebuttals
        if claim_map.get(r.target_claim_id) and
        claim_map[r.target_claim_id].side == Side.OPPOSITION
    )
    opp_rebuttals_made = sum(
        1 for r in rebuttals
        if claim_map.get(r.target_claim_id) and
        claim_map[r.target_claim_id].side == Side.PROPOSITION
    )

    # Count final-speaker claims
    final_speaker_claims_prop = sum(
        1 for c in prop_claims
        if int(c.claim_id.split("_")[1]) == final_slot_num
    )
    final_speaker_claims_opp = sum(
        1 for c in opp_claims
        if int(c.claim_id.split("_")[1]) == final_slot_num
    )

    # Build breakdown string
    breakdown_lines = [
        f"PROPOSITION: {prop_total:.1f} pts",
        f"  Claims: {len(prop_claims)} total, {len(prop_demolished)} demolished "
        f"(claim score: {prop_claim_score:.1f})",
        f"  Rebuttals of Opp: {prop_rebuttals_made} made, {prop_rebuttal_score:.1f} pts earned",
        f"",
        f"OPPOSITION: {opp_total:.1f} pts",
        f"  Claims: {len(opp_claims)} total, {len(opp_demolished)} demolished "
        f"(claim score: {opp_claim_score:.1f})",
        f"  Rebuttals of Prop: {opp_rebuttals_made} made, {opp_rebuttal_score:.1f} pts earned",
        f"",
        f"Final-speaker claims discounted at {_FINAL_SPEAKER_CLAIM_DISCOUNT}× "
        f"(Prop: {final_speaker_claims_prop}, Opp: {final_speaker_claims_opp})",
        f"",
        f"Scoring: evidence_backed={_WEIGHT_EVIDENCE_BACKED}, "
        f"principled={_WEIGHT_PRINCIPLED}, assertion={_WEIGHT_ASSERTION}, "
        f"specific_bonus=+{_WEIGHT_SPECIFIC_BONUS}",
        f"Rebuttals: demolition={_WEIGHT_DEMOLITION_BONUS}, "
        f"strong={_WEIGHT_STRONG_REBUTTAL}, partial={_WEIGHT_PARTIAL_REBUTTAL}, "
        f"weakening={_CLAIM_WEAKENING_FACTOR}×",
    ]

    return AnnotationVerdict(
        claims=claims,
        rebuttals=rebuttals,
        prop_total_claims=len(prop_claims),
        opp_total_claims=len(opp_claims),
        prop_surviving=len(prop_surviving),
        opp_surviving=len(opp_surviving),
        prop_demolished=len(prop_demolished),
        opp_demolished=len(opp_demolished),
        prop_score=prop_total,
        opp_score=opp_total,
        winner=winner,
        margin=margin,
        score_breakdown="\n".join(breakdown_lines),
    )


async def run_annotation_verdict(
    speeches: list[SpeechOutput],
    pois: list[POI],
    motion: str,
    definitions_context: str = "",
) -> AnnotationVerdict:
    """Layer 2: Two LLM calls (extract + map) → mechanical verdict.

    This is the primary verdict mechanism.  The LLM annotates; the
    arithmetic decides.
    """
    transcript = format_transcript(speeches, pois)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_JUDGE_CALLS)

    # Step 1: Extract and classify claims
    extraction = await _throttled_invoke(
        _extract_claims(transcript, motion, definitions_context),
        semaphore,
    )

    # Step 2: Map rebuttals with effectiveness assessment
    mapping = await _throttled_invoke(
        _map_rebuttals(transcript, extraction.claims, motion, definitions_context),
        semaphore,
    )

    # Step 3: Pure arithmetic — no LLM
    verdict = _compute_mechanical_verdict(extraction.claims, mapping.rebuttals)

    return verdict


# ---------------------------------------------------------------------------
# Layer 2b: Engagement-Focused LLM Verdict (primary verdict)
# ---------------------------------------------------------------------------

_ENGAGEMENT_N_JUDGES = 3   # judges per pass
_ENGAGEMENT_PASSES = 2      # pass 1: Prop=A, pass 2: Prop=B


def _anonymize_transcript(
    transcript: str,
    speakers: list[SpeechOutput],
    prop_is_team_a: bool,
) -> str:
    """Replace all speaker names and side labels with anonymous Team A/B labels."""

    anon = transcript

    # Build mapping: speaker_name → "Team X Speaker N"
    prop_speakers = sorted(
        [s for s in speakers if s.side == Side.PROPOSITION],
        key=lambda s: s.speaking_position,
    )
    opp_speakers = sorted(
        [s for s in speakers if s.side == Side.OPPOSITION],
        key=lambda s: s.speaking_position,
    )

    if prop_is_team_a:
        team_a_speakers, team_b_speakers = prop_speakers, opp_speakers
    else:
        team_a_speakers, team_b_speakers = opp_speakers, prop_speakers

    # Replace speaker names (longest first to avoid partial matches)
    name_map: list[tuple[str, str]] = []
    for i, s in enumerate(team_a_speakers, 1):
        name_map.append((s.speaker_name, f"Team A Speaker {i}"))
    for i, s in enumerate(team_b_speakers, 1):
        name_map.append((s.speaker_name, f"Team B Speaker {i}"))

    # Sort by length descending to avoid partial replacement
    name_map.sort(key=lambda x: len(x[0]), reverse=True)

    for real_name, anon_name in name_map:
        anon = anon.replace(real_name, anon_name)

    # Replace side labels
    if prop_is_team_a:
        anon = anon.replace("PROPOSITION", "TEAM A")
        anon = anon.replace("Proposition", "Team A")
        anon = anon.replace("proposition", "team a")
        anon = anon.replace("OPPOSITION", "TEAM B")
        anon = anon.replace("Opposition", "Team B")
        anon = anon.replace("opposition", "team b")
    else:
        anon = anon.replace("OPPOSITION", "TEAM A")
        anon = anon.replace("Opposition", "Team A")
        anon = anon.replace("opposition", "team a")
        anon = anon.replace("PROPOSITION", "TEAM B")
        anon = anon.replace("Proposition", "Team B")
        anon = anon.replace("proposition", "team b")

    return anon


async def _cast_engagement_vote(
    anonymized_transcript: str,
    motion: str,
    judge_index: int,
) -> EngagementVote:
    """One judge evaluates the anonymized debate on engagement quality."""

    from langchain_openai import ChatOpenAI

    # Use a dedicated LLM with temperature=0.8 for natural variance
    judge_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.8,
        max_tokens=4096,
        max_retries=6,
    )

    prompt = f"""You are an expert debate adjudicator evaluating a Cambridge Union
exhibition debate.  The speakers have been ANONYMIZED — you do not know
which side is "Proposition" and which is "Opposition."  The two teams are
labelled "Team A" and "Team B."

Motion: "{motion}"

ANONYMIZED DEBATE TRANSCRIPT:
{anonymized_transcript}

YOUR EVALUATION CRITERIA — in order of importance:

1. ENGAGEMENT WITH OPPOSING ARGUMENTS (40% weight)
   Which team did a better job of DIRECTLY ENGAGING with the other team's
   strongest points?  Did they address the substance of the opposing
   arguments, or did they talk past them?  Credit teams that grappled
   with difficult points rather than ignoring them.

2. ARGUMENT QUALITY (30% weight)
   Which team's arguments were better structured, more logical, and
   better evidenced?  Quality over quantity — one devastating argument
   is worth more than five weak assertions.

3. REBUTTAL EFFECTIVENESS (20% weight)
   Which team's rebuttals actually LANDED — i.e. weakened or demolished
   opposing arguments?  A rebuttal that engages with the specific logic
   of a claim is worth more than a general objection.

4. COHERENT NARRATIVE (10% weight)
   Which team told a more coherent, cumulative story across their three
   speeches?  Did their case build, or did speakers repeat each other?

CRITICAL ANTI-BIAS INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are evaluating ARGUMENTATIVE SKILL, not the truth of the motion.

• Your personal views on the topic are COMPLETELY IRRELEVANT.
• A team you personally disagree with can ABSOLUTELY win if they
  argued better.
• DO NOT reward a team simply for advocating caution, safety, or
  the status quo.  These are not inherently superior positions.
  The cautious team still needs to ARGUE their case with evidence
  and engagement.
• DO NOT penalize a team for advocating a bold or controversial
  position.  If they argued it well with evidence and engagement,
  they deserve credit.
• If one team relies heavily on "what if something goes wrong?"
  without engaging the other team's specific evidence, that is a
  WEAKNESS, not a strength.
• Equally, if one team relies on "look at all these benefits!"
  without addressing the other team's specific concerns, that is
  also a WEAKNESS.

The winning team is the one that BEST ENGAGED with the debate —
the one that left fewer opposing arguments unaddressed and built
the more compelling case through evidence and reasoning.

EVALUATION STEPS:
1. Identify the single strongest argument made by each team.
2. Identify the single best rebuttal in the debate (from either team).
3. Identify the decisive moment — the exchange that most tilted the
   debate one way.
4. Based on the criteria above, determine which team argued BETTER.

Provide your evaluation:
- better_team: "Team A" or "Team B"
- engagement_quality_a: 1-10 — how well Team A engaged with Team B's arguments
- engagement_quality_b: 1-10 — how well Team B engaged with Team A's arguments
- strongest_argument_a: The single strongest argument from Team A
- strongest_argument_b: The single strongest argument from Team B
- best_rebuttal: The best rebuttal in the debate
- decisive_moment: The exchange that most tilted the debate
- key_reason: Why the winning team argued better (must reference SPECIFIC
  arguments and engagement, not general impressions)
- confidence: 0.0 = genuinely too close to call, 1.0 = unambiguous"""

    vote = await judge_llm.with_structured_output(EngagementVote).ainvoke(prompt)
    return vote


async def run_engagement_verdict(
    speeches: list[SpeechOutput],
    pois: list[POI],
    motion: str,
) -> EngagementVerdict:
    """Layer 2b: Dual-pass anonymized engagement-focused evaluation.

    Pass 1: Proposition = Team A, Opposition = Team B
    Pass 2: Proposition = Team B, Opposition = Team A

    3 judges per pass (total 6 votes).  If both passes agree, high
    confidence.  If they disagree, the debate is genuinely contested.
    """
    transcript = format_transcript(speeches, pois)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_JUDGE_CALLS)

    all_votes: list[EngagementVote] = []
    # Track which team label maps to which side for each vote
    vote_side_map: list[Side] = []  # what does "better_team" map to?

    for pass_num in range(_ENGAGEMENT_PASSES):
        prop_is_team_a = (pass_num == 0)  # Pass 1: Prop=A, Pass 2: Prop=B

        anon_transcript = _anonymize_transcript(
            transcript, speeches, prop_is_team_a
        )

        tasks = [
            _throttled_invoke(
                _cast_engagement_vote(anon_transcript, motion, j),
                semaphore,
            )
            for j in range(_ENGAGEMENT_N_JUDGES)
        ]
        votes = list(await asyncio.gather(*tasks))

        for v in votes:
            all_votes.append(v)
            # Map the anonymous team label back to the real side
            if v.better_team.strip().lower() == "team a":
                mapped = Side.PROPOSITION if prop_is_team_a else Side.OPPOSITION
            else:
                mapped = Side.OPPOSITION if prop_is_team_a else Side.PROPOSITION
            vote_side_map.append(mapped)

    # Count votes by real side
    prop_votes = sum(1 for s in vote_side_map if s == Side.PROPOSITION)
    opp_votes = sum(1 for s in vote_side_map if s == Side.OPPOSITION)

    total = prop_votes + opp_votes
    winner = Side.PROPOSITION if prop_votes > opp_votes else Side.OPPOSITION

    # Check if passes agree
    pass1_votes = vote_side_map[:_ENGAGEMENT_N_JUDGES]
    pass2_votes = vote_side_map[_ENGAGEMENT_N_JUDGES:]
    pass1_winner = Side.PROPOSITION if sum(1 for s in pass1_votes if s == Side.PROPOSITION) > _ENGAGEMENT_N_JUDGES / 2 else Side.OPPOSITION
    pass2_winner = Side.PROPOSITION if sum(1 for s in pass2_votes if s == Side.PROPOSITION) > _ENGAGEMENT_N_JUDGES / 2 else Side.OPPOSITION
    pass_agreement = (pass1_winner == pass2_winner)

    # Margin
    winning_frac = max(prop_votes, opp_votes) / total if total > 0 else 0.5
    if prop_votes == opp_votes:
        margin = "split"
    elif winning_frac >= 0.83:  # 5/6 or 6/6
        margin = "landslide"
    elif winning_frac >= 0.67:  # 4/6
        margin = "clear"
    else:
        margin = "narrow"

    mean_confidence = sum(v.confidence for v in all_votes) / len(all_votes) if all_votes else 0.5

    # Build summary
    summary_parts = [
        f"Engagement verdict: {winner.value.upper()} ({margin}).",
        f"Votes: Prop {prop_votes} – Opp {opp_votes} across {total} anonymized votes.",
    ]
    if pass_agreement:
        summary_parts.append("Both anonymization passes agree — verdict is robust to position bias.")
    else:
        summary_parts.append("WARNING: Passes DISAGREE — possible position bias or genuinely close debate.")

    return EngagementVerdict(
        votes=all_votes,
        prop_votes=prop_votes,
        opp_votes=opp_votes,
        winner=winner,
        margin=margin,
        mean_confidence=mean_confidence,
        pass_agreement=pass_agreement,
        engagement_summary=" ".join(summary_parts),
    )


# ---------------------------------------------------------------------------
# Legacy Layer 2: Multi-Judge Panel (kept for backward compatibility)
# ---------------------------------------------------------------------------

async def _cast_single_vote(
    transcript: str,
    motion: str,
    judge_index: int,
    definitions_context: str = "",
) -> JudgeVote:
    """One independent judge reads the transcript and casts a vote."""

    defs_block = ""
    if definitions_context:
        defs_block = f"""
{definitions_context}

Judge speakers within this framework.  If a speaker argues against a
strawman version of the other side's position, that is a weakness.
"""

    prompt = f"""You are judge #{judge_index + 1} on an independent panel evaluating
a Cambridge Union exhibition debate.

Motion: "{motion}"
{defs_block}

You came into this debate genuinely undecided.  You have no stake in the
outcome.  Your ONLY job is to determine which side ARGUED BETTER — which
side made the stronger, more persuasive, better-evidenced CASE.

CRITICAL DE-BIASING INSTRUCTIONS:
- You are judging DEBATING SKILL, not the truth of the motion.
- Your personal views on the topic are IRRELEVANT.  A side you disagree
  with can still win if they argued better.
- "Caution" and "safety" are not automatic virtues in debating.  The
  Opposition still needs to ARGUE their case, not just invoke risk.
- Equally, the Proposition cannot win by assertion alone — they must
  provide compelling evidence and reasoning.
- The side that better engages with the OTHER side's strongest points
  deserves credit.  Ignoring difficult arguments is a weakness.

Full debate transcript:
{transcript}

BEFORE you vote, you MUST complete this exercise:

STEP 1 — STEELMAN EACH SIDE (do this mentally):
  a) What was the SINGLE STRONGEST argument made by the Proposition?
  b) What was the SINGLE STRONGEST argument made by the Opposition?
  c) Which side did a better job of ENGAGING with the other's strongest
     argument?

STEP 2 — Now cast your vote:
1. Vote: PROPOSITION (AYE) or OPPOSITION (NO).
2. Confidence (0.0 = coin flip, 1.0 = absolutely certain).
3. Key reason: The single most important reason for your vote (1–2
   sentences).  This must reference a SPECIFIC argument or exchange,
   not a general sentiment like "they were more cautious."
4. Tipping point: The specific moment, argument, or exchange that was
   most decisive (1–2 sentences).

Rules:
- Focus on ARGUMENTATIVE QUALITY: logical structure, evidence, engagement
  with opposing points, and rhetorical skill.
- Who made the stronger CASE, not who has better credentials.
- Be honest about your confidence: many good debates are genuinely close.
  If you cannot clearly separate the sides, use a low confidence score."""

    vote = await cfg.JUDGE_LLM.with_structured_output(JudgeVote).ainvoke(prompt)
    return vote


async def run_judge_panel(
    speeches: list[SpeechOutput],
    pois: list[POI],
    motion: str,
    n_judges: int | None = None,
    definitions_context: str = "",
) -> PanelVerdict:
    """Layer 2: N independent judges vote, results aggregated.

    Calls are throttled to stay within the TPM budget.
    """
    if n_judges is None:
        n_judges = cfg.JUDGE_PANEL_SIZE

    transcript = format_transcript(speeches, pois)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_JUDGE_CALLS)

    tasks = [
        _throttled_invoke(
            _cast_single_vote(transcript, motion, i, definitions_context),
            semaphore,
        )
        for i in range(n_judges)
    ]
    votes: list[JudgeVote] = list(await asyncio.gather(*tasks))

    ayes = sum(1 for v in votes if v.vote == Side.PROPOSITION)
    noes = sum(1 for v in votes if v.vote == Side.OPPOSITION)

    winner = Side.PROPOSITION if ayes > noes else Side.OPPOSITION

    winning_fraction = max(ayes, noes) / n_judges
    if winning_fraction >= 0.9:
        margin = "landslide"
    elif winning_fraction >= 0.72:
        margin = "clear"
    else:
        margin = "narrow"

    mean_confidence = sum(v.confidence for v in votes) / n_judges

    return PanelVerdict(
        votes=votes,
        ayes=ayes,
        noes=noes,
        winner=winner,
        margin=margin,
        mean_confidence=mean_confidence,
        agreement_ratio=winning_fraction,
    )


# ---------------------------------------------------------------------------
# Layer 3: Argument Graph Audit
# ---------------------------------------------------------------------------

async def audit_argument_graph(
    speeches: list[SpeechOutput],
    pois: list[POI],
    motion: str,
    definitions_context: str = "",
) -> ArgumentAudit:
    """Layer 3: Map every claim, track rebuttals, count survivors."""

    transcript = format_transcript(speeches, pois)

    defs_block = ""
    if definitions_context:
        defs_block = f"""
{definitions_context}

When assessing whether a rebuttal "lands", consider whether it engages
with the actual claim as defined, or whether it attacks a strawman
version of the claim based on different definitions.
"""

    prompt = f"""You are a forensic debate analyst.  Perform a purely structural
analysis of the argument flow in this Cambridge Union exhibition debate.

CRITICAL: This is a LOGICAL analysis, not an opinion poll.  Your personal
views on the motion's truth are completely irrelevant.  A claim is "strong"
if it was well-argued and survived challenge, regardless of whether you
find it comfortable.

Motion: "{motion}"
{defs_block}

Full debate transcript:
{transcript}

Your tasks:

1. Identify EVERY distinct substantive claim made by each speaker.
   (Ignore throat-clearing, pleasantries, and procedural remarks.)

2. For each claim, determine:
   a) Was it rebutted by any later speaker?  If so, by whom?
   b) Was the rebuttal SUCCESSFUL — did it effectively neutralise the
      claim with a specific counter-argument, or merely reassert the
      opposing position?  (Saying "but we should be cautious" is NOT
      a rebuttal of a specific empirical claim.)
   c) Does the claim SURVIVE at the end of the debate?
      A claim survives if it was never challenged, or if the speaker's
      defence was more convincing than the attack.

3. Count surviving claims per side (proposition and opposition).

4. List 2–4 KEY UNCONTESTED CLAIMS — the strongest claims that no one
   even attempted to challenge.  Look on BOTH sides.

5. List 2–4 KEY DEMOLISHED CLAIMS — claims that were decisively destroyed
   by a specific rebuttal and clearly do not survive.  Look on BOTH sides.

6. Determine the STRUCTURAL WINNER — the side with more surviving claims,
   weighted by importance.  A side can win structurally even if they had
   fewer total claims, if their surviving claims are weightier.

7. Write a STRUCTURAL SUMMARY (2–3 sentences) describing the argument
   flow: who set the agenda, where the debate shifted, and how the
   argument landscape looks at the end.

Rules:
- A claim only "survives" if it was never challenged OR the speaker's
  defence was more convincing than the attack.
- A claim is "demolished" only if the rebuttal was clearly decisive —
  not merely contested.
- Vague appeals to "risk" or "caution" do NOT demolish specific empirical
  claims.  Conversely, citing a benefit does NOT demolish a specific
  risk argument.  The rebuttal must engage with the SUBSTANCE of the claim.
- Do NOT conflate rhetorical style with argument structure.  This is a
  purely logical/structural analysis.
- Be rigorous: most debates have many contested claims that are NOT
  decisively demolished."""

    audit = await cfg.JUDGE_LLM.with_structured_output(ArgumentAudit).ainvoke(prompt)
    return audit


# ---------------------------------------------------------------------------
# Orchestrator: run all three layers and synthesise
# ---------------------------------------------------------------------------

async def run_division(
    speeches: list[SpeechOutput],
    pois: list[POI],
    motion: str,
    n_judges: int | None = None,
    definitions_context: str = "",
) -> tuple[DivisionResult, str]:
    """
    Run all three judging layers and synthesise into a `DivisionResult`.

    The layers execute SEQUENTIALLY to stay within the OpenAI TPM budget.
    Within each layer, individual calls are throttled via a semaphore.

    Returns:
        (division_result, verdict_raw_text)

    The verdict_raw_text is a human-readable formatted report of all
    three layers, used for transcript saving and ensemble synthesis.
    """

    rubric: RubricScorecard | None = None
    annotation: AnnotationVerdict | None = None
    engagement: EngagementVerdict | None = None
    audit: ArgumentAudit | None = None

    # --- Layer 1: Rubric (6 throttled calls) ---
    print("  ⚖️  Layer 1: Scoring speeches on rubric…")
    try:
        rubric = await score_speeches(speeches, pois, motion, definitions_context)
        print(f"  ✅ Layer 1 complete — "
              f"Prop {rubric.prop_total:.1f} vs Opp {rubric.opp_total:.1f} → "
              f"{rubric.rubric_winner.value.upper()}")
    except Exception as e:
        print(f"  ⚠️  Layer 1 (Rubric) failed: {e}")

    # --- Layer 2a: Annotation-based mechanical verdict (2 calls) ---
    print("  ⚖️  Layer 2a: Annotating claims & rebuttals…")
    try:
        annotation = await run_annotation_verdict(
            speeches, pois, motion,
            definitions_context=definitions_context,
        )
        print(f"  ✅ Layer 2a complete — "
              f"Prop {annotation.prop_score:.1f} vs Opp {annotation.opp_score:.1f} → "
              f"{annotation.winner.value.upper()} ({annotation.margin})")
        print(f"     Claims: Prop {annotation.prop_surviving}/{annotation.prop_total_claims} "
              f"surviving, Opp {annotation.opp_surviving}/{annotation.opp_total_claims} surviving")
    except Exception as e:
        print(f"  ⚠️  Layer 2a (Annotation) failed: {e}")

    # --- Layer 2b: Engagement-focused LLM verdict (6 calls: 3 judges × 2 passes) ---
    print("  ⚖️  Layer 2b: Anonymized engagement evaluation (3 judges × 2 passes)…")
    try:
        engagement = await run_engagement_verdict(speeches, pois, motion)
        print(f"  ✅ Layer 2b complete — "
              f"Prop {engagement.prop_votes} vs Opp {engagement.opp_votes} → "
              f"{engagement.winner.value.upper()} ({engagement.margin})"
              f"{'  ✓ passes agree' if engagement.pass_agreement else '  ⚠ passes DISAGREE'}")
    except Exception as e:
        print(f"  ⚠️  Layer 2b (Engagement) failed: {e}")

    # --- Layer 3: Argument audit (1 call) ---
    print("  ⚖️  Layer 3: Auditing argument graph…")
    try:
        audit = await audit_argument_graph(speeches, pois, motion, definitions_context)
        print(f"  ✅ Layer 3 complete — "
              f"Prop {audit.prop_claims_surviving} / Opp {audit.opp_claims_surviving} "
              f"surviving → {audit.structural_winner.value.upper()}")
    except Exception as e:
        print(f"  ⚠️  Layer 3 (Audit) failed: {e}")

    # --- Determine final verdict ---
    # PRIMARY signal: engagement verdict (anonymized, debiased LLM evaluation)
    if engagement:
        winner = engagement.winner
        margin = engagement.margin
        ayes = engagement.prop_votes
        noes = engagement.opp_votes
        confidence = engagement.mean_confidence
    elif annotation:
        # Fallback to annotation
        winner = annotation.winner
        margin = annotation.margin
        ayes = annotation.prop_surviving
        noes = annotation.opp_surviving
        score_diff = abs(annotation.prop_score - annotation.opp_score)
        max_score = max(annotation.prop_score, annotation.opp_score, 1.0)
        confidence = min(score_diff / max_score, 1.0)
    elif rubric:
        winner = rubric.rubric_winner
        ayes = sum(1 for s in rubric.scores if s.side == winner)
        noes = len(rubric.scores) - ayes
        margin = "clear" if abs(rubric.prop_total - rubric.opp_total) > 3 else "narrow"
        confidence = 0.5
    else:
        if audit:
            winner = audit.structural_winner
        else:
            winner = Side.PROPOSITION
        ayes = noes = 0
        margin = "unknown"
        confidence = 0.0

    # --- Cross-layer agreement ---
    layer_winners = []
    if rubric:
        layer_winners.append(("Rubric", rubric.rubric_winner))
    if annotation:
        layer_winners.append(("Annotation", annotation.winner))
    if engagement:
        layer_winners.append(("Engagement", engagement.winner))
    if audit:
        layer_winners.append(("Structure", audit.structural_winner))

    unique_winners = set(w for _, w in layer_winners)
    all_agree = len(unique_winners) <= 1

    if all_agree and len(layer_winners) >= 3:
        agreement_note = f"All {len(layer_winners)} evaluation layers agree on the outcome."
    elif len(layer_winners) >= 2:
        verdicts = ", ".join(f"{name} → {w.value.upper()}" for name, w in layer_winners)
        agreement_note = f"Split verdict across layers: {verdicts}."
    else:
        agreement_note = "Partial evaluation — some layers failed."

    # --- Find best/worst speakers from rubric ---
    best_speaker = weakest_speaker = "N/A"
    best_score = worst_score = 0.0
    if rubric and rubric.scores:
        sorted_scores = sorted(rubric.scores, key=lambda s: s.overall, reverse=True)
        best_speaker = sorted_scores[0].speaker_name
        best_score = sorted_scores[0].overall
        weakest_speaker = sorted_scores[-1].speaker_name
        worst_score = sorted_scores[-1].overall

    # --- Build summary ---
    summary_parts = [
        f"The {winner.value.upper()} wins by a {margin} margin.",
        agreement_note,
    ]
    if engagement:
        summary_parts.append(
            f"Engagement verdict: Prop {engagement.prop_votes} – "
            f"Opp {engagement.opp_votes} "
            f"({'passes agree' if engagement.pass_agreement else 'passes DISAGREE'})."
        )
    if annotation:
        summary_parts.append(
            f"Mechanical score: Prop {annotation.prop_score:.1f} vs "
            f"Opp {annotation.opp_score:.1f}."
        )
    if rubric:
        summary_parts.append(
            f"Most effective speaker: {best_speaker} ({best_score:.1f}/10)."
        )
    if audit:
        summary_parts.append(
            f"Structural audit: {audit.prop_claims_surviving} Prop claims "
            f"and {audit.opp_claims_surviving} Opp claims survive."
        )
    summary = " ".join(summary_parts)

    division = DivisionResult(
        motion=motion,
        winner=winner,
        ayes=ayes,
        noes=noes,
        margin=margin,
        confidence=confidence,
        summary=summary,
        rubric=rubric,
        annotation=annotation,
        engagement=engagement,
        panel=None,
        argument_audit=audit,
    )

    # --- Build raw verdict text ---
    verdict_raw = _format_verdict_raw(rubric, annotation, engagement, audit, summary)

    return division, verdict_raw


def _format_verdict_raw(
    rubric: RubricScorecard | None,
    annotation: AnnotationVerdict | None,
    engagement: EngagementVerdict | None,
    audit: ArgumentAudit | None,
    summary: str,
) -> str:
    """Format a human-readable report of all layers."""

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("MULTI-LAYER VERDICT ANALYSIS")
    lines.append("=" * 60)
    lines.append("")

    # Layer 1
    lines.append("LAYER 1: ANALYTICAL RUBRIC SCORING")
    lines.append("-" * 40)
    if rubric:
        for s in rubric.scores:
            side_label = "PROP" if s.side == Side.PROPOSITION else "OPP"
            lines.append(f"  {s.speaker_name} ({side_label})")
            lines.append(f"    Argument Strength:     {s.argument_strength:.1f}/10")
            lines.append(f"    Rebuttal Quality:      {s.rebuttal_quality:.1f}/10")
            lines.append(f"    Evidence Grounding:    {s.evidence_grounding:.1f}/10")
            lines.append(f"    Rhetorical Effect.:    {s.rhetorical_effectiveness:.1f}/10")
            lines.append(f"    Persona Fidelity:      {s.persona_fidelity:.1f}/10")
            lines.append(f"    OVERALL:               {s.overall:.1f}/10")
            lines.append(f"    Rationale: {s.rationale}")
            lines.append("")
        lines.append(
            f"  Prop Total: {rubric.prop_total:.1f}  |  "
            f"Opp Total: {rubric.opp_total:.1f}  →  "
            f"{rubric.rubric_winner.value.upper()}"
        )
    else:
        lines.append("  (Layer 1 failed)")
    lines.append("")

    # Layer 2a — Annotation-based mechanical verdict
    lines.append("LAYER 2a: ANNOTATION-BASED MECHANICAL VERDICT")
    lines.append("-" * 40)
    if annotation:
        lines.append(f"  Claims extracted: {annotation.prop_total_claims} Prop, "
                      f"{annotation.opp_total_claims} Opp")
        lines.append(f"  Rebuttals mapped: {len(annotation.rebuttals)}")
        lines.append("")

        # Show claim details
        lines.append("  CLAIMS:")
        for c in annotation.claims:
            side_label = "PROP" if c.side == Side.PROPOSITION else "OPP"
            demolished = any(
                r.target_claim_id == c.claim_id
                and r.addresses_specific_logic and r.undermines_original
                for r in annotation.rebuttals
            )
            status = "✗ DEMOLISHED" if demolished else "✓ SURVIVES"
            lines.append(f"    [{c.claim_id}] {c.speaker_name} ({side_label}) "
                          f"[{c.claim_type}, {c.specificity}] {status}")
            lines.append(f"      {c.claim_text}")
        lines.append("")

        # Show rebuttal details
        if annotation.rebuttals:
            lines.append("  REBUTTALS:")
            for r in annotation.rebuttals:
                q_logic = "✓" if r.addresses_specific_logic else "✗"
                q_new = "✓" if r.provides_new_information else "✗"
                q_under = "✓" if r.undermines_original else "✗"
                lines.append(f"    {r.rebutting_speaker} → [{r.target_claim_id}] "
                              f"({r.engagement_level}, {r.method})")
                lines.append(f"      Addresses logic: {q_logic}  "
                              f"New info: {q_new}  "
                              f"Undermines: {q_under}")
                lines.append(f"      {r.rebuttal_summary}")
            lines.append("")

        # Score breakdown
        lines.append("  SCORE BREAKDOWN:")
        for line in annotation.score_breakdown.split("\n"):
            lines.append(f"    {line}")
        lines.append("")
        lines.append(f"  → {annotation.winner.value.upper()} ({annotation.margin})")
    else:
        lines.append("  (Layer 2a failed)")
    lines.append("")

    # Layer 2b — Engagement-focused LLM verdict
    lines.append("LAYER 2b: ENGAGEMENT-FOCUSED LLM VERDICT (PRIMARY)")
    lines.append("-" * 40)
    if engagement:
        lines.append(f"  Votes: Prop {engagement.prop_votes} – Opp {engagement.opp_votes}")
        lines.append(f"  Passes agree: {'YES ✓' if engagement.pass_agreement else 'NO ⚠'}")
        lines.append(f"  Mean confidence: {engagement.mean_confidence:.2f}")
        lines.append(f"  → {engagement.winner.value.upper()} ({engagement.margin})")
        lines.append("")

        # Show individual votes
        for i, v in enumerate(engagement.votes, 1):
            pass_label = "Pass 1" if i <= _ENGAGEMENT_N_JUDGES else "Pass 2"
            lines.append(f"  Judge {i} ({pass_label}): {v.better_team} "
                          f"(conf: {v.confidence:.2f})")
            lines.append(f"    Engagement: A={v.engagement_quality_a:.0f}/10, "
                          f"B={v.engagement_quality_b:.0f}/10")
            lines.append(f"    Strongest A: {v.strongest_argument_a}")
            lines.append(f"    Strongest B: {v.strongest_argument_b}")
            lines.append(f"    Best rebuttal: {v.best_rebuttal}")
            lines.append(f"    Decisive moment: {v.decisive_moment}")
            lines.append(f"    Key reason: {v.key_reason}")
            lines.append("")
    else:
        lines.append("  (Layer 2b failed)")
    lines.append("")

    # Layer 3
    lines.append("LAYER 3: ARGUMENT GRAPH AUDIT")
    lines.append("-" * 40)
    if audit:
        lines.append(f"  Prop claims surviving: {audit.prop_claims_surviving}")
        lines.append(f"  Opp claims surviving:  {audit.opp_claims_surviving}")
        lines.append(f"  Structural winner:     {audit.structural_winner.value.upper()}")
        if audit.key_uncontested_claims:
            lines.append("  Uncontested claims:")
            for c in audit.key_uncontested_claims:
                lines.append(f"    • {c}")
        if audit.key_demolished_claims:
            lines.append("  Demolished claims:")
            for c in audit.key_demolished_claims:
                lines.append(f"    • {c}")
        lines.append(f"  Summary: {audit.structural_summary}")
    else:
        lines.append("  (Layer 3 failed)")
    lines.append("")

    # Overall
    lines.append("OVERALL VERDICT")
    lines.append("-" * 40)
    lines.append(f"  {summary}")

    return "\n".join(lines)

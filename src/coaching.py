"""
Epoch-based coaching for student speakers.

After each epoch (batch of debates), this module:
  1. Extracts rubric scores, argument audit data, and judge reasoning
     specific to the two student speakers.
  2. Feeds that data — along with any prior coaching memo — to gpt-4o.
  3. Returns a cumulative CoachingMemo with actionable advice that gets
     injected into the students' preparation phase for the next epoch.

Expert speakers (Shevlin, Barez, Gardner, Floudas) receive NO coaching.
They always give their natural best arguments from their corpus.
"""

from __future__ import annotations

import src.config as cfg
from src.models import (
    CoachingMemo,
    DivisionResult,
    Side,
    SpeechOutput,
)

# The student speaker IDs that receive coaching
STUDENT_IDS = {"student_prop_2", "student_prop_3"}


# ---------------------------------------------------------------------------
# Helpers — extract student-specific data from a batch of results
# ---------------------------------------------------------------------------

def _extract_student_rubric_data(
    results: list[dict],
) -> str:
    """Build a text summary of rubric scores for student speakers across runs."""
    lines: list[str] = []

    for r in results:
        run_num = r["run_number"]
        division: DivisionResult | None = r.get("division")
        if not division or not division.rubric:
            lines.append(f"Run {run_num}: (rubric unavailable)")
            continue

        for score in division.rubric.scores:
            if score.speaker_name.startswith("Student"):
                lines.append(
                    f"Run {run_num} — {score.speaker_name}: "
                    f"argument={score.argument_strength:.1f}, "
                    f"rebuttal={score.rebuttal_quality:.1f}, "
                    f"evidence={score.evidence_grounding:.1f}, "
                    f"rhetoric={score.rhetorical_effectiveness:.1f}, "
                    f"persona={score.persona_fidelity:.1f}, "
                    f"overall={score.overall:.1f}"
                )
                lines.append(f"  Rationale: {score.rationale}")

    return "\n".join(lines) if lines else "(no rubric data)"


def _extract_student_argument_audit(
    results: list[dict],
) -> str:
    """Build a text summary of argument audit data for student claims."""
    lines: list[str] = []

    for r in results:
        run_num = r["run_number"]
        division: DivisionResult | None = r.get("division")
        if not division or not division.argument_audit:
            continue

        audit = division.argument_audit
        student_claims = [
            c for c in audit.claims if c.speaker_name.startswith("Student")
        ]
        if not student_claims:
            continue

        lines.append(f"Run {run_num}:")
        for claim in student_claims:
            status = "SURVIVES" if claim.survives else "DEMOLISHED"
            rebutters = ", ".join(claim.rebutted_by) if claim.rebutted_by else "none"
            lines.append(
                f"  [{status}] {claim.speaker_name}: \"{claim.claim}\" "
                f"(rebutted by: {rebutters})"
            )

        # Also note opposition claims that survived unrebutted
        opp_uncontested = [
            c for c in audit.claims
            if c.side == Side.OPPOSITION and c.survives and not c.rebutted_by
        ]
        if opp_uncontested:
            lines.append("  Unrebutted opposition claims:")
            for c in opp_uncontested:
                lines.append(f"    • {c.speaker_name}: \"{c.claim}\"")

    return "\n".join(lines) if lines else "(no argument audit data)"


def _extract_annotation_feedback_on_students(
    results: list[dict],
) -> str:
    """Extract annotation-based claim/rebuttal data for student speakers."""
    lines: list[str] = []

    for r in results:
        run_num = r["run_number"]
        division: DivisionResult | None = r.get("division")
        if not division or not division.annotation:
            continue

        ann = division.annotation
        # Find student claims and their fate
        for c in ann.claims:
            if any(kw in c.speaker_name.lower() for kw in ["student"]):
                # Check if demolished
                demolished = any(
                    rb.target_claim_id == c.claim_id
                    and rb.addresses_specific_logic and rb.undermines_original
                    for rb in ann.rebuttals
                )
                status = "DEMOLISHED" if demolished else "SURVIVED"
                lines.append(
                    f"Run {run_num}: [{c.claim_id}] {c.claim_text} "
                    f"({c.claim_type}, {c.specificity}) → {status}"
                )
                # Show rebuttals that hit this claim
                for rb in ann.rebuttals:
                    if rb.target_claim_id == c.claim_id:
                        lines.append(
                            f"  Rebutted by {rb.rebutting_speaker}: "
                            f"{rb.rebuttal_summary} "
                            f"(logic={rb.addresses_specific_logic}, "
                            f"new={rb.provides_new_information}, "
                            f"undermines={rb.undermines_original})"
                        )

        # Find student rebuttals (what did students successfully counter?)
        for rb in ann.rebuttals:
            if any(kw in rb.rebutting_speaker.lower() for kw in ["student"]):
                target = next(
                    (c for c in ann.claims if c.claim_id == rb.target_claim_id), None
                )
                target_text = target.claim_text if target else "(unknown claim)"
                effective = rb.addresses_specific_logic and rb.undermines_original
                lines.append(
                    f"Run {run_num}: Student rebuttal → [{rb.target_claim_id}] {target_text[:60]} "
                    f"({'EFFECTIVE' if effective else 'INEFFECTIVE'}: {rb.method})"
                )

    return "\n".join(lines) if lines else "(no annotation data for student speakers)"


def _compute_mean_student_score(results: list[dict]) -> float:
    """Compute the mean overall rubric score for student speakers across runs."""
    scores: list[float] = []
    for r in results:
        division: DivisionResult | None = r.get("division")
        if not division or not division.rubric:
            continue
        for s in division.rubric.scores:
            if s.speaker_name.startswith("Student"):
                scores.append(s.overall)
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Main coaching function
# ---------------------------------------------------------------------------

async def generate_coaching_memo(
    results: list[dict],
    epoch: int,
    prior_memo: CoachingMemo | None = None,
) -> CoachingMemo:
    """
    Analyze a batch of debate results and generate a coaching memo for student speakers.

    Args:
        results: List of per-run result dicts from the ensemble (must include 'division').
        epoch: Current epoch number (1-indexed).
        prior_memo: Coaching memo from the previous epoch, if any.

    Returns:
        A CoachingMemo with actionable advice for the next epoch.
    """

    rubric_data = _extract_student_rubric_data(results)
    audit_data = _extract_student_argument_audit(results)
    judge_data = _extract_annotation_feedback_on_students(results)
    mean_score = _compute_mean_student_score(results)

    prior_section = ""
    if prior_memo:
        prior_section = f"""
PREVIOUS COACHING MEMO (Epoch {prior_memo.epoch}):
Mean student score at that point: {prior_memo.mean_overall_score:.1f}/10

{prior_memo.full_memo}

Track what has improved since this memo and what still needs work.
"""

    prompt = f"""You are a debate coach preparing two Cambridge student speakers for
a 3v3 exhibition debate at the Cambridge Union.

Your students are:
  - Student Speaker (Prop 2): focuses on philosophical/technical arguments
  - Student Speaker (Prop 3): focuses on moral/geopolitical arguments, delivers
    the final Proposition speech

They have just completed Epoch {epoch} of practice debates.  Here is their data:

═══ RUBRIC SCORES (per speech, per run) ═══
{rubric_data}

═══ ARGUMENT AUDIT (which claims survived vs were demolished) ═══
{audit_data}

═══ CLAIM/REBUTTAL ANNOTATIONS (student-related) ═══
{judge_data}

═══ EPOCH STATISTICS ═══
Mean overall score: {mean_score:.1f}/10
Runs in this epoch: {len(results)}
{prior_section}

Based on all of this data, produce a COACHING MEMO for the next epoch.

Structure your memo as follows:

1. WHAT WORKED WELL (2–4 bullet points)
   Arguments and tactics that consistently scored well or survived
   rebuttal.  These should be reinforced.

2. WHAT NEEDS IMPROVEMENT (2–4 bullet points)
   Arguments that were repeatedly demolished or scored poorly.
   For each, explain WHY it failed and suggest a stronger alternative
   approach or defence.

3. OPPOSITION GAPS (2–4 bullet points)
   Key opposition arguments that the students consistently FAILED to
   rebut.  For each, suggest a specific counter-argument or line of
   attack.

4. STRATEGIC ADVICE FOR NEXT ROUND (3–5 bullet points)
   Concrete, actionable recommendations.  Be specific: "Instead of
   arguing X, lead with Y because Z."

5. DIVISION OF LABOUR
   Any adjustments to how Prop 2 and Prop 3 should divide their
   material to avoid overlap and maximise impact.

Rules:
- Be specific and reference the data.  Don't give generic debate advice.
- This memo will be given directly to the speakers before their next
  practice debate.  It needs to be immediately usable.
- Keep the total memo under 800 words.
- Do NOT invent arguments from outside the speakers' corpus.  Only
  recommend reframing or reprioritising material they already have."""

    response = await cfg.ANALYSIS_LLM.ainvoke(prompt)
    memo_text = response.content

    # Parse into structured CoachingMemo
    # We let the LLM generate the full memo and then extract sections.
    # The full_memo is what gets injected into the students' prep.
    memo = CoachingMemo(
        epoch=epoch,
        student_ids=list(STUDENT_IDS),
        full_memo=memo_text,
        mean_overall_score=mean_score,
        prior_memo_summary=(
            prior_memo.full_memo[:500] + "…" if prior_memo else ""
        ),
    )

    # Best-effort extraction of bullet points from sections
    _populate_structured_fields(memo, memo_text)

    return memo


def _populate_structured_fields(memo: CoachingMemo, text: str) -> None:
    """Best-effort extraction of bullet points from the coaching memo sections."""
    import re

    sections = {
        "strengths": r"(?:WHAT WORKED WELL|STRENGTHS)(.*?)(?=\d+\.\s|$)",
        "weaknesses": r"(?:WHAT NEEDS IMPROVEMENT|WEAKNESSES)(.*?)(?=\d+\.\s|$)",
        "missed_rebuttals": r"(?:OPPOSITION GAPS|MISSED REBUTTALS)(.*?)(?=\d+\.\s|$)",
        "actionable_advice": r"(?:STRATEGIC ADVICE|ACTIONABLE ADVICE)(.*?)(?=\d+\.\s|$)",
    }

    for field, pattern in sections.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            block = match.group(1)
            bullets = [
                line.strip().lstrip("-•*").strip()
                for line in block.strip().split("\n")
                if line.strip() and line.strip()[0] in "-•*"
            ]
            if bullets:
                setattr(memo, field, bullets)


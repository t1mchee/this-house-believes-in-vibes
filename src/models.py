"""
Core Pydantic data models for the debate simulation.

These schemas enforce structured output from LLM calls and define
the state that flows through the LangGraph pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Side(str, Enum):
    PROPOSITION = "proposition"
    OPPOSITION = "opposition"


# ---------------------------------------------------------------------------
# Speaker & Persona
# ---------------------------------------------------------------------------

class SpeakerProfile(BaseModel):
    """Static identity of a debate participant."""

    id: str = Field(description="Unique speaker identifier (slug)")
    name: str = Field(description="Full display name")
    side: Side
    speaking_position: int = Field(ge=1, le=3, description="1st, 2nd, or 3rd on their side")
    bio: str = Field(description="Short biographical summary")
    corpus_collection: str = Field(description="ChromaDB collection name for this speaker")


class StyleProfile(BaseModel):
    """Rhetorical style extracted from a speaker's corpus (Phase 0)."""

    speech_register: str = Field(description="e.g. 'formal academic', 'conversational', 'polemical'")
    opening_patterns: list[str] = Field(description="How they typically open a speech")
    rhetorical_devices: list[str] = Field(description="Favourite rhetorical moves")
    disagreement_style: str = Field(description="How they handle opposition")
    signature_phrases: list[str] = Field(default_factory=list)
    closing_patterns: list[str] = Field(description="How they typically close")
    raw_analysis: str = Field(default="", description="Full LLM analysis text (set after extraction)")


class SpeakerData(BaseModel):
    """Everything known about a speaker going into the debate (output of Phase 1)."""

    profile: SpeakerProfile
    style: StyleProfile
    persona_prompt: str = Field(description="The full system prompt for this speaker")
    prep_notes: str = Field(description="Speaker's self-prepared argument notes")
    retrieved_passages: list[str] = Field(
        default_factory=list,
        description="RAG-retrieved passages relevant to the motion",
    )


# ---------------------------------------------------------------------------
# Definitions & Framing (set by Prop 1, contested by Opp 1)
# ---------------------------------------------------------------------------

class TermDefinition(BaseModel):
    """A single key term and how it is defined for the debate."""
    term: str = Field(description="The key term being defined")
    definition: str = Field(description="How the speaker defines this term")


class DefinitionsFrame(BaseModel):
    """The definitional framework for the debate, set by Prop 1."""
    key_terms: list[TermDefinition] = Field(
        description="2-4 key terms from the motion and how they are defined"
    )
    scope: str = Field(
        description="What is IN scope for this debate (e.g. 'AI systems making "
                     "binding decisions under human-designed governance frameworks')"
    )
    exclusions: str = Field(
        default="",
        description="What is explicitly OUT of scope (e.g. 'fully autonomous AI "
                     "with no human oversight')"
    )
    proposition_framing: str = Field(
        description="How the Proposition frames the central question "
                    "(1-2 sentences)"
    )


class DefinitionsContestation(BaseModel):
    """Opposition's response to the Proposition's definitions."""
    accepts_definitions: bool = Field(
        description="Does the Opposition broadly accept the definitions?"
    )
    contested_terms: list[TermDefinition] = Field(
        default_factory=list,
        description="Any terms the Opposition redefines or contests"
    )
    counter_framing: str = Field(
        default="",
        description="How the Opposition reframes the central question, if at all"
    )
    agreed_ground: str = Field(
        default="",
        description="What both sides agree on (useful for identifying real clash)"
    )


# ---------------------------------------------------------------------------
# Speech & POI (Phase 2 outputs)
# ---------------------------------------------------------------------------

class ArgumentPoint(BaseModel):
    """A single argument within a speech."""

    claim: str = Field(description="The core claim being made")
    reasoning: str = Field(description="Supporting reasoning / warrant")
    evidence: Optional[str] = Field(default=None, description="Evidence from corpus, if any")
    is_rebuttal: bool = Field(default=False, description="Is this responding to an opponent?")
    rebuts_speaker: Optional[str] = Field(
        default=None, description="Name of speaker being rebutted"
    )


class SpeechOutput(BaseModel):
    """Structured output of a single debate speech."""

    # Metadata — filled in by the pipeline after generation
    speaker_id: str = Field(default="", description="Set by pipeline")
    speaker_name: str = Field(default="", description="Set by pipeline")
    side: Side = Field(default=Side.PROPOSITION, description="Set by pipeline")
    speaking_position: int = Field(default=0, description="1-6 in overall speaking order; set by pipeline")

    # Content — produced by the LLM
    opening: str = Field(description="Opening lines — hook and framing")
    arguments: list[ArgumentPoint] = Field(description="2-4 main argument points")
    closing: str = Field(description="Closing lines — peroration")
    full_text: str = Field(description="The complete speech as delivered")

    tone: str = Field(description="Self-assessed tone, e.g. 'measured but firm'")
    key_rhetorical_moves: list[str] = Field(
        default_factory=list, description="Rhetorical devices used"
    )
    word_count: int = Field(default=0)


class POI(BaseModel):
    """A Point of Information offered during a speech."""

    from_speaker: str = Field(description="Name of speaker offering the POI")
    to_speaker: str = Field(description="Name of speaker receiving the POI")
    text: str = Field(description="The POI challenge text (1-2 sentences)")
    accepted: bool
    response: Optional[str] = Field(
        default=None, description="Speaker's response if accepted"
    )
    after_argument_index: int = Field(
        description="Which argument point this POI follows"
    )


# ---------------------------------------------------------------------------
# Division / Verdict — Three-Layer Judging System (Phase 3)
# ---------------------------------------------------------------------------

# ── Layer 1: Analytical Rubric Scoring ──

class SpeechScore(BaseModel):
    """Rubric-based score for a single speech (5 dimensions + overall)."""

    speaker_name: str = Field(default="", description="Set by pipeline after LLM call")
    side: Side = Field(default=Side.PROPOSITION, description="Set by pipeline after LLM call")
    argument_strength: float = Field(ge=1, le=10, description="Logical validity and soundness of claims")
    rebuttal_quality: float = Field(ge=1, le=10, description="Engagement with opposing arguments")
    evidence_grounding: float = Field(ge=1, le=10, description="Specificity and verifiability of evidence")
    rhetorical_effectiveness: float = Field(ge=1, le=10, description="Persuasiveness, structure, clarity")
    persona_fidelity: float = Field(ge=1, le=10, description="Authenticity to the real person's voice")
    overall: float = Field(ge=1, le=10, description="Weighted composite — not a simple average")
    rationale: str = Field(description="2-3 sentence justification for the overall score")


class RecalibratedSpeechScore(BaseModel):
    """A single speech's recalibrated score from comparative analysis."""

    speaker_name: str
    rank: int = Field(ge=1, le=6, description="Force rank: 1 = best, 6 = worst")
    overall: float = Field(ge=1, le=10, description="Recalibrated overall score")
    argument_strength: float = Field(ge=1, le=10)
    rebuttal_quality: float = Field(ge=1, le=10)
    evidence_grounding: float = Field(ge=1, le=10)
    rhetorical_effectiveness: float = Field(ge=1, le=10)
    persona_fidelity: float = Field(ge=1, le=10)
    rationale: str = Field(description="Why this speaker is ranked here vs the speakers above/below")


class RecalibrationResult(BaseModel):
    """Result of the comparative recalibration of all speeches."""

    rankings: list[RecalibratedSpeechScore] = Field(
        description="All 6 speeches, ranked from best (rank=1) to worst (rank=6)"
    )


class RubricScorecard(BaseModel):
    """Aggregated rubric scores for all speeches."""

    scores: list[SpeechScore]
    prop_total: float = Field(description="Sum of Proposition speakers' overall scores")
    opp_total: float = Field(description="Sum of Opposition speakers' overall scores")
    rubric_winner: Side


# ── Layer 2: Annotation-Based Mechanical Verdict ──

class ClaimAnnotation(BaseModel):
    """A single substantive claim extracted and classified from a speech."""

    claim_id: str = Field(description="Unique ID, e.g. 'prop_1_a', 'opp_2_b'")
    speaker_name: str
    side: Side
    claim_text: str = Field(description="Brief summary of the claim (1-2 sentences)")
    claim_type: str = Field(description="One of: assertion, evidence_backed, principled")
    specificity: str = Field(description="One of: generic, specific")


class ClaimExtractionResult(BaseModel):
    """All claims extracted from the full debate."""
    claims: list[ClaimAnnotation]


class RebuttalAnnotation(BaseModel):
    """Assessment of a single rebuttal targeting a specific claim."""

    target_claim_id: str = Field(description="The claim_id being rebutted")
    rebutting_speaker: str
    rebuttal_summary: str = Field(description="Brief description of the rebuttal (1-2 sentences)")
    engagement_level: str = Field(
        description="One of: direct (engages the claim's logic), "
                    "indirect (addresses the general theme), "
                    "strawman (attacks a distorted version)"
    )
    method: str = Field(
        description="One of: counter_evidence, logical_flaw, counter_example, reassertion"
    )

    # Three objective effectiveness questions
    addresses_specific_logic: bool = Field(
        description="Does this rebuttal engage with the specific logical "
                    "structure of the original claim?"
    )
    provides_new_information: bool = Field(
        description="Does it provide new information (evidence, counter-example, "
                    "logical analysis) beyond restating the opposing position?"
    )
    undermines_original: bool = Field(
        description="After this rebuttal, does the original claim's core logic "
                    "still hold (False) or has it been undermined (True)?"
    )


class RebuttalMappingResult(BaseModel):
    """All rebuttals mapped and assessed."""
    rebuttals: list[RebuttalAnnotation]


class AnnotationVerdict(BaseModel):
    """Mechanical verdict derived from objective claim/rebuttal annotations."""

    claims: list[ClaimAnnotation] = Field(description="All substantive claims in the debate")
    rebuttals: list[RebuttalAnnotation] = Field(description="All rebuttals mapped to claims")

    # Tallies
    prop_total_claims: int = Field(default=0)
    opp_total_claims: int = Field(default=0)
    prop_surviving: int = Field(default=0, description="Prop claims not demolished")
    opp_surviving: int = Field(default=0, description="Opp claims not demolished")
    prop_demolished: int = Field(default=0, description="Prop claims effectively rebutted")
    opp_demolished: int = Field(default=0, description="Opp claims effectively rebutted")

    # Weighted scores
    prop_score: float = Field(default=0.0, description="Weighted mechanical score")
    opp_score: float = Field(default=0.0, description="Weighted mechanical score")
    winner: Side = Field(description="Side with higher mechanical score")
    margin: str = Field(default="narrow", description="narrow / clear / landslide")

    score_breakdown: str = Field(
        default="",
        description="Human-readable explanation of the scoring arithmetic"
    )


# ── (Legacy) Layer 2: Multi-Judge Panel — kept for backward compatibility ──

class JudgeVote(BaseModel):
    """A single independent judge's vote on the debate."""

    vote: Side = Field(description="proposition (AYE) or opposition (NO)")
    confidence: float = Field(ge=0, le=1, description="0.0 = coin flip, 1.0 = certain")
    key_reason: str = Field(description="Single most important reason for the vote")
    tipping_point: str = Field(description="The specific moment that decided the vote")


class PanelVerdict(BaseModel):
    """Aggregated result from N independent judge votes."""

    votes: list[JudgeVote]
    ayes: int = Field(description="Judges voting for Proposition")
    noes: int = Field(description="Judges voting for Opposition")
    winner: Side
    margin: str = Field(description="narrow / clear / landslide")
    mean_confidence: float = Field(description="Average confidence across all judges")
    agreement_ratio: float = Field(description="Fraction of judges on the winning side")


# ── Layer 2b: Engagement-Focused LLM Verdict ──

class EngagementVote(BaseModel):
    """A single judge's engagement-focused evaluation of the debate.

    The judge evaluates which *anonymized* team argued better based on
    engagement quality, not personal opinion on the topic.
    """

    better_team: str = Field(
        description="Which team argued better: 'Team A' or 'Team B'"
    )
    engagement_quality_a: float = Field(
        ge=1, le=10,
        description="How well Team A engaged with Team B's strongest arguments"
    )
    engagement_quality_b: float = Field(
        ge=1, le=10,
        description="How well Team B engaged with Team A's strongest arguments"
    )
    strongest_argument_a: str = Field(
        description="The single strongest argument made by Team A (1-2 sentences)"
    )
    strongest_argument_b: str = Field(
        description="The single strongest argument made by Team B (1-2 sentences)"
    )
    best_rebuttal: str = Field(
        description="The single best rebuttal in the entire debate — who made it "
                    "and what did they rebut (2-3 sentences)"
    )
    decisive_moment: str = Field(
        description="The exchange or argument that most tilted the debate (2-3 sentences)"
    )
    key_reason: str = Field(
        description="Why the winning team argued better — must reference specific "
                    "arguments and engagement, not general impressions (2-3 sentences)"
    )
    confidence: float = Field(
        ge=0, le=1,
        description="0.0 = genuinely too close to call, 1.0 = unambiguously clear"
    )


class EngagementVerdict(BaseModel):
    """Aggregated result from the dual-pass anonymized engagement evaluation."""

    # Individual votes (3 judges × 2 passes = 6 votes)
    votes: list[EngagementVote] = Field(
        description="All individual judge votes (from both passes)"
    )
    prop_votes: int = Field(
        description="Total votes for Proposition across both passes"
    )
    opp_votes: int = Field(
        description="Total votes for Opposition across both passes"
    )
    winner: Side
    margin: str = Field(description="narrow / clear / landslide / split")
    mean_confidence: float = Field(
        description="Mean confidence across all votes"
    )
    pass_agreement: bool = Field(
        description="True if both passes (A=Prop and A=Opp) agree on the winner"
    )
    engagement_summary: str = Field(
        default="",
        description="Summary of the engagement evaluation"
    )


# ── Layer 3: Argument Graph Audit ──

class ClaimNode(BaseModel):
    """A single claim tracked through the argument graph."""

    speaker_name: str
    side: Side
    claim: str = Field(description="The core claim")
    rebutted_by: list[str] = Field(default_factory=list, description="Speakers who challenged this claim")
    rebuttal_successful: Optional[bool] = Field(default=None, description="Did the rebuttal land?")
    survives: bool = Field(default=True, description="Does this claim stand at the end of the debate?")


class ArgumentAudit(BaseModel):
    """Structural audit of argument survival across the debate."""

    claims: list[ClaimNode]
    prop_claims_surviving: int = Field(description="Number of Proposition claims still standing")
    opp_claims_surviving: int = Field(description="Number of Opposition claims still standing")
    structural_winner: Side = Field(description="Side with more surviving claims (weighted by importance)")
    key_uncontested_claims: list[str] = Field(
        default_factory=list, description="Claims no one challenged"
    )
    key_demolished_claims: list[str] = Field(
        default_factory=list, description="Claims decisively destroyed by rebuttal"
    )
    structural_summary: str = Field(description="2-3 sentence summary of the argument flow")


# ── Combined DivisionResult ──

class DivisionResult(BaseModel):
    """Three-layer verdict combining rubric, panel, and structural analysis."""

    # Top-level outcome (backward compatible with ensemble.py)
    motion: str
    winner: Side = Field(description="Final verdict: proposition (AYE) or opposition (NO)")
    ayes: int = Field(description="Number of panel judges voting AYE")
    noes: int = Field(description="Number of panel judges voting NO")
    margin: str = Field(description="narrow / clear / landslide")
    confidence: float = Field(default=0.5, ge=0, le=1, description="Mean judge confidence")
    summary: str = Field(default="", description="One-paragraph synthesis of the verdict")

    # Three evaluation layers
    rubric: Optional[RubricScorecard] = Field(default=None, description="Layer 1: Analytical rubric scores")
    annotation: Optional[AnnotationVerdict] = Field(default=None, description="Layer 2a: Annotation-based mechanical verdict")
    engagement: Optional["EngagementVerdict"] = Field(default=None, description="Layer 2b: Engagement-focused LLM verdict (primary)")
    panel: Optional[PanelVerdict] = Field(default=None, description="Legacy Layer 2: Multi-judge panel (deprecated)")
    argument_audit: Optional[ArgumentAudit] = Field(default=None, description="Layer 3: Argument graph audit")


# ---------------------------------------------------------------------------
# Coaching / Student Learning
# ---------------------------------------------------------------------------

class CoachingMemo(BaseModel):
    """Cumulative coaching feedback for student speakers across epochs."""

    epoch: int = Field(description="Epoch number that produced this memo (1-indexed)")
    student_ids: list[str] = Field(description="IDs of the student speakers receiving coaching")

    # Structured feedback sections
    strengths: list[str] = Field(
        default_factory=list,
        description="Arguments / tactics that worked well and should be reinforced",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Arguments that were effectively rebutted — need new approach or stronger defence",
    )
    missed_rebuttals: list[str] = Field(
        default_factory=list,
        description="Opposition arguments the students failed to counter",
    )
    actionable_advice: list[str] = Field(
        default_factory=list,
        description="Specific, actionable recommendations for the next epoch",
    )
    full_memo: str = Field(
        default="",
        description="Full coaching memo text (used as strategy_directive for students)",
    )

    # Tracking
    mean_overall_score: float = Field(
        default=0.0,
        description="Mean rubric overall score for student speakers in this epoch",
    )
    prior_memo_summary: str = Field(
        default="",
        description="Summary of the previous epoch's memo (for continuity)",
    )


# ---------------------------------------------------------------------------
# Debate Run (top-level container)
# ---------------------------------------------------------------------------

class DebateRun(BaseModel):
    """A complete debate iteration (speeches + verdict)."""

    iteration: int
    speeches: list[SpeechOutput]
    pois: list[POI]
    division: Optional[DivisionResult] = None


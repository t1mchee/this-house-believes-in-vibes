# 3v3 Debate Simulation Pipeline
## Cambridge Union Exhibition Debate Format

---

## Format Rules (as modelled)

The Cambridge Union exhibition debate is the Thursday-night invited-speaker format, distinct from British Parliamentary competitive debating. The rules that govern the simulation:

| Element | Rule |
|---------|------|
| **Motion** | "This House Believes..." / "This House Would..." — a single proposition |
| **Sides** | Proposition (3 speakers) vs Opposition (3 speakers) |
| **Speaking order** | Strictly alternating: Prop 1 → Opp 1 → Prop 2 → Opp 2 → Prop 3 → Opp 3 |
| **Speech length** | ~7 minutes per speaker (modelled as ~1,200–1,500 words) |
| **Points of Information** | During any speech (except first and last minute), the opposing side may rise and offer a brief challenge (~15 seconds). Speaker may accept or decline. |
| **No team coordination** | Speakers are independent individuals. There is **no** pre-debate team huddle, no shared strategy, no assigned roles. Each speaker crafts their own speech knowing only the motion and their side assignment. |
| **No reply speeches** | The debate ends after the 6th speaker. No summaries, no closing rebuttals. |
| **Floor debate** | After the 6 main speeches, audience members may give brief floor speeches (~2 minutes each). |
| **The Division** | The audience votes by walking through "Ayes" or "Noes" doors. The side with more votes wins. |

### What this means for the pipeline

The previous architecture assumed a team-based competitive format. The exhibition format changes everything:

- **Delete**: Team strategy phase, team internal deliberation, role specialisation (Lead/Evidence/Rebuttal)
- **Delete**: Cross-examination rounds, reply speeches
- **Add**: Points of Information as mid-speech interjections
- **Add**: Floor debate simulation (optional — simulated audience voices)
- **Change**: Each speaker is a fully independent agent with no access to teammates' planning
- **Change**: The verdict is an audience vote (persuasion-based), not a judge scoring rubric
- **Change**: Later speakers must react to what came before, but they're doing so on the fly — they prepared their core arguments *before* hearing the other speeches

---

## Revised Tool Stack

| Layer | Tool | Rationale |
|-------|------|-----------|
| **Orchestration** | LangGraph | Sequential graph with POI branching — cycles only needed for recursive refinement |
| **RAG / Vector Store** | LangChain + ChromaDB | Per-speaker collections, unchanged |
| **Embeddings** | OpenAI `text-embedding-3-large` | Unchanged |
| **LLM (speakers)** | OpenAI o3 | Deep reasoning for persona-faithful speech generation |
| **LLM (POI generator)** | GPT-4o-mini | POIs are short; speed matters |
| **LLM (audience/verdict)** | OpenAI o3 | Needs sophisticated reasoning for vote prediction |
| **Structured outputs** | Pydantic models via LangChain | Enforce speech + POI schemas |
| **Tracing** | LangSmith | Track every call, compare runs |

---

## Pipeline Overview

```
                    CAMBRIDGE UNION EXHIBITION DEBATE
                    ─────────────────────────────────

 PHASE 0          PHASE 1              PHASE 2              PHASE 3
 ────────         ──────────           ──────────           ──────────
 Corpus     ───▶  Persona        ───▶  The Debate     ───▶  The Division
 Ingestion        Construction         (6 speeches          (Audience
                  + Speech Prep        + POIs)              Vote)
                                                                │
                                                                │
                                            ┌───────────────────┘
                                            ▼
                                       PHASE 4
                                       ──────────
                                       Recursive
                                       Refinement
                                       (optional)


 PHASE 2 DETAIL — The Debate:

 ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
 │ PROP 1  │───▶│  OPP 1  │───▶│ PROP 2  │───▶│  OPP 2  │───▶ ...
 │ (opens) │    │         │    │         │    │         │
 └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘
      │              │              │              │
   [POIs]         [POIs]         [POIs]         [POIs]
   from Opp       from Prop      from Opp       from Prop
```

---

## Phase 0: Corpus Ingestion (unchanged)

Same as before: per-speaker ChromaDB collections, metadata-tagged chunks (speaker_id, source_type, date, topic_tags), plus a merged cross-speaker index.

One addition: **extract rhetorical style features** during ingestion. Exhibition debate speeches are performative — wit, anecdote, and rhetorical flourish matter as much as logical rigour. For each speaker, extract:

```python
style_extraction_prompt = """Analyse this corpus by {speaker_name}. Extract:
1. Typical speech register (formal academic / conversational / polemical / 
   humorous / etc.)
2. Opening patterns (do they start with anecdotes? data? provocations? 
   personal stories?)
3. Characteristic rhetorical devices (tricolons, rhetorical questions, 
   analogies, sarcasm, understatement?)
4. How they handle disagreement (aggressive rebuttal? diplomatic 
   concession-then-counter? dismissive? evidence-heavy takedown?)
5. Signature phrases or verbal tics
6. How they typically close a speech or argument
"""
```

---

## Phase 1: Persona Construction + Speech Preparation

### Key difference: speakers prepare independently

In the exhibition format, each speaker knows only:
- The motion
- Which side they're on (Proposition or Opposition)
- Who the other speakers are (but not what they'll say)
- Their own views, knowledge, and rhetorical style

They do **not** know what arguments their teammates will make. This is critical — it means speakers may overlap, may inadvertently contradict each other, and must each independently identify what they consider the strongest arguments. This realistic redundancy is a feature, not a bug.

### Per-speaker preparation (runs in parallel for all 6)

```python
async def prepare_speaker(speaker, motion, side, all_speakers):
    # 1. Retrieve speaker's positions relevant to the motion
    relevant_passages = speaker_store.similarity_search(
        query=motion,
        k=10,
        filter={"speaker_id": speaker.id}
    )
    
    # 2. Build the persona prompt
    persona_prompt = build_persona(speaker, relevant_passages)
    
    # 3. Generate prepared speech notes
    # The speaker prepares their CORE arguments before the debate
    # (Later speakers will adapt on the fly, but they walk in 
    #  with a prepared structure)
    prep_notes = await llm.ainvoke(
        system=persona_prompt,
        messages=[{
            "role": "user", 
            "content": f"""You are preparing to speak at the Cambridge Union 
            in {'support of' if side == 'prop' else 'opposition to'} the 
            motion: "{motion}"
            
            You are speaker {speaker.position} of 3 on your side.
            The other speakers are: {format_speaker_list(all_speakers)}
            
            Prepare your core arguments and key points. You have ~7 minutes.
            You do NOT know what your teammates will argue.
            
            Output your preparation notes — the arguments you intend to 
            make, the evidence you'll draw on, and your planned structure.
            These are YOUR notes; you may adapt during the debate based 
            on what earlier speakers say."""
        }]
    )
    
    return {
        "speaker": speaker,
        "persona_prompt": persona_prompt,
        "prep_notes": prep_notes,
        "retrieved_passages": relevant_passages
    }
```

### The persona prompt (revised for exhibition style)

```
You are {speaker_name}, speaking at the Cambridge Union.

IDENTITY
{bio_and_background}

YOUR POSITIONS (from your own writings and speeches):
{rag_retrieved_passages}

YOUR STYLE
{extracted_style_profile}
You are known for: {rhetorical_characteristics}

THE SETTING
This is a Cambridge Union exhibition debate — a formal but lively setting. 
The audience is largely students and academics. Wit, clarity, and 
conviction matter. You are speaking to persuade a live audience who will 
vote with their feet at the end.

BEHAVIOURAL RULES
- You are {speaker_name}. Stay in character throughout.
- Ground your arguments in your documented positions and knowledge. 
  Do not fabricate views you do not hold.
- {speaker_name} would approach this topic by: {approach_notes}
- You may reference what previous speakers said (if you've heard them), 
  but your core arguments should be your own.
- If you have no documented position on a specific sub-point, draw on 
  your broader worldview to reason about it — as {speaker_name} would 
  — rather than inventing a position from nothing.
- You may accept or decline Points of Information. If you accept, 
  respond briefly and sharply before continuing your speech.
- Think of yourself as "{speaker_name} would argue that..." 
  (third-person framing to maintain consistency)
```

---

## Phase 2: The Debate

This is the core of the pipeline. Six sequential speeches, each one seeing all prior speeches and potentially interrupted by POIs.

### Speech generation: the key mechanism

Each speaker's prompt grows as the debate progresses. Speaker 1 sees only the motion. Speaker 4 sees speeches 1-3 and must respond to them.

```python
async def generate_speech(speaker_data, debate_state):
    speaker = speaker_data["speaker"]
    position = speaker.position  # 1-6 in speaking order
    
    # What this speaker can see
    prior_speeches = debate_state.speeches[:position - 1]
    prior_pois = debate_state.pois  # all POIs so far
    
    # Build the speech prompt
    if position == 1:
        # First proposition speaker: opens the case, no prior context
        task = f"""You are the first speaker for the Proposition. 
        Open the case for the motion: "{debate_state.motion}"
        
        Your role:
        - Define and frame the motion
        - Present your strongest 2-3 arguments
        - Set the tone for your side
        - You are speaking first; there is nothing to rebut yet.
        
        Deliver your speech as {speaker.name} would — in their voice, 
        with their characteristic style and argumentation."""
        
    elif position == 2:
        # First opposition speaker: responds to Prop 1 + presents own case
        task = f"""You are the first speaker for the Opposition against 
        the motion: "{debate_state.motion}"
        
        You have just heard the following speech from {prior_speeches[0].speaker_name}:
        ---
        {prior_speeches[0].text}
        ---
        
        Your role:
        - Respond to the strongest points made by the proposition
        - Present your own 2-3 core arguments against the motion
        - You must both rebut AND build your own case
        
        Deliver your speech as {speaker.name} would."""
        
    else:
        # Speakers 3-6: see all prior speeches, must engage with the debate
        task = f"""You are speaker {(position + 1) // 2} of 3 for the 
        {'Proposition' if speaker.side == 'prop' else 'Opposition'}.
        
        The motion is: "{debate_state.motion}"
        
        The debate so far:
        {format_transcript(prior_speeches, prior_pois)}
        
        Your preparation notes (written before hearing the other speeches):
        {speaker_data['prep_notes']}
        
        Your role:
        - Engage with what has been said — rebut key opposing arguments
        - Advance NEW arguments that haven't been made yet by your side
          (adapt your prep notes: drop points already covered by teammates, 
           strengthen points that are under attack)
        - If you're the 3rd speaker on your side, you are the final voice. 
          Drive home the most compelling case. 
          {'This is the last speech of the debate. Make it count.' if position == 6 else ''}
        
        You did NOT coordinate with your teammates beforehand. If they 
        made a point you disagree with or would frame differently, you may 
        do so — this is natural in exhibition debate.
        
        Deliver your speech as {speaker.name} would."""
    
    # Generate with structured output
    speech = await llm.ainvoke(
        system=speaker_data["persona_prompt"],
        messages=[{"role": "user", "content": task}],
        response_format=SpeechOutput  # see schema below
    )
    
    return speech
```

### Speech output schema

```python
from pydantic import BaseModel
from typing import List, Optional

class ArgumentPoint(BaseModel):
    claim: str                    # The core claim
    reasoning: str                # Why this is true (warrant)
    evidence: Optional[str]       # Supporting evidence from corpus
    is_rebuttal: bool            # Is this responding to an opponent?
    rebuts_speaker: Optional[str] # If rebuttal, who is it responding to?

class SpeechOutput(BaseModel):
    opening: str                  # Opening lines (hook, framing)
    arguments: List[ArgumentPoint] # 2-4 main argument points
    closing: str                  # Closing lines (peroration)
    full_text: str               # The complete speech as delivered
    tone: str                    # Self-assessed tone (e.g. "measured but firm")
    key_rhetorical_moves: List[str]  # What devices were used
```

### Points of Information (POIs)

POIs are the exhibition format's mechanism for real-time adversarial pressure. After each "minute" of a speech (modelled as after each argument point), the opposing side may offer a POI.

```python
async def generate_pois(speech: SpeechOutput, opposing_speakers: list, 
                        debate_state) -> List[POI]:
    """
    For each argument point in the speech, decide whether an opposing 
    speaker rises on a POI and what they say.
    """
    pois = []
    
    for i, argument in enumerate(speech.arguments):
        # Skip first and last argument (protected time)
        if i == 0 or i == len(speech.arguments) - 1:
            continue
        
        # Pick which opposing speaker rises (if any)
        # Probability-weighted: earlier speakers who've already spoken 
        # are more likely to rise (they have established positions)
        poi_decision = await poi_model.ainvoke(f"""
        During a Cambridge Union debate, the current speaker just said:
        "{argument.claim} — {argument.reasoning}"
        
        The opposing speakers are: {format_opponents(opposing_speakers)}
        
        Should any opposing speaker rise on a Point of Information?
        Consider:
        - Is this point vulnerable to a sharp challenge?
        - Would {opponent_name} specifically want to challenge this?
        - POIs should be brief (1-2 sentences), pointed, and designed 
          to wrong-foot the speaker.
        
        Respond with either NO_POI or a POI from a specific speaker.""")
        
        if poi_decision.offers_poi:
            # Does the current speaker accept?
            acceptance = await decide_poi_acceptance(
                speaker=current_speaker,
                poi=poi_decision,
                speech_position=i,
                total_pois_accepted=len([p for p in pois if p.accepted])
            )
            
            if acceptance.accepted:
                # Generate the speaker's response to the POI
                response = await generate_poi_response(
                    speaker=current_speaker,
                    poi=poi_decision.text,
                    current_argument=argument
                )
                pois.append(POI(
                    from_speaker=poi_decision.speaker,
                    to_speaker=current_speaker.name,
                    text=poi_decision.text,
                    accepted=True,
                    response=response.text,
                    after_argument=i
                ))
            else:
                pois.append(POI(
                    from_speaker=poi_decision.speaker,
                    to_speaker=current_speaker.name,
                    text=poi_decision.text,
                    accepted=False,
                    response=None,
                    after_argument=i
                ))
    
    return pois
```

### Full debate execution loop

```python
async def run_debate(debate_state: DebateState) -> DebateState:
    speaking_order = [
        debate_state.prop_speakers[0],  # Prop 1
        debate_state.opp_speakers[0],   # Opp 1
        debate_state.prop_speakers[1],  # Prop 2
        debate_state.opp_speakers[1],   # Opp 2
        debate_state.prop_speakers[2],  # Prop 3
        debate_state.opp_speakers[2],   # Opp 3
    ]
    
    for i, speaker in enumerate(speaking_order):
        # Refresh RAG context based on what's been discussed
        if i > 0:
            recent_topics = extract_topics(debate_state.speeches)
            speaker.retrieved_passages = refresh_rag(
                speaker_id=speaker.id,
                queries=[debate_state.motion] + recent_topics,
                k=8
            )
        
        # Generate the speech
        speech = await generate_speech(
            speaker_data=speaker,
            debate_state=debate_state
        )
        
        # Generate POIs during the speech
        opposing_side = (debate_state.opp_speakers if speaker.side == 'prop' 
                        else debate_state.prop_speakers)
        pois = await generate_pois(
            speech=speech,
            opposing_speakers=opposing_side,
            debate_state=debate_state
        )
        
        # If any POIs were accepted, regenerate the speech with POI 
        # responses woven in (or insert them at the right points)
        if any(p.accepted for p in pois):
            speech = await weave_pois_into_speech(speech, pois, speaker)
        
        # Update state
        debate_state.speeches.append(speech)
        debate_state.pois.extend(pois)
    
    return debate_state
```

---

## Phase 3: The Division (Verdict)

In the exhibition format, the verdict isn't a judge's score — it's a **persuasion-based audience vote**. The question isn't "who made better arguments?" but "did the Proposition or Opposition persuade more people?"

### Modelling the audience vote

```python
async def simulate_division(debate_state: DebateState) -> DivisionResult:
    """
    Simulate the audience division (vote).
    
    We model this in two ways and compare:
    1. Direct verdict: an LLM acting as a "median audience member"
    2. Multi-perspective panel: several simulated audience personas
    """
    
    full_transcript = format_full_transcript(
        debate_state.speeches, debate_state.pois
    )
    
    # --- Method 1: Direct verdict ---
    direct = await verdict_model.ainvoke(f"""
    You have just attended a Cambridge Union exhibition debate on the 
    motion: "{debate_state.motion}"
    
    Here is the full debate transcript:
    {full_transcript}
    
    You are a thoughtful audience member at the Cambridge Union — likely 
    a student or academic. You came into this debate genuinely undecided.
    
    Based purely on what you heard tonight, which side was more 
    persuasive? Vote AYE (Proposition) or NO (Opposition).
    
    Before voting, analyse:
    1. What were the core tensions in this debate?
    2. Which arguments landed most effectively?
    3. Which rebuttals were most damaging?
    4. Were there any moments that shifted the debate decisively?
    5. Which speaker was most compelling, and why?
    
    Then cast your vote and explain your reasoning.""")
    
    # --- Method 2: Multi-perspective panel ---
    # Simulate 5-7 audience members with different priors
    audience_personas = [
        "A left-leaning PPE student who came in slightly sympathetic to the Proposition",
        "A conservative-leaning Law student who came in slightly sympathetic to the Opposition",
        "A natural sciences PhD student with no strong prior on this topic",
        "A politically engaged History student who has written on this topic",
        "An international student unfamiliar with the UK framing of this issue",
    ]
    
    votes = []
    for persona in audience_personas:
        vote = await verdict_model.ainvoke(f"""
        You are: {persona}
        You have just heard this debate at the Cambridge Union:
        {full_transcript}
        
        Vote AYE or NO. Briefly explain what persuaded you.""")
        votes.append(vote)
    
    # --- Aggregate ---
    return DivisionResult(
        motion=debate_state.motion,
        direct_verdict=direct,
        panel_votes=votes,
        ayes=count_ayes(votes),
        noes=count_noes(votes),
        winner="Proposition" if count_ayes(votes) > count_noes(votes) else "Opposition",
        core_tensions=direct.core_tensions,
        decisive_moments=direct.decisive_moments,
        most_compelling_speaker=direct.most_compelling_speaker,
        analysis=direct.full_analysis
    )
```

### Division output schema

```python
class DivisionResult(BaseModel):
    motion: str
    winner: str                        # "Proposition" or "Opposition"
    ayes: int                          # Simulated vote count
    noes: int
    margin: str                        # "narrow" / "clear" / "landslide"
    
    # Analytical outputs
    core_tensions: List[str]           # The 2-3 fundamental disagreements
    decisive_moments: List[str]        # Turning points in the debate
    strongest_argument_prop: str       # Best argument for Proposition
    strongest_argument_opp: str        # Best argument for Opposition
    most_compelling_speaker: str       # Who won the room?
    weakest_link: str                  # Who underperformed?
    
    # Per-speaker assessment
    speaker_performances: List[SpeakerAssessment]
    
    # For recursive refinement
    feedback: Dict[str, List[str]]     # speaker_id -> improvement notes

class SpeakerAssessment(BaseModel):
    speaker_name: str
    effectiveness: float               # 1-10
    persona_fidelity: float            # 1-10 (did they sound like themselves?)
    key_contribution: str              # What they uniquely added
    missed_opportunity: Optional[str]  # What they should have said
```

---

## Phase 4: Recursive Refinement (adapted)

Recursion in exhibition format is different. You're not refining a team strategy — you're asking: "Given the verdict feedback, what would a *better* simulation of each speaker look like?"

### What changes between iterations

1. **Better RAG retrieval**: The verdict identifies missed arguments and weak points. Use these as new retrieval queries against each speaker's corpus to surface material they should have used.

2. **Improved speech adaptation**: Later speakers can be re-prompted to respond more sharply to specific arguments that the verdict identified as decisive.

3. **POI quality**: If the verdict notes that key challenges went unraised, the POI generation can be re-weighted to target those specific weaknesses.

4. **Persona correction**: If the verdict flags that a speaker drifted from their documented style or positions, the persona prompt is reinforced with additional grounding material.

### What stays fixed

- The motion and side assignments
- The speaking order
- Each speaker's core identity and corpus
- The audience composition

### Termination criteria

```python
def should_terminate(debate_state: DebateState) -> bool:
    if debate_state.iteration >= 3:
        return True  # Hard cap
    
    if debate_state.iteration >= 2:
        # Check if verdict is stable
        prev_winner = debate_state.history[-2].division.winner
        curr_winner = debate_state.history[-1].division.winner
        prev_tensions = set(debate_state.history[-2].division.core_tensions)
        curr_tensions = set(debate_state.history[-1].division.core_tensions)
        
        # If same winner AND same core tensions, we've converged
        if prev_winner == curr_winner and prev_tensions == curr_tensions:
            return True
    
    # Check argument novelty
    prev_args = get_all_claims(debate_state.history[-2].speeches)
    curr_args = get_all_claims(debate_state.history[-1].speeches)
    novelty = 1.0 - semantic_similarity(prev_args, curr_args)
    
    if novelty < 0.1:  # Less than 10% new material
        return True
    
    return False
```

---

## LangGraph State & Graph Definition

```python
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

class DebateState(TypedDict):
    # Configuration
    motion: str
    prop_speakers: List[SpeakerProfile]  # 3 speakers
    opp_speakers: List[SpeakerProfile]   # 3 speakers
    
    # Persona data (Phase 1)
    speaker_data: Dict[str, SpeakerData]  # speaker_id -> persona + prep
    
    # Debate transcript (Phase 2) 
    speeches: List[SpeechOutput]          # 6 speeches in order
    pois: List[POI]                       # all POIs
    
    # Division result (Phase 3)
    division: Optional[DivisionResult]
    
    # Recursive refinement
    iteration: int
    history: List[DebateRun]              # prior iterations
    should_terminate: bool

# Build the graph
graph = StateGraph(DebateState)

# Nodes
graph.add_node("ingest", corpus_ingestion_node)      # Phase 0 (once)
graph.add_node("prepare", parallel_speaker_prep_node) # Phase 1
graph.add_node("speech_1", speech_node)               # Prop 1
graph.add_node("speech_2", speech_node)               # Opp 1
graph.add_node("speech_3", speech_node)               # Prop 2
graph.add_node("speech_4", speech_node)               # Opp 2
graph.add_node("speech_5", speech_node)               # Prop 3
graph.add_node("speech_6", speech_node)               # Opp 3
graph.add_node("division", division_node)             # Phase 3
graph.add_node("refine_check", check_termination)     # Phase 4 gate
graph.add_node("refine", refinement_node)             # Phase 4

# Edges — linear debate flow
graph.add_edge("ingest", "prepare")
graph.add_edge("prepare", "speech_1")
graph.add_edge("speech_1", "speech_2")
graph.add_edge("speech_2", "speech_3")
graph.add_edge("speech_3", "speech_4")
graph.add_edge("speech_4", "speech_5")
graph.add_edge("speech_5", "speech_6")
graph.add_edge("speech_6", "division")
graph.add_edge("division", "refine_check")

# Conditional: continue refining or stop
graph.add_conditional_edges(
    "refine_check",
    lambda state: "refine" if not state["should_terminate"] else "end",
    {"refine": "refine", "end": END}
)
graph.add_edge("refine", "prepare")  # Loop back to re-prepare

graph.set_entry_point("ingest")
app = graph.compile()
```

---

## Key Design Differences from the Previous Architecture

| Aspect | Previous (team competitive) | Revised (exhibition) |
|--------|---------------------------|---------------------|
| **Speaker coordination** | Team strategy session, role allocation | None — speakers prepare independently |
| **Speech structure** | Assigned roles (lead, evidence, rebuttal) | Each speaker builds their own case |
| **Speaking format** | Rounds (opening, rebuttal, cross-exam, closing) | 6 alternating speeches, no rounds |
| **Real-time interaction** | Cross-examination phase | Points of Information mid-speech |
| **Redundancy** | Avoided by team planning | Expected and natural — speakers may overlap |
| **Verdict** | Multi-judge scoring panel | Audience vote (persuasion-based) |
| **What matters most** | Argument quality + technical debating | Persuasion + rhetorical performance + conviction |
| **Reply speeches** | Yes | No |
| **Floor debate** | No | Yes (optional to simulate) |

---

## Evaluation

### Persona fidelity metrics (unchanged in importance)

1. **Corpus grounding rate**: % of claims traceable to the speaker's actual corpus
2. **Style match**: LLM-as-judge comparison of generated speech style vs real speeches
3. **Position accuracy**: Do simulated positions match documented positions?
4. **Differentiation**: Pairwise cosine distance between all 6 speakers' outputs (should be high)

### Exhibition-format-specific metrics

5. **Adaptation quality**: Do later speakers (positions 3-6) meaningfully engage with what came before, or do they just deliver their prep notes unchanged?
6. **POI realism**: Are Points of Information sharp, targeted, and well-timed?
7. **Persuasive arc**: Does the debate build tension and reach a natural climax, or does it feel like 6 disconnected monologues?
8. **Vote prediction accuracy**: If you have real Cambridge Union division results for similar motions, does the simulated vote predict the same direction?

### Ground truth (if available)

The Cambridge Union's YouTube channel has hundreds of full debate recordings. For any debate where you have the same speakers on the same topic, you can compare:
- Did the simulation identify the same core tensions?
- Did each speaker make similar arguments to their real speech?
- Did the simulation predict the correct division outcome?

---

## Cost & Latency

| Phase | LLM Calls | Estimated Cost | Latency |
|-------|-----------|---------------|---------|
| Corpus ingestion (one-time) | 6 profiles + embeddings | ~$2-5 | Minutes |
| Speaker prep (6 parallel) | 6 calls + 6 RAG queries | ~$0.50 | 15s (parallel) |
| Debate (6 speeches + POIs) | 6 speech calls + ~8 POI calls | ~$4-8 | 3-5 min (sequential) |
| Division (5-7 audience votes) | 7 calls | ~$1-2 | 20s (parallel) |
| **Total (single run)** | | **~$8-16** | **~5 min** |
| **Total (3 iterations)** | | **~$25-50** | **~15 min** |

---

## Implementation Order

1. **Week 1**: Corpus ingestion + persona extraction for all 6 speakers
2. **Week 2**: Single speech generation — get one speaker producing a realistic 7-minute exhibition speech with RAG grounding
3. **Week 3**: Full 6-speech debate with sequential context accumulation — no POIs yet
4. **Week 4**: Add POI system and speech weaving
5. **Week 5**: Division simulation + recursive refinement loop
6. **Week 6**: Evaluation against real Cambridge Union debates (YouTube transcripts)

# Speaker Data Guide

## Directory Structure

Each speaker gets a folder under `data/speakers/`. The folder name becomes
their `speaker_id` (use lowercase, underscores, no spaces).

```
data/speakers/
├── jane_smith/
│   ├── bio/
│   │   └── bio.md              ← SHORT bio + key positions (always include this)
│   │
│   ├── speeches/               ← Transcripts of talks, lectures, debate appearances
│   │   ├── oxford_ai_lecture_2024.md
│   │   ├── cambridge_union_oct2023.md
│   │   └── ted_talk_2022.pdf
│   │
│   ├── interviews/             ← Interview transcripts, podcast appearances, Q&As
│   │   ├── bbc_hardtalk_2024.md
│   │   └── economist_interview.pdf
│   │
│   └── writings/               ← Articles, op-eds, book excerpts, papers
│       ├── guardian_oped_ai_regulation.md
│       ├── governing_the_machine_ch1.pdf
│       └── ft_column_2024.md
│
├── john_doe/
│   ├── bio/
│   │   └── bio.md
│   ├── speeches/
│   ├── interviews/
│   └── writings/
│
└── ... (6 speakers total)
```

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Markdown | `.md` | **Preferred** — cleanest for the pipeline |
| Plain text | `.txt` | Works fine |
| PDF | `.pdf` | Automatically extracted via `unstructured` |

## Tips for Good Corpus Data

1. **More is better** — aim for 10-20 documents per speaker if possible
2. **Variety matters** — mix speeches, interviews, and writings to capture
   different registers (formal talks vs casual interviews)
3. **Topical relevance** — include material on the debate topic AND on
   adjacent topics (the RAG system will find what's relevant)
4. **The bio is critical** — this is the backbone of the persona prompt.
   Include their background, key positions, and known rhetorical style.
5. **YouTube transcripts** — if a speaker has talks/debates on YouTube, you
   can pull transcripts using the `youtube-transcript-api` package (already
   installed). See `src/corpus/youtube.py`.

## Markdown File Format

For best results, structure your `.md` files with some metadata at the top:

```markdown
# Title of Speech / Article / Interview

- **Speaker**: Jane Smith
- **Date**: 2024-03-15
- **Source**: BBC HARDtalk
- **Type**: Interview transcript

---

[Full text content here...]
```

The metadata header is optional but helps with retrieval quality.


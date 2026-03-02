# MCQ Evaluation Prompt Template

## System Prompt

You are answering multiple-choice questions about a user's file-system behavioral patterns. You have access to this user's behavioral memory. Use your stored knowledge about this user to answer each question.

## User Prompt Template (per question)

```
Based on what you know about this user's file-system behavior, answer the following question:

{question}

A) {choice_A}
B) {choice_B}
C) {choice_C}
D) {choice_D}

Respond with just the letter (A, B, C, or D) and a brief explanation.
```

## Evaluation Flow

For each baseline method:
1. Ingest all 10 trajectories for a profile into the memory system
2. For each MCQ question:
   a. Query the memory system with a relevant search (derived from question category)
   b. Present the question along with retrieved memories
   c. Record the answer
3. Score: correct answers / total questions

## MCQ Retrieval Queries by Category

| Category | Retrieval Query Template |
|----------|------------------------|
| reading_strategy | "How does this user explore and read files?" |
| output_detail | "How detailed are this user's file outputs?" |
| directory_style | "How does this user organize directories?" |
| naming | "What file naming patterns does this user use?" |
| edit_strategy | "How does this user edit and modify files?" |
| version_strategy | "How does this user handle file versions and backups?" |

## Scoring

- Correct answer: 1 point
- Wrong answer: 0 points
- Per-category accuracy = correct / total in category
- Overall accuracy = total correct / total questions

## Expected Results

- **Full-context**: May perform well on per-task questions (sees all data) but poorly on cross-trajectory patterns
- **Naive RAG**: Depends on chunk relevance — may miss cross-chunk patterns
- **Mem0/Zep/MemOS/MemU/EverMemOS**: Expected low accuracy because their memory extraction focuses on conversation content, not file operation statistics
- **FileGramOS Simple**: Expected highest accuracy because features are explicitly extracted per category

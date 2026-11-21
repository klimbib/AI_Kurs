# RAG over transistor datasheets: report

Setup: 32 PDF/Markdown files, 28 test questions, hybrid retrieval + rerank, answer model `openai/gpt-4o-mini`, bot prompt `v1`, judge prompt `judge-v2`.

## Metrics

### Ingest
| Metric | Value |
|---|---|
| Files | 32 |
| Exact duplicates skipped | 1 (`DMN3404L_Diodes.pdf` = `DMN3404L_Diodes copy.pdf`) |

### Retrieval (28 questions, `output/search/eval.json`)
| Method | Recall | MRR | Chunk recall | Latency / query | Missed questions |
|---|---|---|---|---|---|
| vector | 0.79 | 0.67 | 0.62 | 35 ms | q05, q15, q16, q19, q20 |
| bm25 | 0.92 | 0.74 | 0.82 | 0.5 ms | q03, q19 |
| hybrid | 0.83 | 0.76 | 0.69 | 34 ms | q15, q16, q19, q20 |
| hybrid + rerank | **0.96** | **0.81** | 0.79 | 1867 ms | q19 |

### Answers (`evals/evals.json`, `output/rag/answers.json`)
| Metric | Value |
|---|---|
| Pass | 0.75 (21 / 28) |
| Facts | 0.75 |
| Cited | 0.75 |
| Judge (correct) | 0.71 |
| Grounded | 0.74 |
| Abstain (correct refusals) | 1.00 |
| False abstain | 0.21 |
| Answered / retrieval hit | 19 / 28, 23 / 28 |
| Tokens in / out | 32,183 / 1,096 |
| Cost | $0.0055 for 28 questions |
| Time | 82 s |

| Question kind | Pass rate |
|---|---|
| deutsch | 1.00 |
| no_answer | 1.00 |
| paraphrase | 0.78 |
| access | 0.75 |
| exact | 0.75 |
| yes_no | **0.25** |

Failed: q08, q09, q11, q19, q20, q21, q24.

### Judge calibration (14 hand-labelled answers, `judge-v2`)
| Judge | Agreement | False pass | False fail |
|---|---|---|---|
| deterministic | 0.93 | 0.00 | 0.07 |
| correct | 0.86 | 0.00 | 0.14 |
| grounded | **0.31** | 0.31 | 0.38 |

## Errors and difficulties

1. **False abstains (5 of 7 failures).** The bot answered "I could not find this" although the right chunk was retrieved: q09, q11 (list questions over the `transistor_max_*.md` tables) and q20, q21 (yes/no questions). q19 was also a refusal, but there retrieval missed the chunk.
2. **Yes/no questions are the weakest kind (0.25).** The v1 prompt only allows answers that are literally in the sources, so a conclusion such as "50 V is not more than 100 V" is refused. Prompt v2 allows Yes/No answers; it was tested, but did not give an essential improvement and was not made the default.
3. **The grounded judge is unreliable.** Agreement is 0.31 even with `judge-v2` (0.23 with `judge-v1`), with false passes and false 

## Conclusion and improvements

The pipeline retrieves well (recall 0.96 with rerank) but answers too cautiously: the main loss is in the generation prompt, not in retrieval. Pass rate is 0.75.

Technical data sheets are well stuctured, but contains the lot of nummerical information and not contains the  good verbal description. The lack of verbal imformation lead to wrong " not find" answer, specially in yes/no category. Possible improvement: give more context to technical data, build  dada bank with nummerical information and perform determitistic preselection, if query requests some analyse ot selection based on nummerical data.   




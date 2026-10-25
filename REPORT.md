# Datasheet extraction: results of runs V1-V5

All runs use `top_p = 1` and the prompt `prompts/extract_V1.txt`, except V4 (`extract_V2.txt`, which adds two rules about values that are missing or not exactly in the text), 26 datasheets (first 2 pages each).
Accuracy is measured against `data/golden.json` (11 of the 26 sheets, 10 fields each = 110 values).

| Run | Model | Temperature | Prompt file | PDF read method | Latency p50 / p95 (s) | Price per 1000 sheets | Deviation from golden | Failed (Pydantic) |
|---|---|---|---|---|---|---|---|---|
| V1 | openai/gpt-4o-mini | 0 | extract_V1.txt | pymupdf* | 1.9 / 2.9 | $0.39 | 6.4 % | 0 % |
| V1 | llama3.2:3b (local) | 0 | extract_V1.txt | pymupdf* | 73.7 / 103.5 | $0.00 | 20.0 % | 42.3 % |
| V2 | openai/gpt-4o-mini | 1 | extract_V1.txt | pymupdf* | 2.4 / 4.7 | $0.41 | 10.0 % | 3.8 % |
| V2 | minimax/minimax-m3.1-flash | 1 | extract_V1.txt | pymupdf* | 1.2 / 3.2 | $0.32 | 5.5 % | 0 % |
| V3 | openai/gpt-4o-mini | 0 | extract_V1.txt | pymupdf* | 2.1 / 2.6 | $0.39 | 6.4 % | 0 % |
| V3 | minimax/minimax-m3.1-flash | 0 | extract_V1.txt | pymupdf* | 1.4 / 4.1 | $0.33 | 4.5 % | 0 % |
| V4 | openai/gpt-4o-mini | 0 | extract_V2.txt | pymupdf | 1.9 / 2.5 | $0.39 | 7.3 % | 0 % |
| V4 | minimax/minimax-m3.1-flash | 0 | extract_V2.txt | pymupdf | 1.5 / 4.1 | $0.32 | 4.5 % | 0 % |
| V5 | openai/gpt-4o-mini | 0 | extract_V1.txt | unstructured (fast) | 4.0 / 5.7 | $0.50 | 11.8 % | 3.8 % |
| V5 | minimax/minimax-m3.1-flash | 0 | extract_V1.txt | unstructured (fast) | 3.5 / 5.8 | $0.30 | 5.5 % | 0 % |

\* The log of this run does not name the reader. `pymupdf` was the only reader at that time.

## Column definitions

- **Deviation from golden:** share of the 110 golden values (11 sheets x 10 fields) that differ from `golden.json`. It is `100 % - TOTAL rate` from the log.
- **Failed (Pydantic):** share of all 26 sheets whose answer was still invalid after the correction round. `corrected` sheets count as valid.
- **Price per 1000 sheets:** measured tokens x list price (gpt-4o-mini $0.15 / $0.60, MiniMax M3.1 Flash $0.09 / $0.50 per 1M tokens, input / output). The prices come from the OpenRouter page and were not re-checked.
- **Latency:** total time per sheet including a possible correction round.

## Observations

- **MiniMax M3.1 Flash** has the lowest deviation (4.5 % at temperature 0), the lowest price and the lowest latency. It needs the correction round for 23-27 % of the sheets (V3, V4) but ends with 0 % failed.
- **gpt-4o-mini** reaches 6.4 % deviation with almost no corrections.
- **Prompt V2 (V4)** does not help: gpt-4o-mini gets slightly worse (7.3 % vs 6.4 %), MiniMax stays at 4.5 %. Compared with V3, the only change in V4 is the prompt, so this difference comes from the prompt.
- **Temperature 1 (V2)** is worse than temperature 0 for gpt-4o-mini (10.0 % vs 6.4 % deviation, one failed sheet). MiniMax is not clearly affected (5.5 % vs 4.5 %).
- **unstructured (V5)** is worse than pymupdf for gpt-4o-mini (11.8 % vs 6.4 %), more expensive (more input tokens, about 2,860 vs 2,140 per sheet) and about twice as slow. For MiniMax it is cheaper but equal or slightly worse in accuracy (5.5 % vs 4.5 %).
- **llama3.2:3b** is free but slow (p50 74 s) and has the worst accuracy and the most failed sheets.

## Limits

- Only 11 sheets are in the golden file. One wrong value changes the deviation by 0.9 percentage points, so differences of a few percent are within the noise.
- p95 over 26 sheets is almost the second slowest value.
- Each run was done once. At temperature 0 the models are almost deterministic, but not guaranteed.

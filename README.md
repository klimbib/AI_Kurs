# Extracting component parameters from datasheets (PDF) with an LLM

`main.py` reads PDF datasheets of electronic components (transistors), asks an LLM to return JSON
(`part_number`, `component`, `type`, temperature, max voltage, max current), validates the answer
with Pydantic and compares the result with the reference file `data/golden.json`.
Every run produces a log with accuracy, latency, cost and the share of valid answers.

## Requirements

- Python 3.10 or newer
- Dependencies from `requirements.txt`: `openai`, `python-dotenv`, `pymupdf`, `pydantic`
- An OpenRouter API key (for `openai/gpt-4o-mini` and `minimax/minimax-m3.1-flash`)
- Optional:
  - [Ollama](https://ollama.com) for local models (for example `llama3.2:3b`)
  - `unstructured[pdf]` to read PDFs with `unstructured` (large package, pulls in `torch`)

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your key:

```
OPENROUTER_API_KEY=sk-or-...
```

For local models:

```powershell
ollama pull llama3.2:3b
```

For `unstructured` (optional): uncomment the line in `requirements.txt` or run
`pip install "unstructured[pdf]"`. The `hi_res` strategy also needs Poppler and Tesseract installed.

## Running

```powershell
python main.py
```

The script processes all `data/*.pdf` files for every model in `TARGETS`, prints the result for each file
and saves:

- `results/<OUT_DIR>/results.json`: the model answers (with status, tokens and latency)
- `results/<OUT_DIR>/log.txt`: run settings, share of valid answers, latency p50/p95,
  cost per 1000 documents, accuracy per category and a list of deviations from `golden.json`

If you do not change `OUT_DIR`, a new run overwrites the previous log. Change it before each experiment.

## Settings

The main parameters are at the top of `main.py`:

| Parameter | Meaning |
|---|---|
| `PROMPT_FILE` | prompt file from `prompts/`, for example `prompts/extract_V1.txt` |
| `OUT_DIR` | output folder, for example `results/V6` |
| `PAGES` | how many of the first PDF pages to read (default 2) |
| `PDF_READER` | `"pymupdf"` (default) or `"unstructured"` |
| `UNSTRUCTURED_STRATEGY` | `"fast"` (text only) or `"hi_res"` (layout and tables); only used with `unstructured` |
| `TARGETS` | list of models: name, `base_url`, key, model, timeout in seconds |
| `PRICES` | price per 1M tokens (input, output) used for the cost calculation |
| `SHEETS` | number of documents the cost is normalised to in the log (1000) |

Sampling parameters (`temperature`, `top_p`) are in `config.json`: the defaults are under `default`,
and a single model can override them under `models`, for example:

```json
{
  "sampling": {
    "default": {"temperature": 0, "top_p": 1},
    "models": {"minimax/minimax-m3.1-flash": {"temperature": 0.3}}
  }
}
```

To add a model, add a line to `TARGETS` and its price to `PRICES`. For a local model the price is `(0.0, 0.0)`.

## Project structure

```
main.py          main script
config.json      sampling parameters (temperature, top_p)
schema.json      template of the structure of one result (field descriptions)
requirements.txt dependencies
REPORT.md        comparison table of runs V1-V5
data/            PDF datasheets and golden.json (reference for 11 files)
prompts/         prompt files
results/         run results (results.json, log.txt)
```

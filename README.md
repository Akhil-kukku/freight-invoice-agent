# Freight Invoice Validation Agent

A multi-agent AI pipeline that automates freight invoice auditing — extracting structured data from raw invoices, validating charges against agreed carrier rates, and drafting escalation emails for any discrepancy found. Built to mirror a real back-office workflow used by logistics coordinators and billing specialists.

## What it does

Three specialized agents collaborate through a LangGraph state machine to process a single invoice end to end, with no human intervention unless something is flagged:

1. **Extractor Agent** — reads raw invoice text and pulls out structured fields (carrier, invoice number, lane, weight, charged amount) as JSON.
2. **Validator Agent** — looks up the matching lane in a rate card, calculates any overcharge in code (not in the LLM, to avoid arithmetic hallucination), and uses the LLM only to phrase a clear, human-readable reason for the result.
3. **Escalation Agent** — if a discrepancy is found, drafts a professional email to the carrier flagging the issue and requesting correction. If the invoice is clean, it skips this step automatically.

## Example run

**Input invoice:**
```
Carrier: FastFreight LLC
Lane: LA to Chicago
Weight: 38000 lbs
Charge: $1,450.00
```

**Agreed rate card:** $1,200 for this lane, up to 40,000 lbs

**Output:**
```
Status: FLAGGED
Reason: Invoice exceeds the agreed rate by $250.00.
Overcharge: $250

Escalation email drafted automatically, addressed to the carrier,
requesting review and correction.
```

A second invoice that matches the rate card exactly is correctly auto-approved with no escalation, demonstrating the agent discriminates between valid and invalid charges rather than flagging everything.

## Why I split math from language

Early versions asked the LLM to both calculate the overcharge and explain it in one step. With a small local model (Qwen 2.5 3B), this produced confident but factually wrong numbers — a classic LLM arithmetic hallucination. I refactored the validator so all numeric comparison happens in plain Python, and the LLM is only responsible for turning a verified result into natural language. This is the same separation of concerns used in production agent systems: deterministic logic for anything that must be correct, LLM reasoning for anything that benefits from language flexibility.

## Architecture

```
Invoice text
     │
     ▼
┌─────────────┐
│  Extractor   │  → structured JSON (carrier, lane, weight, charge)
└─────┬────────┘
      ▼
┌─────────────┐
│  Validator   │  → rate card lookup + Python math + LLM-phrased reason
└─────┬────────┘
      ▼
┌─────────────┐
│  Escalation  │  → drafts email only if flagged, skips if approved
└──────────────┘
```

Orchestration is handled by **LangGraph**, with each agent as a node and a linear edge path (`extractor → validator → escalation → END`). State (the invoice data, validation result, and email draft) flows through a shared `TypedDict` between nodes.

## Tech stack

- **LLM**: Ollama running Qwen 2.5 (3B), fully local — also tested against Google Gemini's free API tier, with a one-line model swap to switch providers
- **Orchestration**: LangGraph (multi-agent state machine)
- **Language**: Python 3.12
- **Data**: JSON rate card, plain-text invoice samples (designed to extend easily to real OCR'd PDFs)

## Running it locally

```bash
# 1. Install Ollama and pull a model
ollama pull qwen2.5:3b

# 2. Install Python dependencies
pip install ollama langgraph

# 3. Run the pipeline against sample invoices
python agent_ollama.py
```

No API keys or paid services required — this runs entirely offline once Ollama and the model are installed.

## What I'd add next

- Real OCR ingestion (Tesseract / AWS Textract) so it accepts scanned PDF invoices instead of plain text
- RAG over a larger rate card using a vector database (ChromaDB) instead of exact-match lookup, to handle fuzzy lane naming
- A FastAPI endpoint so the pipeline can be triggered by an inbound email or webhook
- An evaluation harness with a larger set of synthetic invoices to measure detection accuracy systematically
- Automatic provider fallback (Gemini → Ollama) for resilience when a rate limit is hit

## Why I built this

This project was built to demonstrate the core skills behind an AI Agent Developer role in logistics tech: designing multi-agent systems with LangGraph, working with LLM APIs (Claude, Gemini, and local open-weight models), and applying AI to a real operational problem — auditing freight invoices, a task currently done manually by billing specialists at transportation companies.

## Automation layer (n8n)

To demonstrate production-style orchestration beyond a manual script run, the agent 
is also wrapped in a FastAPI endpoint and triggered through an n8n workflow:

- n8n trigger → HTTP call to the Python agent → conditional routing based on 
  validation status (flagged invoices route to an alert step, approved invoices 
  route to a logging step)
- Workflow exported at `n8n/freight-invoice-workflow.json` — importable into any 
  n8n instance

This separates concerns the way a real deployment would: LangGraph handles the 
AI reasoning, n8n handles triggering, routing, and integration with external 
systems like email or Slack.
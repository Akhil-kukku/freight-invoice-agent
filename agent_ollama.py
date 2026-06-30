import json
from ollama import chat
from typing import TypedDict
from langgraph.graph import StateGraph, END

# =====================================================
# Load the rate card
# =====================================================

with open("data/rate_card.json") as f:
    rate_card = json.load(f)

MODEL = "qwen2.5:3b"  # Change if you downloaded another model


# =====================================================
# Helper function to call Ollama
# =====================================================

def ask_ollama(prompt: str) -> str:
    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    text = response["message"]["content"].strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


# =====================================================
# Shared State
# =====================================================

class InvoiceState(TypedDict):
    raw_text: str
    extracted: dict
    validation_result: dict
    escalation_email: str


# =====================================================
# Agent 1 - Extractor
# =====================================================

def extractor_agent(state: InvoiceState) -> InvoiceState:
    print("Agent 1: Extracting invoice data...")

    prompt = f"""
Extract the following fields from this freight invoice.

Return ONLY valid JSON.

{{
    "carrier_name":"",
    "invoice_number":"",
    "lane":"",
    "weight_lbs":0,
    "charged_amount_usd":0
}}

Rules:
- weight_lbs must be numeric.
- charged_amount_usd must be numeric.
- Remove commas and dollar signs.
- Do not include units like lbs.

Invoice:

{state["raw_text"]}
"""

    response = ask_ollama(prompt)

    try:
        extracted = json.loads(response)
    except json.JSONDecodeError:
        print("Invalid JSON returned by Ollama.")
        print(response)
        extracted = {}

    print(f"Extracted: {extracted}")

    return {
        **state,
        "extracted": extracted
    }


# =====================================================
# Agent 2 - Validator (Python)
# =====================================================

def validator_agent(state: InvoiceState) -> InvoiceState:
    print("Agent 2: Validating against rate card...")

    inv = state["extracted"]

    matching_rate = None
    for rate in rate_card:
        if rate["lane"].lower() == inv.get("lane", "").lower():
            matching_rate = rate
            break

    if not matching_rate:
        result = {
            "status": "no_match",
            "reason": f"No rate card entry found for lane '{inv.get('lane')}'.",
            "overcharge_usd": 0
        }

        print(f"Validation: {result}")

        return {
            **state,
            "validation_result": result
        }

    charged = float(inv.get("charged_amount_usd", 0))
    agreed = float(matching_rate["agreed_rate_usd"])

    weight = float(inv.get("weight_lbs", 0))
    max_weight = float(matching_rate["max_weight_lbs"])

    overcharge = round(charged - agreed, 2)
    overweight = weight > max_weight

    if overcharge <= 0 and not overweight:
        result = {
            "status": "approved",
            "reason": "Invoice matches the agreed rate and is within the weight limit.",
            "overcharge_usd": 0
        }

    elif overweight and overcharge > 0:
        result = {
            "status": "flagged",
            "reason": (
                f"Invoice exceeds the agreed rate by ${overcharge:.2f} "
                f"and exceeds the maximum allowed weight of {int(max_weight)} lbs."
            ),
            "overcharge_usd": overcharge
        }

    elif overweight:
        result = {
            "status": "flagged",
            "reason": (
                f"Invoice exceeds the maximum allowed weight of "
                f"{int(max_weight)} lbs."
            ),
            "overcharge_usd": 0
        }

    else:
        result = {
            "status": "flagged",
            "reason": (
                f"Invoice exceeds the agreed rate by ${overcharge:.2f}."
            ),
            "overcharge_usd": overcharge
        }

    print(f"Validation: {result}")

    return {
        **state,
        "validation_result": result
    }


# =====================================================
# Agent 3 - Escalation
# =====================================================

def escalation_agent(state: InvoiceState) -> InvoiceState:

    result = state["validation_result"]

    if result.get("status") == "approved":
        print("Agent 3: No escalation needed.")
        return {
            **state,
            "escalation_email": ""
        }

    print("Agent 3: Drafting escalation email...")

    inv = state["extracted"]

    prompt = f"""
Write a professional but firm email to {inv.get("carrier_name")} regarding invoice {inv.get("invoice_number")}.

Issue:
{result.get("reason")}

Overcharge Amount:
${result.get("overcharge_usd", 0)}

Keep the email under 100 words.
"""

    email = ask_ollama(prompt)

    print("\nDraft Email:\n")
    print(email)

    return {
        **state,
        "escalation_email": email
    }


# =====================================================
# Build LangGraph Pipeline
# =====================================================

def build_pipeline():

    graph = StateGraph(InvoiceState)

    graph.add_node("extractor", extractor_agent)
    graph.add_node("validator", validator_agent)
    graph.add_node("escalation", escalation_agent)

    graph.set_entry_point("extractor")

    graph.add_edge("extractor", "validator")
    graph.add_edge("validator", "escalation")
    graph.add_edge("escalation", END)

    return graph.compile()


# =====================================================
# Run
# =====================================================

if __name__ == "__main__":

    test_files = [
        "data/sample_invoice.txt",
        "data/sample_invoice_2.txt"
    ]

    pipeline = build_pipeline()

    for filename in test_files:

        print("\\n" + "=" * 50)
        print(f"Processing: {filename}")
        print("=" * 50)

        with open(filename) as f:
            invoice_text = f.read()

        result = pipeline.invoke({
            "raw_text": invoice_text
        })

        print("\\n--- FINAL RESULT ---")
        print(f"Status : {result['validation_result']['status'].upper()}")
        print(f"Reason : {result['validation_result']['reason']}")

        if result["escalation_email"]:
            print("\\nEscalation Email:\\n")
            print(result["escalation_email"])
        else:
            print("\\nNo escalation needed.")
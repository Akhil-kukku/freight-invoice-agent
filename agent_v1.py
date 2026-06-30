import json
import os
from typing import TypedDict

from dotenv import load_dotenv
from google import genai
from langgraph.graph import END, StateGraph

# --------------------------------------------------
# Load Environment
# --------------------------------------------------

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL = "gemini-2.5-flash"

# --------------------------------------------------
# Load Rate Card
# --------------------------------------------------

with open("data/rate_card.json", "r") as f:
    rate_card = json.load(f)

# --------------------------------------------------
# Shared State
# --------------------------------------------------

class InvoiceState(TypedDict, total=False):
    raw_text: str
    extracted: dict
    validation_result: dict
    escalation_email: str

# --------------------------------------------------
# Helper Function
# --------------------------------------------------

def ask_gemini(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return response.text.strip()

# --------------------------------------------------
# Agent 1
# --------------------------------------------------

def extractor_agent(state: InvoiceState):

    print("\nAgent 1 → Extracting invoice...")

    prompt = f"""
Extract these fields from the freight invoice.

Return ONLY valid JSON.

Fields:
- carrier_name
- invoice_number
- lane
- weight_lbs
- charged_amount_usd

Invoice:

{state["raw_text"]}
"""

    response = ask_gemini(prompt)

    try:
        extracted = json.loads(response)
    except json.JSONDecodeError:
        print("Extraction failed.")
        extracted = {}

    print(extracted)

    return {
        **state,
        "extracted": extracted
    }

# --------------------------------------------------
# Agent 2
# --------------------------------------------------

def validator_agent(state: InvoiceState):

    print("\nAgent 2 → Validating invoice...")

    invoice = state["extracted"]

    matching_rate = None

    for rate in rate_card:

        if rate["lane"].lower() == invoice.get("lane", "").lower():

            matching_rate = rate
            break

    prompt = f"""
You are a freight invoice auditor.

Invoice:

Lane:
{invoice.get("lane")}

Weight:
{invoice.get("weight_lbs")}

Charged:
{invoice.get("charged_amount_usd")}

Rate Card:

{json.dumps(matching_rate, indent=2) if matching_rate else "NO MATCH"}

Return ONLY JSON.

{{
"status":"approved | flagged | no_match",
"reason":"...",
"overcharge_usd":0
}}
"""

    response = ask_gemini(prompt)

    try:
        result = json.loads(response)
    except json.JSONDecodeError:

        result = {
            "status": "error",
            "reason": "Model returned invalid JSON.",
            "overcharge_usd": 0
        }

    print(result)

    return {
        **state,
        "validation_result": result
    }

# --------------------------------------------------
# Agent 3
# --------------------------------------------------

def escalation_agent(state: InvoiceState):

    result = state["validation_result"]

    if result["status"] == "approved":

        print("\nAgent 3 → Invoice Approved")

        return {
            **state,
            "escalation_email": ""
        }

    print("\nAgent 3 → Drafting Email")

    invoice = state["extracted"]

    prompt = f"""
Write a professional email.

Carrier:
{invoice.get("carrier_name")}

Invoice:
{invoice.get("invoice_number")}

Issue:
{result.get("reason")}

Overcharge:
${result.get("overcharge_usd")}

Keep under 100 words.
"""

    email = ask_gemini(prompt)

    print(email)

    return {
        **state,
        "escalation_email": email
    }

# --------------------------------------------------
# Build Graph
# --------------------------------------------------

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

# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    pipeline = build_pipeline()

    test_files = [
        "data/sample_invoice.txt",
        "data/sample_invoice_2.txt",
    ]

    for file in test_files:

        print("\n" + "=" * 60)
        print(file)
        print("=" * 60)

        with open(file) as f:
            invoice = f.read()

        result = pipeline.invoke(
            {
                "raw_text": invoice
            }
        )

        print("\nFINAL RESULT")
        print("-" * 40)

        print("Status :", result["validation_result"]["status"])
        print("Reason :", result["validation_result"]["reason"])

        if result["escalation_email"]:
            print("\nEMAIL")
            print(result["escalation_email"])
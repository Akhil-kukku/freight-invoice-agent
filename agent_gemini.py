import os
import json

from google import genai
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash-lite"

# Load the rate card
with open("data/rate_card.json") as f:
    rate_card = json.load(f)


# Helper function to call Gemini
def ask_gemini(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    text = response.text.strip()

    # Remove markdown code fences if Gemini returns them
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


# Define what data flows between agents
class InvoiceState(TypedDict):
    raw_text: str
    extracted: dict
    validation_result: dict
    escalation_email: str


# =====================================================
# AGENT 1: Extractor
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

Invoice:

{state["raw_text"]}
"""

    response = ask_gemini(prompt)

    try:
        extracted = json.loads(response)
    except json.JSONDecodeError:
        print("Invalid JSON returned by Gemini.")
        extracted = {}

    print(f"Extracted: {extracted}")

    return {
        **state,
        "extracted": extracted
    }


# =====================================================
# AGENT 2: Validator
# =====================================================
def validator_agent(state: InvoiceState) -> InvoiceState:
    print("Agent 2: Validating against rate card...")

    inv = state["extracted"]

    matching_rate = None

    for rate in rate_card:
        if rate["lane"].lower() == inv.get("lane", "").lower():
            matching_rate = rate
            break

    prompt = f"""
You are a freight billing auditor.

Invoice:

Lane: {inv.get("lane")}
Weight: {inv.get("weight_lbs")} lbs
Charged: ${inv.get("charged_amount_usd")}

Rate Card:

{json.dumps(matching_rate) if matching_rate else "NO MATCHING LANE FOUND"}

Return ONLY valid JSON.

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
        print("Invalid JSON returned by Gemini.")
        result = {
            "status": "error",
            "reason": "Gemini returned invalid JSON",
            "overcharge_usd": 0
        }

    print(f"Validation: {result}")

    return {
        **state,
        "validation_result": result
    }


# =====================================================
# AGENT 3: Escalation
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
${result.get("overcharge_usd",0)}

Keep the email under 100 words.
"""

    email = ask_gemini(prompt)

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

        print("\n" + "=" * 50)
        print(f"Processing: {filename}")
        print("=" * 50)

        with open(filename) as f:
            invoice_text = f.read()

        result = pipeline.invoke({
            "raw_text": invoice_text
        })

        print("\n--- FINAL RESULT ---")
        print(f"Status : {result['validation_result']['status'].upper()}")
        print(f"Reason : {result['validation_result']['reason']}")

        if result["escalation_email"]:
            print("\nEscalation Email:\n")
            print(result["escalation_email"])
        else:
            print("\nNo escalation needed.")
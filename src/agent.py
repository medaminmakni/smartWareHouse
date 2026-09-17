import os
from typing import Dict, Any, List

import google.generativeai as genai

from src.rag_engine import get_engine


class WarehouseAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.rag = get_engine()

        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('models/gemini-2.5-flash')
        else:
            self.model = None

    # ------------------------------------------------------------- retrieval

    @staticmethod
    def _build_queries(client_name: str, product: str, status: str) -> List[str]:
        """
        Builds the retrieval queries for a gate decision.

        The previous implementation used a single query, "Consignes pour le
        client {client_name}". That only ever matched client_profiles.md, the
        one document that mentions clients by name. The documents that actually
        govern the decision — warehouse_logic.md and gate_pickup_rules.md —
        never name a client, so the highest-value chunks were the least likely
        to be retrieved.

        Three queries are used instead, each aimed at a different part of the
        knowledge base, and their results are merged.
        """
        queries = [
            # 1. Client-specific notes: priority, preferred gate, history.
            f"Client {client_name} priorité porte préférée consignes particulières",
            # 2. Operational policy: gate assignment, verification, safety limits.
            "règles opérationnelles entrepôt attribution de porte "
            "vérification de commande sécurité retrait des marchandises "
            "gate assignment pickup verification safety",
        ]
        # 3. Status-specific handling, only when a status is known.
        if status:
            queries.append(
                f"statut de commande {status} procédure de retrait au portail "
                f"order status {status} pickup confirmation flow"
            )
        # 4. Product handling, when the order names one.
        if product:
            queries.append(f"manutention produit {product} chargement stock")
        return queries

    def _retrieve_rules(self, client_name: str, product: str, status: str,
                        per_query: int = 3, keep: int = 6) -> List[str]:
        queries = self._build_queries(client_name, product, status)
        hits = self.rag.query_multi(queries, n_results=per_query)

        # Keep the best chunks overall, but guarantee at least one from the
        # policy documents so a decision is never made on client notes alone.
        chosen: List[str] = []
        seen = set()

        policy_hits = [h for h in hits if h[1].get("category") == "policies"]
        for doc, meta, _ in policy_hits[:2]:
            if doc not in seen:
                chosen.append(doc)
                seen.add(doc)

        for doc, meta, _ in hits:
            if len(chosen) >= keep:
                break
            if doc not in seen:
                chosen.append(doc)
                seen.add(doc)

        return chosen

    # --------------------------------------------------------------- reason

    def reason(self, vehicle_data: Dict[str, Any]) -> str:
        """
        Calculates a gate decision.

        Facts come from the database and must be exact. Rules come from the RAG
        index and are matched semantically. The model only arbitrates between
        the two, and is constrained to a JSON response.
        """
        from src.database import get_complete_arrival_info

        plate = vehicle_data.get('plate')
        facts = get_complete_arrival_info(plate)

        facts_text = ("No active pickup order found for this plate. "
                      "Action: HOLD for manual verification.")
        client_name = "Unknown"
        product = ""
        status = ""

        if facts:
            client_name = facts['client_nom']
            status = facts['commande_statut']
            product = facts.get('produit_nom', '') or ""
            vehicle = facts.get('plaque_vehicule', plate)

            if status == 'awaiting_pickup':
                facts_text = (f"Vehicle {vehicle} for client {client_name}. "
                              f"Product: {product}. "
                              f"Status: AWAITING_PICKUP - Ready for collection.")
            elif status == 'picked_up':
                facts_text = (f"Vehicle {vehicle} for client {client_name}. "
                              f"Status: ALREADY PICKED UP - Order was already collected.")
            else:
                facts_text = (f"Vehicle {vehicle} for client {client_name}. "
                              f"Status: {status}.")

        context_chunks = self._retrieve_rules(client_name, product, status)
        context_text = "\n---\n".join(context_chunks)

        prompt = f"""
        You are the Warehouse Intelligence Agent for a PICKUP-ONLY warehouse.
        Clients arrive to collect their pre-ordered goods.

        FACTS: {facts_text}
        RULES: {context_text}
        VEHICLE: {plate} at {vehicle_data.get('time')}

        Status meanings:
        - awaiting_pickup: Order is ready, client can proceed to gate for pickup
        - picked_up: Order already collected

        Base the decision on FACTS. Use RULES only for gate choice, priority and
        special handling. If the RULES do not cover the situation, say so in the
        analysis rather than inventing a rule.

        Mandatory: Respond ONLY with a JSON object.
        Template:
        {{
            "analysis": "Explanation of your decision",
            "gate": "D-01 to E-05",
            "priority": "LOW | MEDIUM | HIGH | CRITICAL",
            "action": "PICKUP | HOLD | REJECT"
        }}
        """

        if self.model:
            response = self.model.generate_content(prompt)
            return response.text
        return ('{"analysis": "Error: Gemini API not configured", '
                '"gate": "N/A", "priority": "N/A", "action": "HOLD"}')


if __name__ == "__main__":
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        print("Error: GEMINI_API_KEY not found in environment variables.")
    else:
        agent = WarehouseAgent(API_KEY)
        mock_vehicle = {"plate": "302-502-TUN", "time": "10:15 AM"}
        print("\nAnalyzing warehouse arrival...")
        print(agent.reason(mock_vehicle))

"""
Retrieval evaluation for the SmartWarehouse RAG knowledge base.

Why this exists
---------------
The gate agent's decision quality depends on whether the right policy chunks
reach the prompt. That was never measured, so a retrieval failure looked
exactly like a reasoning failure. This script measures it.

It compares two configurations on the same labelled queries:

  BEFORE  all-MiniLM-L6-v2 (English-only model)
          paragraph chunking (content.split("\\n\\n"))
          one query: "Consignes pour le client {client}"

  AFTER   paraphrase-multilingual-MiniLM-L12-v2
          heading-aware, size-banded chunking
          several queries per decision, results merged

Metric: recall@k — of the documents a decision genuinely needs, how many
appear in the top k retrieved chunks.

This is a small, hand-labelled set (8 scenarios). It is enough to catch the
failure it was built to catch; it is not a benchmark.

Run
---
    pip install sentence-transformers numpy
    python eval_retrieval.py
"""

import glob
import os
import re
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag_engine import _chunk_markdown

DATA_DIR = "data"
K = 5

OLD_MODEL = "all-MiniLM-L6-v2"
NEW_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# --------------------------------------------------------------- scenarios
# Each scenario is a gate decision the agent actually has to make.
# "needs" lists the source documents that carry the rules for that decision.

SCENARIOS = [
    {
        "name": "Known client, order ready for pickup",
        "client": "City Schools",
        "product": "stationery",
        "status": "awaiting_pickup",
        "needs": {"warehouse_logic.md", "client_profiles.md"},
    },
    {
        "name": "Known client, order already collected",
        "client": "RetailCorp",
        "product": "electronics",
        "status": "picked_up",
        "needs": {"gate_pickup_rules.md"},
    },
    {
        "name": "Unknown plate, no matching order",
        "client": "Unknown",
        "product": "",
        "status": "",
        "needs": {"warehouse_logic.md"},
    },
    {
        "name": "Which gate should a pickup vehicle be sent to",
        "client": "RetailCorp",
        "product": "",
        "status": "awaiting_pickup",
        "needs": {"warehouse_logic.md"},
    },
    {
        "name": "How many vehicles may be in the bay at once",
        "client": "City Schools",
        "product": "",
        "status": "awaiting_pickup",
        "needs": {"warehouse_logic.md"},
    },
    {
        "name": "Client needs forklift assistance",
        "client": "RetailCorp",
        "product": "electronics",
        "status": "awaiting_pickup",
        "needs": {"client_profiles.md"},
    },
    {
        "name": "Admin confirmation dialog after arrival",
        "client": "Ahmed",
        "product": "Claviers USB",
        "status": "awaiting",
        "needs": {"gate_pickup_rules.md"},
    },
    {
        "name": "Which language should the assistant reply in",
        "client": "City Schools",
        "product": "",
        "status": "",
        "needs": {"chatbot_rules.md"},
        "chat": True,
    },
]


def old_queries(s: Dict) -> List[str]:
    """The single query the agent used before."""
    return [f"Consignes pour le client {s['client']}"]


def new_queries(s: Dict) -> List[str]:
    """The queries the agent builds now (mirrors WarehouseAgent._build_queries)."""
    if s.get("chat"):
        return [
            "règles du chatbot ton langue gestion des commandes "
            "identification du client comportement général"
        ]
    q = [
        f"Client {s['client']} priorité porte préférée consignes particulières",
        "règles opérationnelles entrepôt attribution de porte "
        "vérification de commande sécurité retrait des marchandises "
        "gate assignment pickup verification safety",
    ]
    if s["status"]:
        q.append(
            f"statut de commande {s['status']} procédure de retrait au portail "
            f"order status {s['status']} pickup confirmation flow"
        )
    if s["product"]:
        q.append(f"manutention produit {s['product']} chargement stock")
    return q


# ------------------------------------------------------------------ chunking

def load_chunks(new_style: bool) -> Tuple[List[str], List[str]]:
    texts, sources = [], []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "**/*.md"), recursive=True)):
        content = open(path, encoding="utf-8").read()
        name = os.path.basename(path)
        chunks = (_chunk_markdown(content) if new_style
                  else [c.strip() for c in content.split("\n\n") if c.strip()])
        texts.extend(chunks)
        sources.extend([name] * len(chunks))
    return texts, sources


# ---------------------------------------------------------------- retrieval

def retrieve(model, texts, sources, queries, k):
    doc_emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    q_emb = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    sims = q_emb @ doc_emb.T                      # (n_queries, n_chunks)
    best = sims.max(axis=0)                       # best score per chunk, any query
    top = np.argsort(-best)[:k]
    return [sources[i] for i in top], [float(best[i]) for i in top]


def run(label, model_name, new_style, query_fn):
    print(f"\n{'='*74}\n{label}\n  model  : {model_name}\n"
          f"  chunks : {'heading-aware, size-banded' if new_style else 'split on blank lines'}")
    model = SentenceTransformer(model_name)
    texts, sources = load_chunks(new_style)
    print(f"  indexed: {len(texts)} chunks\n{'-'*74}")

    hits = 0
    total = 0
    for s in SCENARIOS:
        got, scores = retrieve(model, texts, sources, query_fn(s), K)
        found = s["needs"] & set(got)
        hits += len(found)
        total += len(s["needs"])
        mark = "OK  " if found == s["needs"] else ("part" if found else "MISS")
        print(f"  [{mark}] {s['name'][:46]:46} "
              f"{len(found)}/{len(s['needs'])}  top: {', '.join(dict.fromkeys(got))[:52]}")
    recall = hits / total
    print(f"{'-'*74}\n  recall@{K}: {hits}/{total} = {recall:.0%}")
    return recall


if __name__ == "__main__":
    before = run("BEFORE", OLD_MODEL, new_style=False, query_fn=old_queries)
    after = run("AFTER", NEW_MODEL, new_style=True, query_fn=new_queries)
    print(f"\n{'='*74}")
    print(f"  recall@{K}:  {before:.0%}  ->  {after:.0%}")
    print(f"{'='*74}\n")

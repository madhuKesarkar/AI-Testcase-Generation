"""
test_case_chain.py
==================

The LangChain part of the workflow. Everything AI-related lives here so the
Flask webhook (jira_webhook_langchain.py) stays a thin transport layer.

What this module gives you
--------------------------
1. Typed test cases via `llm.with_structured_output(...)` -> no more regex
   parsing of a Markdown table.
2. An LCEL chain: `prompt | structured_llm`. Call it with `.invoke()`,
   `.batch()` (parallel sections!) or `.stream()`.
3. Semantic duplicate detection with `OpenAIEmbeddings` + cosine similarity,
   so reworded duplicates are caught, not just exact title matches.

LangChain concepts demonstrated (for learning)
----------------------------------------------
- ChatPromptTemplate.from_messages  -> prompt as data, not an f-string
- ChatOpenAI                        -> provider-agnostic chat model wrapper
- .with_structured_output(Model)    -> model output validated into Pydantic
- LCEL `|` composition             -> prompt | model is itself a Runnable
- OpenAIEmbeddings                  -> vectors for semantic comparison
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Provider SDKs are imported lazily inside the factories below, so you only need
# to `pip install` the one you actually use (openai / ollama / google).

# --------------------------------------------------------------------------- #
# 1. Output schema  ->  replaces parse_test_cases_from_markdown()
# --------------------------------------------------------------------------- #


class TestCase(BaseModel):
    """One test case. The LLM is forced to return this exact shape."""

    description: str = Field(description="Short, specific test case title")
    preconditions: str = Field(
        description="User state, test data and environment needed before starting"
    )
    test_steps: list[str] = Field(
        description="Ordered actions, one action per list item, no numbering prefix"
    )
    expected_result: str = Field(
        description="Final observable, verifiable outcome (UI + state + navigation)"
    )


class TestCaseSuite(BaseModel):
    """Wrapper so the model returns a list under a named key."""

    test_cases: list[TestCase] = Field(description="Between 5 and 8 non-redundant test cases")


# --------------------------------------------------------------------------- #
# 2. Prompt  ->  replaces the giant f-string
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are a seasoned QA engineer and expert at writing structured, \
detailed test cases from Jira issues.

Design concise but comprehensive coverage: functional behavior, UI interactions, \
edge cases, user flows and navigation logic - WITHOUT redundant or near-duplicate \
test cases.

Guidelines:
- Produce between 5 and 8 test cases total.
- Cover: core happy paths, UI/field validation, navigation & routing/URL behavior, \
and a few key negative/edge scenarios (invalid data, missing data, auth issues).
- Do NOT create many stylistic variants of the same scenario.
- Do NOT explode the matrix across browsers/devices unless the description implies it.
- test_steps must be plain action strings, one step per item, with NO "1." / "2." prefixes.
- preconditions must mention user state, test data and environment."""

HUMAN_PROMPT = """### Issue Context
- Issue Key: {issue_key}
- Summary: {summary}
- Description:
{description}"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]
)


# --------------------------------------------------------------------------- #
# 3. The chain
# --------------------------------------------------------------------------- #


def _build_llm():
    """
    Return a chat model for LLM_PROVIDER (openai | ollama | google).

    THIS is the LangChain payoff: everything else in this file - the prompt,
    the `PROMPT | structured_llm` chain, `.invoke()`, `.batch()`, the Pydantic
    structured output - is untouched when you switch providers here.
    """
    provider = (os.getenv("LLM_PROVIDER") or "ollama").lower()
    temperature = float(os.getenv("LLM_TEMPERATURE") or "0.4")

    if provider == "openai":  # paid
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            timeout=60,
            max_retries=2,
        )

    if provider == "ollama":  # free, fully local, no API key
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("LLM_MODEL", "llama3.1:8b"),
            temperature=temperature,
        )

    if provider in ("google", "gemini"):  # free tier, no credit card
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL", "gemini-1.5-flash"),
            temperature=temperature,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (use openai | ollama | google)")


@lru_cache(maxsize=1)
def get_chain():
    """prompt | structured_llm  -- built once, reused across requests."""
    structured_llm = _build_llm().with_structured_output(TestCaseSuite)
    return PROMPT | structured_llm


def generate_test_cases(issue_key: str, summary: str, description: str) -> list[TestCase]:
    """Run the chain for one issue/section and return typed TestCase objects."""
    suite: TestCaseSuite = get_chain().invoke(
        {"issue_key": issue_key, "summary": summary, "description": description}
    )
    return suite.test_cases


def generate_test_cases_for_sections(
    issue_key: str, summary: str, sections: list[dict]
) -> list[TestCase]:
    """
    Generate for many description sections in parallel with `.batch()`.

    `sections` is the output of split_test_cases_by_section(): [{title, content}, ...]
    """
    inputs = [
        {
            "issue_key": f"{issue_key} - Section {i}: {s['title']}",
            "summary": summary,
            "description": s["content"],
        }
        for i, s in enumerate(sections, 1)
    ]
    suites: list[TestCaseSuite] = get_chain().batch(inputs, config={"max_concurrency": 4})
    out: list[TestCase] = []
    for suite in suites:
        out.extend(suite.test_cases)
    return out


# --------------------------------------------------------------------------- #
# 4. TestRail payload mapping  (plain Python, kept out of the chain)
# --------------------------------------------------------------------------- #


def to_testrail_payload(tc: TestCase) -> dict:
    """Map a typed TestCase to TestRail's add_case body."""
    return {
        "title": tc.description,
        "type_id": 3,  # "Other" - adjust to your TestRail config
        "custom_preconds": tc.preconditions,
        "custom_steps_separated": [
            {"content": step, "expected": tc.expected_result} for step in tc.test_steps
        ],
    }


# --------------------------------------------------------------------------- #
# 5. Semantic duplicate detection  ->  upgrades exact-title matching
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _embeddings():
    """
    Embeddings for semantic dedupe. Defaults to match LLM_PROVIDER but can be
    overridden with EMBEDDING_PROVIDER. If the model isn't available,
    filter_semantic_duplicates() falls back to exact-title matching.
    """
    provider = (
        os.getenv("EMBEDDING_PROVIDER") or os.getenv("LLM_PROVIDER") or "ollama"
    ).lower()
    model = os.getenv("EMBEDDING_MODEL") or None  # None -> per-provider default below

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model or "text-embedding-3-small")

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model or "nomic-embed-text")

    if provider in ("google", "gemini"):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=model or "models/text-embedding-004")

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def filter_semantic_duplicates(
    candidates: list[TestCase],
    existing_titles: Iterable[str],
    threshold: float | None = None,
) -> tuple[list[TestCase], list[dict]]:
    """
    Split `candidates` into (to_upload, skipped).

    A candidate is a duplicate if its title's cosine similarity to ANY existing
    title, OR to an already-accepted candidate in this batch, is >= threshold.
    Falls back to case-insensitive exact match if embeddings are unavailable.
    """
    if threshold is None:
        threshold = float(os.getenv("DUPLICATE_SIMILARITY_THRESHOLD", "0.86"))

    existing_titles = [t for t in existing_titles if t]
    cand_titles = [c.description for c in candidates]
    if not cand_titles:
        return [], []

    try:
        cand_vecs = np.array(_embeddings().embed_documents(cand_titles))
        existing_vecs = (
            np.array(_embeddings().embed_documents(existing_titles))
            if existing_titles
            else np.empty((0, cand_vecs.shape[1]))
        )
    except Exception as e:  # network / quota / key issues -> safe fallback
        print(f"WARN semantic dedupe unavailable, using exact match: {e}")
        seen = {t.strip().lower() for t in existing_titles}
        keep, skipped = [], []
        for c in candidates:
            key = c.description.strip().lower()
            if key in seen:
                skipped.append({"title": c.description, "reason": "exact-duplicate"})
            else:
                seen.add(key)
                keep.append(c)
        return keep, skipped

    sim_to_existing = (
        _cosine_matrix(cand_vecs, existing_vecs).max(axis=1)
        if len(existing_vecs)
        else np.zeros(len(candidates))
    )

    keep: list[TestCase] = []
    kept_vecs: list[np.ndarray] = []
    skipped: list[dict] = []

    for i, c in enumerate(candidates):
        best = float(sim_to_existing[i])
        source = "existing TestRail case"
        if kept_vecs:
            intra = float(_cosine_matrix(cand_vecs[i : i + 1], np.array(kept_vecs))[0].max())
            if intra > best:
                best, source = intra, "another generated case"
        if best >= threshold:
            skipped.append(
                {"title": c.description, "similarity": round(best, 3), "matched": source}
            )
        else:
            keep.append(c)
            kept_vecs.append(cand_vecs[i])

    return keep, skipped

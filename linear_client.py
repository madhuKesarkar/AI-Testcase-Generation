"""
linear_client.py
================

Upload generated test cases to Linear.

Linear has no "test case" entity, so each TestCase -> one Linear issue:
  - title       = tc.description
  - description = markdown block with preconditions / steps / expected result
  - optional    : grouped under a parent issue that carries the Jira key
  - optional    : tagged with a label (LINEAR_LABEL) for easy filtering

Linear API: a single GraphQL endpoint.
Auth: a Personal API key sent raw in the `Authorization` header.
Create one at:  Linear -> Settings -> Security & access -> Personal API keys
"""

from __future__ import annotations

import os
from functools import lru_cache

import requests

from test_case_chain import TestCase

API_URL = "https://api.linear.app/graphql"


# --------------------------------------------------------------------------- #
# Low-level GraphQL
# --------------------------------------------------------------------------- #
def _headers() -> dict:
    key = os.getenv("LINEAR_API_KEY")
    if not key:
        raise RuntimeError("LINEAR_API_KEY is not set (.env)")
    return {"Authorization": key, "Content-Type": "application/json"}


def _gql(query: str, variables: dict | None = None) -> dict:
    r = requests.post(
        API_URL,
        json={"query": query, "variables": variables or {}},
        headers=_headers(),
        timeout=30,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        raise RuntimeError(f"Linear GraphQL error: {body['errors']}")
    return body["data"]


# --------------------------------------------------------------------------- #
# Lookups (cached for the process)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def resolve_team_id() -> str:
    """LINEAR_TEAM_ID if set, else look it up from LINEAR_TEAM_KEY (e.g. 'QA')."""
    if os.getenv("LINEAR_TEAM_ID"):
        return os.environ["LINEAR_TEAM_ID"]

    want = (os.getenv("LINEAR_TEAM_KEY") or "").lower()
    data = _gql("{ teams(first: 250) { nodes { id key name } } }")
    nodes = data["teams"]["nodes"]
    for t in nodes:
        if want and t["key"].lower() == want:
            return t["id"]
    raise RuntimeError(
        "Could not resolve Linear team. Set LINEAR_TEAM_ID, or LINEAR_TEAM_KEY to one of: "
        + ", ".join(t["key"] for t in nodes)
    )


@lru_cache(maxsize=1)
def resolve_label_id() -> str | None:
    name = os.getenv("LINEAR_LABEL")
    if not name:
        return None
    team_id = resolve_team_id()
    data = _gql(
        "query($id: String!){ team(id:$id){ labels(first:250){ nodes{ id name } } } }",
        {"id": team_id},
    )
    for lbl in data["team"]["labels"]["nodes"]:
        if lbl["name"].lower() == name.lower():
            return lbl["id"]
    print(f"WARN Linear label {name!r} not found on team; creating issues without it")
    return None


def fetch_existing_titles() -> list[str]:
    """Recent issue titles for the team - input to semantic dedupe."""
    team_id = resolve_team_id()
    data = _gql(
        """query($id: ID!) {
             issues(filter: { team: { id: { eq: $id } } },
                    first: 250, orderBy: updatedAt) {
               nodes { title }
             }
           }""",
        {"id": team_id},
    )
    return [n["title"] for n in data["issues"]["nodes"]]


# --------------------------------------------------------------------------- #
# Create
# --------------------------------------------------------------------------- #
_CREATE = """mutation($input: IssueCreateInput!) {
  issueCreate(input: $input) {
    success
    issue { id identifier url title }
  }
}"""


def _description(tc: TestCase, source_key: str) -> str:
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(tc.test_steps, 1))
    return (
        f"**Source issue:** {source_key}\n\n"
        f"**Preconditions**\n{tc.preconditions}\n\n"
        f"**Steps**\n{steps}\n\n"
        f"**Expected result**\n{tc.expected_result}\n"
    )


def _create(input_: dict) -> dict | None:
    res = _gql(_CREATE, {"input": input_})["issueCreate"]
    return res["issue"] if res["success"] else None


def _create_parent(source_key: str, summary: str, label_id: str | None) -> str | None:
    issue = _create(
        {
            "teamId": resolve_team_id(),
            "title": f"[{source_key}] {summary} - AI test cases",
            "description": f"Parent issue grouping AI-generated test cases for {source_key}.",
            **({"labelIds": [label_id]} if label_id else {}),
        }
    )
    return issue["id"] if issue else None


def upload_test_cases(cases: list[TestCase], source_key: str, summary: str) -> list[dict]:
    """Create one Linear issue per test case. Returns the created issue dicts."""
    team_id = resolve_team_id()
    label_id = resolve_label_id()

    parent_id = None
    if (os.getenv("LINEAR_CREATE_PARENT") or "").lower() in ("1", "true", "yes"):
        parent_id = _create_parent(source_key, summary, label_id)
        if parent_id:
            print(f"OK  created parent issue for {source_key}")

    created: list[dict] = []
    for tc in cases:
        issue = _create(
            {
                "teamId": team_id,
                "title": tc.description,
                "description": _description(tc, source_key),
                **({"labelIds": [label_id]} if label_id else {}),
                **({"parentId": parent_id} if parent_id else {}),
            }
        )
        if issue:
            print(f"OK  {issue['identifier']}  {issue['title']}")
            created.append(issue)
        else:
            print(f"FAIL {tc.description}")
    return created

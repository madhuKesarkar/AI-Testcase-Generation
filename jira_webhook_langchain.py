"""
jira_webhook_langchain.py
=========================

LangChain version of jira_webhook.py.

Flow:
  receive Jira webhook
  -> label gate + already-processed gate
  -> split description into sections (plain regex, unchanged)
  -> generate typed test cases via the LangChain chain (test_case_chain.py)
  -> semantic duplicate filtering against existing destination titles
  -> upload to the destination (DESTINATION_SYSTEM = linear | testrail)
  -> track processed issues in uploaded_issues.json

Run:
    pip install -r requirements.txt
    cp .env.example .env      # then fill it in
    python jira_webhook_langchain.py
"""

from __future__ import annotations

import json
import os
import re

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request

from test_case_chain import (
    filter_semantic_duplicates,
    generate_test_cases,
    generate_test_cases_for_sections,
    to_testrail_payload,
)

load_dotenv()

app = Flask(__name__)

UPLOAD_TRACK_FILE = "uploaded_issues.json"
MAX_TEST_CASES_PER_ISSUE = 20
TRIGGER_LABEL = "aitestcase"
DESTINATION = (os.getenv("DESTINATION_SYSTEM") or "linear").lower()


# --------------------------------------------------------------------------- #
# Local processed-issue cache
# --------------------------------------------------------------------------- #
def load_uploaded_issues() -> dict:
    if os.path.exists(UPLOAD_TRACK_FILE):
        try:
            with open(UPLOAD_TRACK_FILE) as f:
                return json.load(f)
        except Exception as e:
            print(f"WARN could not read {UPLOAD_TRACK_FILE}: {e}")
    return {}


def save_uploaded_issues(data: dict) -> None:
    try:
        with open(UPLOAD_TRACK_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        print(f"WARN could not write {UPLOAD_TRACK_FILE}: {e}")


# --------------------------------------------------------------------------- #
# Description section splitting (unchanged from original - not an LLM concern)
# --------------------------------------------------------------------------- #
def split_test_cases_by_section(description: str) -> list[dict]:
    pattern = r"(?:TEST CASE:|h4\.\s\*)([^\n*]+)\*?\n(.*?)(?=\n(?:TEST CASE:|h4\.\s\*)|\Z)"
    matches = re.findall(pattern, description or "", re.DOTALL)
    return [{"title": t.strip(), "content": c.strip()} for t, c in matches]


# --------------------------------------------------------------------------- #
# Destination: Linear
# --------------------------------------------------------------------------- #
def linear_existing_titles() -> list[str]:
    from linear_client import fetch_existing_titles

    return fetch_existing_titles()


def linear_upload(cases, issue_key, summary) -> int:
    from linear_client import upload_test_cases

    return len(upload_test_cases(cases, issue_key, summary))


# --------------------------------------------------------------------------- #
# Destination: TestRail (kept for reference; needs TESTRAIL_* env vars)
# --------------------------------------------------------------------------- #
def _tr_env():
    return (
        os.getenv("TESTRAIL_BASE_URL"),
        os.getenv("TESTRAIL_USERNAME"),
        os.getenv("TESTRAIL_API_KEY"),
        int(os.getenv("TESTRAIL_PROJECT_ID", "0")),
    )


def testrail_existing_titles(section_id: int) -> list[str]:
    base_url, username, api_key, project_id = _tr_env()
    url = f"{base_url}/index.php?/api/v2/get_cases/{project_id}&section_id={section_id}"
    try:
        r = requests.get(url, auth=(username, api_key), timeout=30)
        r.raise_for_status()
        payload = r.json()
        cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
        return [(c.get("title") or "").strip() for c in cases]
    except Exception as e:
        print(f"WARN could not fetch existing TestRail cases: {e}")
        return []


def testrail_add_case(section_id: int, body: dict) -> bool:
    base_url, username, api_key, _ = _tr_env()
    url = f"{base_url}/index.php?/api/v2/add_case/{section_id}"
    try:
        r = requests.post(
            url,
            json=body,
            auth=(username, api_key),
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if r.status_code == 200:
            print(f"OK uploaded: {body['title']}")
            return True
        print(f"FAIL {body['title']} -> {r.status_code} {r.text}")
        return False
    except requests.RequestException as e:
        print(f"FAIL request error for {body['title']}: {e}")
        return False


# --------------------------------------------------------------------------- #
# Webhook
# --------------------------------------------------------------------------- #
@app.route("/jira-webhook", methods=["POST"])
def jira_webhook():
    try:
        data = request.json or {}
        if "issue" not in data:
            return jsonify({"status": "error", "message": "Invalid request payload"}), 400

        fields = data["issue"].get("fields", {})
        issue_key = data["issue"].get("key", "N/A")
        summary = fields.get("summary", "N/A")
        description = fields.get("description", "N/A") or ""
        labels = [str(l).lower() for l in (fields.get("labels") or [])]

        if TRIGGER_LABEL not in labels:
            return jsonify({"status": "skipped", "reason": f"{TRIGGER_LABEL} label not present"}), 200

        uploaded = load_uploaded_issues()
        if issue_key in uploaded:
            return jsonify({"status": "skipped", "message": f"{issue_key} already processed"}), 200

        # --- generate via LangChain ---
        sections = split_test_cases_by_section(description)
        if sections:
            print(f"Generating for {len(sections)} section(s) of {issue_key}")
            cases = generate_test_cases_for_sections(issue_key, summary, sections)
        else:
            print(f"Generating from full description of {issue_key}")
            cases = generate_test_cases(issue_key, summary, description)

        if not cases:
            return jsonify({"status": "error", "message": "No test cases generated"}), 500

        if len(cases) > MAX_TEST_CASES_PER_ISSUE:
            print(f"Trimming {len(cases)} -> {MAX_TEST_CASES_PER_ISSUE}")
            cases = cases[:MAX_TEST_CASES_PER_ISSUE]

        # --- dedupe + upload to the configured destination ---
        if DESTINATION == "linear":
            existing = linear_existing_titles()
            to_upload, skipped = filter_semantic_duplicates(cases, existing)
            for s in skipped:
                print(f"SKIP duplicate: {s}")
            uploaded_count = linear_upload(to_upload, issue_key, summary)

        elif DESTINATION == "testrail":
            section_id = int(os.getenv("TESTRAIL_DEFAULT_SECTION_ID"))
            existing = testrail_existing_titles(section_id)
            to_upload, skipped = filter_semantic_duplicates(cases, existing)
            for s in skipped:
                print(f"SKIP duplicate: {s}")
            uploaded_count = sum(
                testrail_add_case(section_id, to_testrail_payload(tc)) for tc in to_upload
            )

        else:
            return jsonify(
                {"status": "error", "message": f"Unknown DESTINATION_SYSTEM: {DESTINATION}"}
            ), 500

        uploaded[issue_key] = {
            "destination": DESTINATION,
            "generated": len(cases),
            "uploaded": uploaded_count,
            "skipped_duplicates": len(skipped),
        }
        save_uploaded_issues(uploaded)

        return jsonify(
            {
                "status": "success",
                "issue": issue_key,
                "destination": DESTINATION,
                "generated": len(cases),
                "uploaded": uploaded_count,
                "skipped_duplicates": skipped,
            }
        ), 200

    except Exception as e:
        print(f"Webhook error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "destination": DESTINATION}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

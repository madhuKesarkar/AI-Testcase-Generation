"""
try_linear.py - generate test cases with the LangChain chain and push them
to Linear, without Jira or Flask.

    pip install -r requirements.txt
    # in .env: LLM_PROVIDER=ollama (or google), plus:
    #   LINEAR_API_KEY=lin_api_...
    #   LINEAR_TEAM_KEY=QA           (or LINEAR_TEAM_ID=...)
    #   LINEAR_LABEL=ai-testcase     (optional)
    #   LINEAR_CREATE_PARENT=true    (optional)
    python try_linear.py
"""

from dotenv import load_dotenv

from linear_client import fetch_existing_titles, upload_test_cases
from test_case_chain import filter_semantic_duplicates, generate_test_cases

load_dotenv()

ISSUE_KEY = "DEMO-1"
SUMMARY = "User can log in with email and password"
DESCRIPTION = """
As a registered user I want to log in using my email and password so that I can
access my dashboard.

Acceptance Criteria:
- Valid credentials redirect to /dashboard.
- Invalid password shows an inline error and stays on /login.
- Empty email or password disables the Login button.
- After 5 failed attempts the account is locked for 15 minutes.
"""


def main() -> None:
    cases = generate_test_cases(ISSUE_KEY, SUMMARY, DESCRIPTION)
    print(f"\n{len(cases)} test cases generated\n")

    existing = fetch_existing_titles()
    print(f"{len(existing)} existing Linear issue titles fetched for dedupe")

    to_upload, skipped = filter_semantic_duplicates(cases, existing)
    for s in skipped:
        print(f"  skip duplicate: {s}")

    print(f"\nCreating {len(to_upload)} Linear issue(s)...\n")
    created = upload_test_cases(to_upload, ISSUE_KEY, SUMMARY)
    print(f"\nDone. {len(created)} issue(s) created:")
    for issue in created:
        print(f"  {issue['identifier']}  {issue['url']}")


if __name__ == "__main__":
    main()

"""
try_chain.py - run the LangChain pipeline locally without Jira or TestRail.

    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...      # or put it in .env
    python try_chain.py

Good for learning: tweak the prompt / model / threshold in test_case_chain.py
and immediately see the effect here.
"""

from dotenv import load_dotenv

from test_case_chain import (
    filter_semantic_duplicates,
    generate_test_cases,
    to_testrail_payload,
)

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

# Pretend these already exist in TestRail - note the reworded near-duplicate.
EXISTING_TITLES = [
    "Successful login with valid email and password",
    "Login button stays disabled when form is incomplete",
]


def main() -> None:
    cases = generate_test_cases(ISSUE_KEY, SUMMARY, DESCRIPTION)
    print(f"\n=== {len(cases)} test cases generated (typed TestCase objects) ===\n")
    for i, tc in enumerate(cases, 1):
        print(f"{i}. {tc.description}")
        print(f"   preconditions: {tc.preconditions}")
        for step in tc.test_steps:
            print(f"     - {step}")
        print(f"   expected: {tc.expected_result}\n")

    keep, skipped = filter_semantic_duplicates(cases, EXISTING_TITLES)
    print(f"=== dedupe: {len(keep)} to upload, {len(skipped)} skipped ===")
    for s in skipped:
        print(f"   skip: {s}")

    print("\n=== TestRail payload for the first kept case ===")
    if keep:
        import json

        print(json.dumps(to_testrail_payload(keep[0]), indent=2))


if __name__ == "__main__":
    main()

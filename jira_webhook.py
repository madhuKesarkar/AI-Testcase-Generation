

import openai
from flask import Flask, request, jsonify
import os
import re
import requests
import json
from dotenv import load_dotenv


load_dotenv()


# Initialize Flask app
app = Flask(__name__)


# Constants
UPLOAD_TRACK_FILE = "uploaded_issues.json"
MAX_TEST_CASES_PER_ISSUE = 20  # hard cap to avoid explosion




def load_uploaded_issues():
   if os.path.exists(UPLOAD_TRACK_FILE):
       try:
           with open(UPLOAD_TRACK_FILE, "r") as f:
               return json.load(f)
       except Exception as e:
           print(f"⚠️ Could not read {UPLOAD_TRACK_FILE}: {e}")
           return {}
   return {}




def save_uploaded_issues(data):
   try:
       with open(UPLOAD_TRACK_FILE, "w") as f:
           json.dump(data, f, indent=2)
   except OSError as e:
       # If disk is full or write fails, log but don't crash webhook
       print(f"⚠️ Could not write {UPLOAD_TRACK_FILE}: {e}")




# Ensure OpenAI API key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
   raise ValueError("Missing OpenAI API Key. Set OPENAI_API_KEY in the environment.")


client = openai.OpenAI(api_key=api_key)




# Function to generate structured test cases using GPT
def generate_test_cases(issue_key, summary, description):


   prompt = f"""
You are a **seasoned QA engineer** and expert in generating structured and detailed test cases based on Jira issues.
Your goal is to design **concise but comprehensive test coverage**, accounting for **functional behavior, UI interactions, edge cases,
user flows, and navigation logic**, *without* creating redundant or overly similar test cases.


---


### Issue Context
- Issue Key: {issue_key}
- Summary: {summary}
- Description: {description}


---


### Test Case Output Format (IMPORTANT)
Return a single Markdown table with **exactly these 5 columns**:


| Test Case No. | Test Case Description | Preconditions | Test Steps | Expected Result |
|--------------|-----------------------|---------------|------------|-----------------|
| 1 | [Short title] | [Detailed preconditions] | [Multi-step actions] | [Verifiable outcome] |
| 2 | ... | ... | ... | ... |


**Preconditions**:
- Describe **user state**, **test data**, and **environment**.
- Example: `User is registered with an active account; browser is on the Login page; test user credentials are available.`


**Test Steps**:
- Use a **numbered list in plain text**, *inside the cell*, one per line, or as an inline list:
 - `1. Open the application URL`
 - `2. Enter a valid registered email`
 - `3. Enter the correct password`
 - `4. Click the "Login" button`
- Do **not** use bullet points (`-` or `*`) inside the cell.
- Do **not** wrap steps in additional Markdown, just plain text with numbers.


**Expected Result**:
- Describe the final observable outcome (UI + state + navigation), clear and testable.


---


### Test Case Design Guidelines


- Generate **between 5 and 8 test cases total** for this issue.
- Cover:
 - Core happy paths
 - UI validations and field behavior
 - Navigation and routing / URL behavior
 - A small number of key edge cases & negative scenarios (invalid data, missing data, auth issues)
- **Avoid duplicates or very similar variants.** Do *not* create many stylistic variations of the same scenario.
- Do **not** explode the matrix across browsers/devices unless explicitly implied by the description.


Only return the Markdown table, nothing else.
"""


   try:
       response = client.chat.completions.create(
           model="gpt-4",
           messages=[{"role": "user", "content": prompt}],
           temperature=0.4,
       )
       return response.choices[0].message.content
   except Exception as e:
       print(f"OpenAI API Error: {e}")
       return None




# Function to parse test cases from GPT Markdown table and format for TestRail
def parse_test_cases_from_markdown(markdown_table, default_precondition=""):
   rows = markdown_table.strip().split("\n")
   test_cases = []


   for row in rows:
       row = row.strip()
       if not row.startswith("|"):
           continue


       # Skip header & separator rows
       if "Test Case No." in row or "---" in row:
           continue


       # Split columns
       parts = [col.strip() for col in row.strip("|").split("|")]


       # Expecting 5 columns: No, Description, Preconditions, Test Steps, Expected Result
       if len(parts) < 5:
           print(f"⚠️ Skipped row due to unexpected column count: {row}")
           continue


       _, tc_description, tc_preconds, tc_steps_raw, tc_expected = parts


       # Basic validation (description + expected cannot be empty)
       if not tc_description or not tc_expected:
           print(f"⚠️ Skipped row due to empty description/expected: {row}")
           continue


       # Use generated preconditions if present, else fall back to provided default
       preconds = tc_preconds if tc_preconds else default_precondition


       # ---------- STEP SPLITTING LOGIC ----------
       step_lines = []


       # 1) First, try splitting by real line breaks
       for line in tc_steps_raw.splitlines():
           clean = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
           if clean:
               step_lines.append(clean)


       # 2) If we still have only one blob, split by inline "1. 2. 3." patterns
       if len(step_lines) <= 1:
           # Example:
           # "1. Open URL 2. Enter email 3. Enter password 4. Click Login"
           chunks = re.split(r"\s*\d+[\.\)]\s*", tc_steps_raw)
           chunks = [c.strip(" .") for c in chunks if c.strip()]
           if len(chunks) > 1:
               step_lines = chunks


       # 3) Fallback: if still nothing, treat the whole cell as one step
       if not step_lines:
           step_lines = [tc_steps_raw.strip()]


       # Build TestRail steps: each step gets the same overall expected result
       custom_steps = [
           {
               "content": step,
               "expected": tc_expected,
           }
           for step in step_lines
       ]
       # ------------------------------------------


       test_case = {
           "title": tc_description,
           "type_id": 3,  # "Other" (adjust if you use different TestRail types)
           "custom_preconds": preconds,
           "custom_steps_separated": custom_steps,
       }


       test_cases.append(test_case)


   return test_cases


#  Helper to choose TestRail Section based on Jira labels
def choose_testrail_section_name(jira_labels, summary, description):
   labels = set([l.lower() for l in (jira_labels or [])])


   # Routing labels (you can change these names anytime)
   if "testrail_school" in labels:
       return "School Functional Test"
   if "testrail_edge" in labels:
       return "Edge Cases"


   # Default route
   return "Org Functional Test"


def tr_get_sections(project_id: int, suite_id: int):
   base_url = os.getenv("TESTRAIL_BASE_URL")
   username = os.getenv("TESTRAIL_USERNAME")
   api_key = os.getenv("TESTRAIL_API_KEY")


   url = f"{base_url}/index.php?/api/v2/get_sections/{project_id}&suite_id={suite_id}"
   r = requests.get(url, auth=(username, api_key))
   r.raise_for_status()
   return r.json()


def get_section_id_by_name(project_id: int, suite_id: int, section_name: str) -> int:
   sections = tr_get_sections(project_id, suite_id)
   for s in sections:
       if s.get("name", "").strip().lower() == section_name.strip().lower():
           return int(s["id"])


   raise RuntimeError(
       f"TestRail section '{section_name}' not found in project_id={project_id}, suite_id={suite_id}. "
       f"Create it in TestRail (or add auto-create logic)."
   )


# Upload to TestRail API
# def upload_test_cases_to_testrail(test_cases, section_id):
#     base_url = os.getenv("TESTRAIL_BASE_URL")
#     username = os.getenv("TESTRAIL_USERNAME")
#     api_key = os.getenv("TESTRAIL_API_KEY")
#     project_id = int(os.getenv("TESTRAIL_PROJECT_ID"))


#     headers = {"Content-Type": "application/json"}


#     # Step 1: Get existing test cases from the section (correct API format)
#     get_url = f"{base_url}/index.php?/api/v2/get_cases/{project_id}&section_id={section_id}"
#     existing_titles = set()


#     # try:
#     #     response = requests.get(get_url, auth=(username, api_key), headers=headers)
#     #     if response.status_code == 200:
#     #         existing_cases = response.json()
#     #         for case in existing_cases:
#     #             existing_titles.add(case["title"].strip().lower())
#     #         print(f"📋 Existing titles found: {len(existing_titles)}")
#     #     else:
#     #         print("⚠️ Failed to fetch existing cases. Proceeding without duplicate check.")
#     # except Exception as e:
#     #     print(f"⚠️ Error fetching existing cases: {e}")


#     # # Step 2: Upload new test cases if not duplicate
#     # for test_case in test_cases:
#     #     title_clean = test_case["title"].strip().lower()
#     #     if title_clean in existing_titles:
#     #         print(f"⏩ Skipped duplicate (by title match): {test_case['title']}")
#     #         continue


#     #     url = f"{base_url}/index.php?/api/v2/add_case/{section_id}"


#     #     try:
#     #         print("🔧 Uploading test case:")
#     #         print(json.dumps(test_case, indent=2))


#     #         response = requests.post(
#     #             url, json=test_case, auth=(username, api_key), headers=headers
#     #         )


#     #         if response.status_code == 200:
#     #             print(f"✅ Uploaded: {test_case['title']}")
#     #         else:
#     #             print(f"❌ Failed to upload: {test_case['title']}")
#     #             print("🔴 Status Code:", response.status_code)
#     #             print("🔴 Response Body:", response.text)


#     #     except requests.exceptions.RequestException as e:
#     #         print(f"❌ Request failed: {e}")
#     #         print("Payload that caused error:", json.dumps(test_case, indent=2))


# try:
#         response = requests.get(get_url, auth=(username, api_key), headers=headers)
#         if response.status_code == 200:
#             existing_cases = response.json()
#             for case in existing_cases:
#                 title = (case.get("title") or "").strip().lower()
#                 if title:
#                     existing_titles.add(title)
#             print(f"📋 Existing titles found in section {section_id}: {len(existing_titles)}")
#         else:
#             print("⚠️ Failed to fetch existing cases. Proceeding without duplicate check.")
#             print("Status:", response.status_code, "Body:", response.text)
#     except Exception as e:
#         print(f"⚠️ Error fetching existing cases: {e}")


#     # Step 2: Upload new test cases if not duplicate
#     for test_case in test_cases:
#         title_clean = test_case["title"].strip().lower()
#         if title_clean in existing_titles:
#             print(f"⏩ Skipped duplicate (by title match): {test_case['title']}")
#             continue


#         url = f"{base_url}/index.php?/api/v2/add_case/{section_id}"


#         try:
#             print("🔧 Uploading test case:")
#             print(json.dumps(test_case, indent=2))


#             response = requests.post(
#                 url, json=test_case, auth=(username, api_key), headers=headers
#             )


#             if response.status_code == 200:
#                 print(f"✅ Uploaded: {test_case['title']}")
#             else:
#                 print(f"❌ Failed to upload: {test_case['title']}")
#                 print("🔴 Status Code:", response.status_code)
#                 print("🔴 Response Body:", response.text)


#         except requests.exceptions.RequestException as e:
#             print(f"❌ Request failed: {e}")
#             print("Payload that caused error:", json.dumps(test_case, indent=2))


def upload_test_cases_to_testrail(test_cases, section_id):
   base_url = os.getenv("TESTRAIL_BASE_URL")
   username = os.getenv("TESTRAIL_USERNAME")
   api_key = os.getenv("TESTRAIL_API_KEY")
   project_id = int(os.getenv("TESTRAIL_PROJECT_ID"))


   headers = {"Content-Type": "application/json"}


   # Step 1: Get existing test cases from the section (correct API format)
   # get_url = f"{base_url}/index.php?/api/v2/get_cases/{project_id}&section_id={section_id}"


   project_id = int(os.getenv("TESTRAIL_PROJECT_ID"))
   get_url = f"{base_url}/index.php?/api/v2/get_cases/{project_id}&section_id={section_id}"


   existing_titles = set()


   try:
       response = requests.get(get_url, auth=(username, api_key), headers=headers)
       if response.status_code == 200:
           existing_cases = response.json()
           for case in existing_cases:
               title = (case.get("title") or "").strip().lower()
               if title:
                   existing_titles.add(title)
           print(f"📋 Existing titles found in section {section_id}: {len(existing_titles)}")
       else:
           print("⚠️ Failed to fetch existing cases. Proceeding without duplicate check.")
           print("Status:", response.status_code, "Body:", response.text)
   except Exception as e:
       print(f"⚠️ Error fetching existing cases: {e}")


   # Step 2: Upload new test cases if not duplicate
   for test_case in test_cases:
       title_clean = test_case["title"].strip().lower()
       if title_clean in existing_titles:
           print(f"⏩ Skipped duplicate (by title match): {test_case['title']}")
           continue


       url = f"{base_url}/index.php?/api/v2/add_case/{section_id}"


       try:
           print("🔧 Uploading test case:")
           print(json.dumps(test_case, indent=2))


           response = requests.post(
               url, json=test_case, auth=(username, api_key), headers=headers
           )


           if response.status_code == 200:
               print(f"✅ Uploaded: {test_case['title']}")
           else:
               print(f"❌ Failed to upload: {test_case['title']}")
               print("🔴 Status Code:", response.status_code)
               print("🔴 Response Body:", response.text)


       except requests.exceptions.RequestException as e:
           print(f"❌ Request failed: {e}")
           print("Payload that caused error:", json.dumps(test_case, indent=2))


# Helper to split sections from description
def split_test_cases_by_section(description):
   pattern = (
       r"(?:TEST CASE:|h4\.\s\*)([^\n*]+)\*?\n(.*?)(?=\n(?:TEST CASE:|h4\.\s\*)|\Z)"
   )
   matches = re.findall(pattern, description, re.DOTALL)


   sections = []
   for title, content in matches:
       sections.append({"title": title.strip(), "content": content.strip()})
   return sections




# def jira_webhook():
#     try:
#         data = request.json
#         if not data or "issue" not in data:
#             return (
#                 jsonify({"status": "error", "message": "Invalid request payload"}),
#                 400,
#             )


#         issue_key = data.get("issue", {}).get("key", "N/A")
#         issue_summary = data.get("issue", {}).get("fields", {}).get("summary", "N/A")
#         issue_description = (
#             data.get("issue", {}).get("fields", {}).get("description", "N/A")
#         )
      


#         print(f"Issue Key: {issue_key}")
#         print(f"Summary: {issue_summary}")
#         print(f"Description: {issue_description}")


@app.route("/jira-webhook", methods=["POST"])
def jira_webhook():
   try:
       data = request.json
       if not data or "issue" not in data:
           return jsonify({"status": "error", "message": "Invalid request payload"}), 400


       issue_key = data.get("issue", {}).get("key", "N/A")
       issue_summary = data.get("issue", {}).get("fields", {}).get("summary", "N/A")
       issue_description = data.get("issue", {}).get("fields", {}).get("description", "N/A")


       jira_labels = data.get("issue", {}).get("fields", {}).get("labels", []) or []


       # ✅ Trigger gate: only proceed if AITestCase label is present
       labels_lower = [str(l).lower() for l in jira_labels]
       if "aitestcase" not in labels_lower:
           print("⏩ Skipping issue: AITestCase label not present")
           return jsonify({"status": "skipped", "reason": "AITestCase label not present"}), 200


       print(f"Issue Key: {issue_key}")
       print(f"Summary: {issue_summary}")
       print(f"Description: {issue_description}")
       print(f"Labels: {jira_labels}")


       # Load uploaded issues from local cache
       uploaded_issues = load_uploaded_issues()


       # ❗ Always skip if we've already generated for this issue
       if issue_key in uploaded_issues:
           print(f"⏩ Skipping upload: Test cases for {issue_key} already uploaded.")
           return (
               jsonify(
                   {
                       "status": "skipped",
                       "message": f"Already uploaded test cases for {issue_key}",
                   }
               ),
               200,
           )


       test_case_sections = split_test_cases_by_section(issue_description)


       parsed = []
       all_test_cases_raw = ""


       if test_case_sections:
           for idx, section in enumerate(test_case_sections, 1):
               section_title = section["title"]
               section_content = section["content"]


               print(f"Generating test cases for section: {section_title}")
               test_cases_md = generate_test_cases(
                   issue_key=f"{issue_key} - Section {idx}: {section_title}",
                   summary=issue_summary,
                   description=section_content,
               )


               if test_cases_md:
                   section_parsed = parse_test_cases_from_markdown(
                       test_cases_md, default_precondition=section_title
                   )
                   parsed.extend(section_parsed)
                   all_test_cases_raw += (
                       f"\n### Test Cases for {section_title}\n{test_cases_md}\n"
                   )
       else:
           print("No TEST CASE or h4 sections found. Generating from entire description.")
           test_cases_md = generate_test_cases(
               issue_key, issue_summary, issue_description
           )
           if test_cases_md:
               parsed = parse_test_cases_from_markdown(
                   test_cases_md, default_precondition=issue_summary
               )
               all_test_cases_raw = test_cases_md


       # Hard cap per issue to avoid explosion
       if parsed:
           if len(parsed) > MAX_TEST_CASES_PER_ISSUE:
               print(f"🔻 Trimming test cases from {len(parsed)} to {MAX_TEST_CASES_PER_ISSUE}")
               parsed = parsed[:MAX_TEST_CASES_PER_ISSUE]


           section_id = int(os.getenv("TESTRAIL_DEFAULT_SECTION_ID"))
           upload_test_cases_to_testrail(parsed, section_id=section_id)


           # Save successful upload record (store only summary info to keep file small)
           uploaded_issues[issue_key] = {
               "count": len(parsed)
           }
           save_uploaded_issues(uploaded_issues)


           return jsonify({"status": "success", "count": len(parsed)}), 200
       else:
           return (
               jsonify({"status": "error", "message": "Failed to generate test cases"}),
               500,
           )


   except Exception as e:
       print(f"Webhook Handling Error: {str(e)}")
       return jsonify({"status": "error", "message": str(e)}), 500




if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5001, debug=True)


  

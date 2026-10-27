# Databricks App end-to-end test: Offer Management Console

Run 2026-10-27T20:37:21+00:00 against `https://offer-prioritization-console-7474656617733963.aws.databricksapps.com` with `scripts/app_e2e_test.py`.
The script sends the same Dash callback requests the browser sends, authenticated with a workspace
OAuth token, then reads the written feedback back from Unity Catalog.

## 1. App is up

`GET /` -> HTTP 200 in 142 ms

## 2. Member list loaded from Lakebase

`GET /_dash-layout` -> HTTP 200 in 119 ms; member dropdown has **100** members (the Lakebase work queue). Header: `100 members • Lakebase`.

## 3. Member lookup (Lakebase read through the app)

Callback `member-dropdown.value = MBR_014716CC` -> HTTP 200 in 286 ms.

Offers rendered: CHRON_003, COST_001, MH_001, MH_002, PREV_001 (top offer `COST_001`).

Rendered page text (first 40 text nodes):

```text
Member Profile
MBR_014716CC
Age
35
Risk Score
45
Conditions
0
Tenure
30 mo
Claims
2
Engagements
14
Health Indicators:
✗ Diabetes
✗ Cardiovascular
✗ Respiratory
✗ Mental Health
✗ Complex Care
Recommended Offers
5 offers ranked by relevance
1
HSA/FSA Optimization
ID: COST_001
RELEVANCE
80
AI RECOMMENDATION INSIGHT
This member is still paying a large share of costs toward their deductible, so learning how to better use an HSA or FSA can help them keep more money in their pocket. Their medical spending so far shows they do use care, but not constantly, which is a good time to plan ahead for expected and unexpected expenses. The program can walk them through which expenses are eligible, how much to set aside, and how to time contributions so they get the most tax benefit. This can make future doctor visits, prescriptions, or procedures more affordable and predictable.
View Decision Factors
Key Decision Factors
The following member attributes influenced this recommendation:
💰
Remaining deductible
Supports this offer
This factor positively supports this recommendation.
📊
Medical services usage
Supports this offer
This factor positively supports this recommendation.
```

First AI recommendation insight shown to the care manager:

> This member is still paying a large share of costs toward their deductible, so learning how to better use an HSA or FSA can help them keep more money in their pocket. Their medical spending so far shows they do use care, but not constantly, which is a good time to plan ahead for expected and unexpected expenses. The program can walk them through which expenses are eligible, how much to set aside, and how to time contributions so they get the most tax benefit. This can make future doctor visits, prescriptions, or procedures more affordable and predictable.

## 4. Approve the top offer (write to Delta)

Callback `approve-btn.n_clicks` for `MBR_014716CC` / `COST_001` -> HTTP 200 in 3969 ms; app replied **✓ Approved (saved)**.

## 5. Submit a care-manager comment (parameterized SQL)

The comment deliberately contains quotes and SQL-like text: `Member asked for a call back Tuesday; O'Neil said: don't email -- ok; DROP TABLE x;`

Callback `submit-text-btn.n_clicks` -> HTTP 200 in 1578 ms; app replied **✓ Comment submitted!**.

## 6. Feedback read back from Unity Catalog

`SELECT ... FROM bhavik_demos.offer_prioritization.offer_feedback WHERE member_id = 'MBR_014716CC'`

| member_id | offer_id | feedback | feedback_text | feedback_time |
| --- | --- | --- | --- | --- |
| MBR_014716CC | COST_001 | approved |  | 2026-10-27T20:37:24.078Z |
| MBR_014716CC | COST_001 | comment | Member asked for a call back Tuesday; O'Neil said: don't email -- ok; DROP TABLE x; | 2026-10-27T20:37:26.629Z |

**Result: PASS**: the approval and the comment (stored verbatim, quotes and SQL text intact) are in the governed Delta table the retraining notebook reads.

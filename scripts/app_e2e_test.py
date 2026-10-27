"""End-to-end test of the deployed Offer Management Console (Databricks App).

Drives the app's real Dash callbacks over HTTPS with a workspace OAuth token, exactly the
requests the browser sends: load the page, look up a member (reads Lakebase), approve the
top offer and submit a care-manager comment (writes Delta via the SQL Statement API), then
reads the feedback back from Unity Catalog. Writes evidence/app_e2e_test.md.

Usage:
  python -I scripts/app_e2e_test.py --profile fevm-classic-stable
"""
import argparse
import json
import re
import subprocess
import time
import urllib.request
from datetime import datetime, timezone

APP_URL = "https://offer-prioritization-console-7474656617733963.aws.databricksapps.com"
FEEDBACK_TABLE = "bhavik_demos.offer_prioritization.offer_feedback"


def token(profile):
    out = subprocess.run(["databricks", "auth", "token", "--profile", profile, "-o", "json"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)["access_token"]


def sql(profile, statement):
    out = subprocess.run(["databricks", "experimental", "aitools", "tools", "query", statement,
                          "--profile", profile], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def http(tok, path, body=None):
    req = urllib.request.Request(APP_URL + path, method="POST" if body else "GET",
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read().decode()
        return r.status, (time.perf_counter() - t0) * 1000, raw


def pid(d):
    """Dash stringifies pattern-matching ids with sorted keys and no spaces."""
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def texts(node, out):
    """Collect all string leaves from a Dash component tree."""
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, list):
        for n in node:
            texts(n, out)
    elif isinstance(node, dict):
        for k, v in node.items():
            if k in ("children", "props", "value", "label") or isinstance(v, (dict, list)):
                texts(v, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", default="evidence/app_e2e_test.md")
    a = ap.parse_args()
    tok = token(a.profile)
    started = datetime.now(timezone.utc)
    md = ["# Databricks App end-to-end test: Offer Management Console", "",
          f"Run {started.isoformat(timespec='seconds')} against `{APP_URL}` with `scripts/app_e2e_test.py`.",
          "The script sends the same Dash callback requests the browser sends, authenticated with a workspace",
          "OAuth token, then reads the written feedback back from Unity Catalog.", ""]

    status, ms, _ = http(tok, "/")
    md += ["## 1. App is up", "", f"`GET /` -> HTTP {status} in {ms:.0f} ms", ""]

    status, ms, raw = http(tok, "/_dash-layout")
    layout = json.loads(raw)
    dropdown = next(n for n in _walk(layout) if isinstance(n, dict) and n.get("props", {}).get("id") == "member-dropdown")
    options = dropdown["props"]["options"]
    member = options[0]["value"]
    banner = [t for t in texts(layout, []) if "members" in t.lower() and "lakebase" in t.lower()]
    md += ["## 2. Member list loaded from Lakebase", "",
           f"`GET /_dash-layout` -> HTTP {status} in {ms:.0f} ms; member dropdown has **{len(options)}** members "
           f"(the Lakebase work queue). Header: `{banner[0] if banner else 'n/a'}`.", ""]

    body = {"output": "recommendations-container.children",
            "outputs": {"id": "recommendations-container", "property": "children"},
            "inputs": [{"id": "member-dropdown", "property": "value", "value": member}],
            "changedPropIds": ["member-dropdown.value"], "state": []}
    status, ms, raw = http(tok, "/_dash-update-component", body)
    tree = json.loads(raw)["response"]["recommendations-container"]["children"]
    flat = [t.strip() for t in texts(tree, []) if t and t.strip()]
    ids = sorted({m.group(1) for m in re.finditer(r'"offer":\s*"([A-Z]+_\d{3})"', raw)})
    offer_id = re.search(r'"offer":\s*"([A-Z]+_\d{3})"', raw).group(1)
    reasons = [t for t in flat if len(t) > 160]
    md += ["## 3. Member lookup (Lakebase read through the app)", "",
           f"Callback `member-dropdown.value = {member}` -> HTTP {status} in {ms:.0f} ms.", "",
           f"Offers rendered: {', '.join(ids)} (top offer `{offer_id}`).", "",
           "Rendered page text (first 40 text nodes):", "", "```text", "\n".join(flat[:40]), "```", "",
           f"First AI recommendation insight shown to the care manager:", "", f"> {reasons[0] if reasons else 'n/a'}", ""]

    approve = {"approve-btn": {"member": member, "offer": offer_id, "type": "approve-btn"},
               "reject-btn": {"member": member, "offer": offer_id, "type": "reject-btn"}}
    out_ids = [{"member": member, "offer": offer_id, "type": t} for t in ("feedback-status", "feedback-status", "approve-btn", "reject-btn")]
    body = {"output": _spec([("feedback-status", "children"), ("feedback-status", "style"), ("approve-btn", "disabled"), ("reject-btn", "disabled")]),
            "outputs": [{"id": out_ids[0], "property": "children"}, {"id": out_ids[1], "property": "style"},
                        {"id": out_ids[2], "property": "disabled"}, {"id": out_ids[3], "property": "disabled"}],
            "inputs": [{"id": approve["approve-btn"], "property": "n_clicks", "value": 1},
                       {"id": approve["reject-btn"], "property": "n_clicks", "value": None}],
            "changedPropIds": [pid(approve["approve-btn"]) + ".n_clicks"], "state": []}
    status, ms, raw = http(tok, "/_dash-update-component", body)
    approve_status = json.loads(raw)["response"][pid(out_ids[0])]["children"]
    md += ["## 4. Approve the top offer (write to Delta)", "",
           f"Callback `approve-btn.n_clicks` for `{member}` / `{offer_id}` -> HTTP {status} in {ms:.0f} ms; "
           f"app replied **{approve_status}**.", ""]

    comment = "Member asked for a call back Tuesday; O'Neil said: don't email -- ok; DROP TABLE x;"
    tb = {"member": member, "offer": offer_id, "type": "submit-text-btn"}
    ft = {"member": member, "offer": offer_id, "type": "feedback-text"}
    ts = {"member": member, "offer": offer_id, "type": "text-feedback-status"}
    body = {"output": _spec([("text-feedback-status", "children"), ("text-feedback-status", "style"), ("submit-text-btn", "disabled"), ("feedback-text", "disabled")]),
            "outputs": [{"id": ts, "property": "children"}, {"id": ts, "property": "style"},
                        {"id": tb, "property": "disabled"}, {"id": ft, "property": "disabled"}],
            "inputs": [{"id": tb, "property": "n_clicks", "value": 1}],
            "state": [{"id": ft, "property": "value", "value": comment}],
            "changedPropIds": [pid(tb) + ".n_clicks"]}
    status, ms, raw = http(tok, "/_dash-update-component", body)
    text_status = json.loads(raw)["response"][pid(ts)]["children"]
    md += ["## 5. Submit a care-manager comment (parameterized SQL)", "",
           f"The comment deliberately contains quotes and SQL-like text: `{comment}`", "",
           f"Callback `submit-text-btn.n_clicks` -> HTTP {status} in {ms:.0f} ms; app replied **{text_status}**.", ""]

    time.sleep(3)
    rows = sql(a.profile, f"""SELECT member_id, offer_id, feedback, feedback_text, feedback_time FROM {FEEDBACK_TABLE}
                              WHERE member_id = '{member}' AND feedback_time >= '{started.strftime('%Y-%m-%d %H:%M:%S')}'
                              ORDER BY feedback_time""")
    md += ["## 6. Feedback read back from Unity Catalog", "", f"`SELECT ... FROM {FEEDBACK_TABLE} WHERE member_id = '{member}'`", "",
           "| member_id | offer_id | feedback | feedback_text | feedback_time |", "| --- | --- | --- | --- | --- |"]
    md += [f"| {r['member_id']} | {r['offer_id']} | {r['feedback']} | {str(r['feedback_text']).replace('|', '/')} | {r['feedback_time']} |" for r in rows]
    ok = any(r["feedback"] == "approved" for r in rows) and any(r["feedback_text"] == comment for r in rows)
    md += ["", f"**Result: {'PASS' if ok else 'FAIL'}**: the approval and the comment (stored verbatim, quotes "
               f"and SQL text intact) are in the governed Delta table the retraining notebook reads.", ""]
    with open(a.out, "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


def _walk(node):
    yield node
    if isinstance(node, dict):
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _spec(outputs):
    """Multi-output pattern-matching callback spec string, as registered by Dash."""
    return ".." + "...".join(f'{{"member":["MATCH"],"offer":["MATCH"],"type":"{t}"}}.{prop}' for t, prop in outputs) + ".."


if __name__ == "__main__":
    main()

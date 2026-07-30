"""
Phase 2 — Okta System Log puller.

Pulls real System Log events from an Okta Developer org via the Okta API,
and normalizes them into a shape close to the pipeline's existing
access_logs_unlabeled.csv schema (entity_id, entity_type, timestamp,
source_ip, geo_location, resource_accessed, auth_method,
session_duration_sec, command_sequence, device_fingerprint,
failed_auth_attempts) — with clear notes on what fields real Okta data
does NOT actually give us cleanly, since this affects Phase 3's feature
code.

Usage:
    python okta_puller.py --hours 24 --limit 1000 --out data/okta_raw_events.json
"""

import os
import argparse
import json
import time
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

ORG_URL = os.environ.get("OKTA_ORG_URL", "").rstrip("/")
API_TOKEN = os.environ.get("OKTA_API_TOKEN", "")


def fetch_system_logs(since_hours=24, limit=1000):
    """
    Pulls raw System Log events from Okta, following pagination via the
    'next' Link header until either we run out of results or hit `limit`.
    """
    if not ORG_URL or not API_TOKEN:
        raise RuntimeError(
            "OKTA_ORG_URL and OKTA_API_TOKEN must be set in .env before running this script."
        )

    since = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )

    url = f"{ORG_URL}/api/v1/logs"
    headers = {
        "Authorization": f"SSWS {API_TOKEN}",
        "Accept": "application/json",
    }
    params = {"since": since, "sortOrder": "ASCENDING", "limit": min(limit, 1000)}

    all_events = []
    while url and len(all_events) < limit:
        resp = requests.get(url, headers=headers, params=params, timeout=15)

        if resp.status_code == 429:
            # Okta rate limit — respect the reset header and retry.
            reset_at = int(resp.headers.get("X-Rate-Limit-Reset", time.time() + 5))
            wait = max(reset_at - time.time(), 1)
            print(f"Rate limited. Waiting {wait:.0f}s...")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        batch = resp.json()
        all_events.extend(batch)
        print(f"Pulled {len(batch)} events (total so far: {len(all_events)})")

        # Pagination: Okta returns a 'next' link in the Link header once
        # subsequent params are baked in, so we stop passing `params` after
        # the first request.
        params = None
        link_header = resp.headers.get("Link", "")
        next_url = None
        for part in link_header.split(","):
            if 'rel="next"' in part:
                next_url = part[part.find("<") + 1 : part.find(">")]
        url = next_url

        if not batch:
            break  # no more events, even if a next link is technically present

    return all_events[:limit]


def normalize_event(evt):
    """
    Maps one raw Okta System Log event onto a row shape close to the
    pipeline's existing schema.

    IMPORTANT — what real data breaks that synthetic data hid:

    1. entity_id: Okta events are keyed by `actor.id` (a real Okta user ID
       like '00u1a2b3c4d5e6f7g8h9') rather than the clean 'user_XXXXXXXX'
       format generate_data.py invented. Downstream code that assumes a
       specific ID format/length needs to treat entity_id as an opaque
       string, not something to parse.

    2. entity_type: Okta doesn't cleanly separate 'user' vs 'service_account'
       vs 'edge_device' the way the synthetic generator does. We infer it
       from `actor.type` (e.g. 'User', 'PublicClientApp', 'SystemPrincipal'),
       but many real service-to-service calls show up as generic actor
       types that don't map 1:1 onto our three synthetic categories — this
       needs a fallback ('unknown') rather than forcing a guess.

    3. session_duration_sec: Okta's System Log is an EVENT log, not a
       SESSION log — there is no single event that hands you "this session
       lasted N seconds" the way the synthetic generator did by
       construction. We do NOT fabricate this; it's left as None here, and
       real session duration would need to be derived later by pairing
       session-start and session-end events per entity, which is nontrivial
       and won't always succeed (some sessions never see a clean end event
       e.g. a token simply expiring silently).

    4. geo_location: Okta gives structured geographical data
       (client.geographicalContext.city/state/country) when available, but
       this is frequently NULL/missing for legitimate reasons (VPNs,
       corporate proxies, mobile carriers not resolving to a specific city).
       Missing geo isn't itself suspicious and must not be treated as such.

    5. device_fingerprint: Okta gives `client.userAgent.rawUserAgent` and
       `client.device`, not a single stable device ID the way synthetic
       data invented one. Two events from the "same" physical device can
       show slightly different raw user-agent strings (browser version
       auto-updates between sessions), so naive exact-string matching will
       undercount a device's repeat visits.

    6. failed_auth_attempts: Okta logs each failed attempt as its OWN
       separate event (`user.authentication.auth_via_mfa` failures,
       `user.session.start` outcome=FAILURE, etc.) rather than a single
       session row with a failure COUNT. Getting a per-session failure
       count requires grouping raw events by actor + a time window
       ourselves — again, a real aggregation step, not a free field.

    7. resource_accessed / command_sequence: Okta's log describes the
       AUTH EVENT ITSELF (what type of auth flow, which app), not which
       internal resources/commands were touched afterward — that's a
       different system entirely (app-level audit logs) not present here.
       We map `target[].displayName` as a rough proxy, but it is NOT the
       same concept as "which internal resources did this session touch."
    """
    actor = evt.get("actor") or {}
    client = evt.get("client") or {}
    geo = (client.get("geographicalContext") or {})
    outcome = (evt.get("outcome") or {})
    targets = evt.get("target") or []

    return {
        "session_id": evt.get("uuid"),
        "entity_id": actor.get("id"),
        "entity_type": actor.get("type") or "unknown",
        "timestamp": evt.get("published"),
        "source_ip": client.get("ipAddress"),
        "geo_location": ", ".join(
            filter(None, [geo.get("city"), geo.get("state"), geo.get("country")])
        )
        or None,
        "resource_accessed": ", ".join(t.get("displayName") or "" for t in targets) or None,
        "auth_method": (client.get("authenticationContext") or {}).get(
            "authenticationProvider"
        ),
        "session_duration_sec": None,  # see normalize_event docstring, point 3
        "command_sequence": evt.get("eventType"),
        "device_fingerprint": (client.get("userAgent") or {}).get("rawUserAgent"),
        "failed_auth_attempts": 1 if outcome.get("result") == "FAILURE" else 0,
        "raw_outcome_result": outcome.get("result"),
        "raw_event_type": evt.get("eventType"),
    }

def main():
    parser = argparse.ArgumentParser(description="Pull Okta System Log events.")
    parser.add_argument("--hours", type=int, default=24, help="How far back to pull events from.")
    parser.add_argument("--limit", type=int, default=1000, help="Max number of events to pull.")
    parser.add_argument(
        "--out", type=str, default="data/okta_raw_events.json", help="Output JSON file path."
    )
    args = parser.parse_args()

    print(f"Pulling Okta System Log events from the last {args.hours}h (limit {args.limit})...")
    raw_events = fetch_system_logs(since_hours=args.hours, limit=args.limit)
    print(f"Pulled {len(raw_events)} raw events.")

    normalized = [normalize_event(e) for e in raw_events]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"raw_events": raw_events, "normalized_events": normalized}, f, indent=2)

    print(f"Wrote {len(normalized)} normalized events to {args.out}")
    print("(Both raw and normalized events are saved, so nothing from Okta's original")
    print(" event is lost even though normalize_event() only maps a subset of fields.)")


if __name__ == "__main__":
    main()
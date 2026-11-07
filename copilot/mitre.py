"""MITRE ATT&CK mappings + response playbooks for predicted attack types."""

from __future__ import annotations

MITRE_MAP = {
    "brute_force": {
        "technique_id": "T1110",
        "technique": "Brute Force",
        "tactic": "Credential Access",
        "description": "Repeated authentication attempts against one or more accounts.",
        "response": [
            "Lock or temporarily disable the targeted account after confirming with the owner.",
            "Block or rate-limit the source IP at the edge.",
            "Require MFA re-enrollment if any attempt succeeded.",
            "Review successful logins from the same IP for the last 24 hours.",
        ],
    },
    "impossible_travel": {
        "technique_id": "T1078",
        "technique": "Valid Accounts",
        "tactic": "Initial Access / Defense Evasion",
        "description": "Authentication from two locations that cannot be reached in the elapsed time.",
        "response": [
            "Verify the user's current location out of band.",
            "Revoke active sessions and refresh tokens for the account.",
            "Check for VPN/proxy egress that could explain the location jump.",
            "Review resource access from both locations.",
        ],
    },
    "credential_stuffing": {
        "technique_id": "T1110.004",
        "technique": "Credential Stuffing",
        "tactic": "Credential Access",
        "description": "One source IP attempts many accounts using likely leaked credentials.",
        "response": [
            "Block the source IP and any related ASN temporarily.",
            "Force password resets for accounts with successful attempts.",
            "Check breach-intelligence exposure for targeted accounts.",
            "Enable adaptive MFA / impossible-travel policies.",
        ],
    },
    "lateral_movement": {
        "technique_id": "T1021",
        "technique": "Remote Services",
        "tactic": "Lateral Movement",
        "description": "An entity touches an unusual sequence of internal resources in one session.",
        "response": [
            "Isolate the source host or device from the network.",
            "Review the resource chain for sensitive data access.",
            "Check for new service creation, scheduled tasks, or credential dumping.",
            "Reset credentials used from the source host.",
        ],
    },
    "device_spoofing": {
        "technique_id": "T1036",
        "technique": "Masquerading",
        "tactic": "Defense Evasion",
        "description": "A known entity suddenly presents a new or inconsistent device fingerprint.",
        "response": [
            "Verify the device with the user or device owner.",
            "Require device compliance / certificate re-enrollment.",
            "Look for concurrent sessions from the old and new fingerprints.",
            "Block the new fingerprint until verified.",
        ],
    },
    "low_and_slow_exfiltration": {
        "technique_id": "T1030",
        "technique": "Data Transfer Size Limits",
        "tactic": "Exfiltration",
        "description": "Small, repeated transfers that stay below per-session volume thresholds.",
        "response": [
            "Quantify total data moved by the entity over the last 7 days.",
            "Review destination IPs/domains and data classification.",
            "Throttle or block the destination if it is not sanctioned.",
            "Preserve logs before containment for incident response.",
        ],
    },
    "insider_drift": {
        "technique_id": "T1078",
        "technique": "Valid Accounts",
        "tactic": "Initial Access",
        "description": "A gradual behaviour change that may be legitimate role drift or a low-signal insider.",
        "response": [
            "Compare the behaviour change against HR/role changes.",
            "Ask the user's manager to confirm new responsibilities.",
            "Monitor for escalation in resource sensitivity or volume.",
            "Do not auto-contain until the change is confirmed malicious.",
        ],
    },
}

DEFAULT = {
    "technique_id": "T1078",
    "technique": "Valid Accounts",
    "tactic": "Initial Access",
    "description": "Suspicious access using an account that may be compromised.",
    "response": [
        "Verify the activity with the account owner.",
        "Review recent sessions and resource access for the entity.",
        "Revoke sessions if the activity cannot be explained.",
    ],
}


def get_mitre(attack_type):
    if not attack_type:
        return dict(DEFAULT)
    key = str(attack_type).strip().lower().replace(" ", "_").replace("-", "_")
    return dict(MITRE_MAP.get(key, DEFAULT))


def all_techniques():
    return {key: dict(value) for key, value in MITRE_MAP.items()}

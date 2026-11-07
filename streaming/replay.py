"""Replay synthetic (or exported Okta) access logs into Kafka.

This is the demo driver: it gives the streaming pipeline a realistic event
flow without needing a live Okta org. Events are published in timestamp order
and keyed by ``entity_id`` so per-entity ordering is preserved.

Usage:
    python streaming/replay.py --limit 5000 --delay-ms 25

Ground-truth labels are merged in only so the demo dashboard can show whether
an alert was genuinely an attack. The models never read the ``label`` field.
"""

import argparse
import json
import os
import time

import pandas as pd
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
from dotenv import load_dotenv

load_dotenv()

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


def ensure_topic(topic_name: str, partitions: int = 6):
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP_SERVERS})
    existing = admin.list_topics(timeout=10).topics
    if topic_name in existing:
        print(f"Topic '{topic_name}' already exists")
        return
    futures = admin.create_topics(
        [NewTopic(topic_name, num_partitions=partitions, replication_factor=1)]
    )
    for name, future in futures.items():
        future.result(timeout=10)
        print(f"Created topic '{name}' with {partitions} partitions")


def _clean(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description="Replay access logs into Kafka.")
    parser.add_argument("--input", default="data/access_logs_unlabeled.csv")
    parser.add_argument("--labels", default="data/ground_truth_labels.csv")
    parser.add_argument("--topic", default="raw_sessions")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--delay-ms", type=float, default=0.0)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "entity_id", "session_id"])
    frame = frame.sort_values("timestamp")

    if args.labels and os.path.exists(args.labels):
        labels = pd.read_csv(args.labels)
        keep = [column for column in ("session_id", "label", "anomaly_type") if column in labels]
        frame = frame.merge(labels[keep], on="session_id", how="left")

    if args.limit and args.limit > 0:
        frame = frame.head(args.limit)

    ensure_topic(args.topic)
    producer = Producer({"bootstrap.servers": BOOTSTRAP_SERVERS})

    sent = 0
    for _, row in frame.iterrows():
        event = {key: _clean(value) for key, value in row.to_dict().items()}
        event["timestamp_epoch"] = int(row["timestamp"].timestamp())
        event["timestamp"] = row["timestamp"].isoformat()

        producer.produce(
            args.topic,
            key=str(event["entity_id"]).encode("utf-8"),
            value=json.dumps(event, default=str).encode("utf-8"),
        )
        producer.poll(0)
        sent += 1

        if sent % 1000 == 0:
            print(f"Replayed {sent}/{len(frame)} events")
        if args.delay_ms:
            time.sleep(args.delay_ms / 1000)

    producer.flush(timeout=30)
    print(f"Replayed {sent} events into '{args.topic}'")


if __name__ == "__main__":
    main()

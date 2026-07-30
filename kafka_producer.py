"""
Phase 2 — Kafka producer.

Reads the normalized events written by okta_puller.py and publishes each
one onto a Kafka topic, keyed by entity_id so that all of one entity's
events land on the same partition (needed for ordered, stateful
per-entity processing in Phase 3's streaming features).

Usage:
    python kafka_producer.py --in data/okta_raw_events.json --topic access-logs
"""

import os
import json
import argparse
import time

from confluent_kafka import Producer, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic
from dotenv import load_dotenv

load_dotenv()

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


def ensure_topic(topic_name, num_partitions=6, replication_factor=1):
    """
    Creates the topic if it doesn't already exist. num_partitions=6 gives
    entity-keyed messages somewhere to actually spread out across — with
    only 1 partition, keying by entity_id would be pointless since
    everything lands in the same place anyway.
    """
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP_SERVERS})
    existing_topics = admin.list_topics(timeout=10).topics

    if topic_name in existing_topics:
        print(f"Topic '{topic_name}' already exists — skipping creation.")
        return

    new_topic = NewTopic(
        topic_name, num_partitions=num_partitions, replication_factor=replication_factor
    )
    futures = admin.create_topics([new_topic])
    for name, future in futures.items():
        try:
            future.result()
            print(f"Created topic '{name}' with {num_partitions} partitions.")
        except KafkaException as e:
            print(f"Failed to create topic '{name}': {e}")
            raise


def delivery_report(err, msg):
    """Called once per message, either on success or permanent failure."""
    if err is not None:
        print(f"Delivery failed for key={msg.key()}: {err}")
    # Not printing every successful delivery individually — too noisy for
    # hundreds of events. Summary counts are printed at the end instead.


def produce_events(events, topic_name):
    producer = Producer({"bootstrap.servers": BOOTSTRAP_SERVERS})

    sent = 0
    skipped_no_entity = 0

    for event in events:
        entity_id = event.get("entity_id")
        if not entity_id:
            # Some Okta events genuinely have no actor (system-level events
            # with a null actor.id) — these can't be partitioned by entity,
            # so we skip them here rather than silently keying on "None".
            skipped_no_entity += 1
            continue

        key = entity_id.encode("utf-8")
        value = json.dumps(event).encode("utf-8")

        producer.produce(topic_name, key=key, value=value, callback=delivery_report)
        # poll(0) triggers any pending delivery callbacks without blocking —
        # necessary so the internal librdkafka queue doesn't fill up on a
        # large batch.
        producer.poll(0)
        sent += 1

    # Block until all outstanding messages are delivered or fail.
    producer.flush(timeout=30)

    print(f"Sent {sent} events to topic '{topic_name}'.")
    if skipped_no_entity:
        print(f"Skipped {skipped_no_entity} events with no entity_id (couldn't be partitioned).")


def main():
    parser = argparse.ArgumentParser(description="Publish normalized Okta events to Kafka.")
    parser.add_argument(
        "--in", dest="input_path", type=str, default="data/okta_raw_events.json"
    )
    parser.add_argument("--topic", type=str, default="access-logs")
    parser.add_argument("--partitions", type=int, default=6)
    args = parser.parse_args()

    with open(args.input_path) as f:
        data = json.load(f)
    events = data["normalized_events"]
    print(f"Loaded {len(events)} normalized events from {args.input_path}")

    ensure_topic(args.topic, num_partitions=args.partitions)
    produce_events(events, args.topic)


if __name__ == "__main__":
    main()
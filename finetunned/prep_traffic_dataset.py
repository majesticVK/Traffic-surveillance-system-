import json
import random
import argparse


LOCATIONS = ["Main St", "Ring Road", "Highway 5", "School Zone", "Market Crossing"]
VEHICLE_COUNTS = [0, 1, 2, 3, 5, 8, 12, 20]
LIGHT_STATES = ["red", "green", "yellow", "unknown"]
SPEEDS = [0, 30, 45, 60, 75, 90, 110]
TIMES = ["08:15", "12:30", "17:45", "22:10", "02:30", "06:00"]

QA_TEMPLATES = [
    (
        "{count} vehicles detected at {loc} at {time}, traffic light is {light}",
        lambda c, loc, t, l, s: (
            f"Heavy congestion detected at {loc} — {c} vehicles present at {t}. "
            f"The traffic light shows {l}. Recommend monitoring for signal compliance."
            if c >= 5 else
            f"Light traffic at {loc} at {t} with {c} vehicle(s). "
            f"Signal state is {l}. Conditions appear normal."
        ),
    ),
    (
        "Vehicle detected in intersection at {loc} while traffic light is red at {time}",
        lambda c, loc, t, l, s: (
            f"A red-light violation was recorded at {loc} at {t}. "
            f"A vehicle entered the intersection against a red signal. "
            f"This is a high-risk event and should be flagged for review."
        ),
    ),
    (
        "Vehicle travelling at {speed} km/h in a {limit} km/h zone at {loc}",
        lambda c, loc, t, l, s: (
            f"Speed violation detected at {loc}: vehicle recorded at {s} km/h "
            f"in a 60 km/h zone. This exceeds the limit by {s-60} km/h and "
            f"constitutes a traffic violation."
            if s > 60 else
            f"Vehicle speed at {loc} is {s} km/h, within the posted limit. No violation."
        ),
    ),
    (
        "No vehicles detected at {loc} at {time}, traffic light is {light}",
        lambda c, loc, t, l, s: (
            f"No traffic activity at {loc} at {t}. Road is clear. "
            f"Traffic light is showing {l}."
        ),
    ),
    (
        "Pedestrian detected on road at {loc} at {time}",
        lambda c, loc, t, l, s: (
            f"A pedestrian was detected on the road at {loc} at {t}. "
            f"This is a safety risk — vehicles in the area should be alerted. "
            f"Flagging as high-risk event."
        ),
    ),
]


def create_sample(template_tuple):
    q_tmpl, a_fn = template_tuple
    count = random.choice(VEHICLE_COUNTS)
    loc = random.choice(LOCATIONS)
    t = random.choice(TIMES)
    light = random.choice(LIGHT_STATES)
    speed = random.choice(SPEEDS)

    question = q_tmpl.format(count=count, loc=loc, time=t, light=light, speed=speed, limit=60)
    answer = a_fn(count, loc, t, light, speed)
    return {"question": question, "answer": answer}


def generate_synthetic(output_file, num_samples=200):
    with open(output_file, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            sample = create_sample(random.choice(QA_TEMPLATES))
            f.write(json.dumps(sample) + "\n")
    print(f"Wrote {num_samples} synthetic samples to {output_file}")


def events_to_dataset(event_file, output_file):
    """Convert recorded TrafficEvents to training samples."""
    with open(event_file) as f:
        events = json.load(f)

    with open(output_file, "w") as out:
        for e in events:
            question = (
                f"{e['vehicle_count']} vehicles at {e['location']} at {e['timestamp'][:16]}, "
                f"light: {e['light_state']}, speed: {e.get('speed_kmh', 0):.0f} km/h"
            )
            answer = (
                f"Traffic event recorded: {e['event_type']} at {e['location']}. "
                f"{e['vehicle_count']} vehicle(s) present, signal state was {e['light_state']}."
            )
            out.write(json.dumps({"question": question, "answer": answer}) + "\n")

    print(f"Converted {len(events)} events to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=int, help="Generate N synthetic samples")
    parser.add_argument("--from_events", type=str, help="Path to traffic_events.json")
    parser.add_argument("--output", type=str, default="traffic_qa.jsonl")
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic(args.output, args.synthetic)
    elif args.from_events:
        events_to_dataset(args.from_events, args.output)
    else:
        print("Usage: --synthetic 200   or   --from_events traffic_events.json")

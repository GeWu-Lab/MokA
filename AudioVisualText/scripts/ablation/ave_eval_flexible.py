import argparse
import json
import re
from pathlib import Path

import jsonlines
import numpy as np
from sklearn.metrics import accuracy_score


def load_event_mapping(annotations_path: Path):
    vocab = set()
    with annotations_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vocab.add(line.split("&")[0])
    mapping = {"none": 0}
    for idx, event in enumerate(sorted(vocab)):
        mapping[event.lower()] = idx + 1
    return mapping


def parse_pred_event_and_ranges(pred_text: str, mapping):
    matches = re.findall(r"<event>(.*?)</event>", pred_text)
    if len(matches) != 1:
        return None, None

    event_content = matches[0].strip()
    pred_event_temp = event_content.lower()
    pred_ranges = []

    if pred_event_temp in mapping:
        pred_event = pred_event_temp
        ranges = re.findall(r"<range>(.*?)</range>", pred_text)
        if len(ranges) == 0:
            return None, None
        for range_str in ranges:
            parts = range_str.strip().split(",")
            if len(parts) != 2:
                continue
            try:
                pred_start = int(parts[0].strip())
                pred_end = int(parts[1].strip())
            except ValueError:
                continue
            pred_ranges.append((pred_start, pred_end))
        if len(pred_ranges) == 0:
            return None, None
        return pred_event, pred_ranges

    # Fallback format used in some generations:
    # <event>Some event (0 3), (7 10)</event>
    time_matches = re.findall(r"\(\s*(\d+)\s+(\d+)\s*\)", event_content)
    if len(time_matches) == 0:
        return None, None
    for start_str, end_str in time_matches:
        pred_ranges.append((int(start_str), int(end_str)))

    first_range_match = re.search(r"\(\s*\d+\s+\d+\s*\)", event_content)
    if first_range_match is None:
        return None, None
    pred_event = event_content[: first_range_match.start()].strip().rstrip(",").lower()
    if pred_event not in mapping:
        return None, None
    return pred_event, pred_ranges


def evaluate(inference_path: Path, annotations_path: Path):
    mapping = load_event_mapping(annotations_path)
    rows = []
    with jsonlines.open(str(inference_path), "r") as reader:
        for sample in reader:
            rows.append(sample)

    n_samples = len(rows)
    n_frames = n_samples * 10
    pred_labels = np.zeros(n_frames)
    real_labels = np.zeros(n_frames)

    valid = 0
    c = 0
    for sample in rows:
        answer = sample["output"].replace("</s>", "").strip()
        pred = sample["predict"]

        event_match = re.findall(r"event:(.*?)start_time", answer)
        if len(event_match) != 1:
            c += 10
            continue
        event = event_match[0].strip().lower()
        if event not in mapping:
            c += 10
            continue

        try:
            start_time = int(answer.split(" ")[-2].split(":")[-1])
            end_time = int(answer.split(" ")[-1].split(":")[-1])
        except ValueError:
            c += 10
            continue

        pred_event, pred_ranges = parse_pred_event_and_ranges(pred, mapping)
        if pred_event is None:
            c += 10
            continue

        valid += 1
        for i in range(10):
            if start_time <= i <= end_time:
                real_labels[c] = mapping[event]
            if any(pred_start <= i <= pred_end for pred_start, pred_end in pred_ranges):
                pred_labels[c] = mapping[pred_event]
            c += 1

    acc = accuracy_score(real_labels, pred_labels)
    return {
        "inference_path": str(inference_path),
        "annotations_path": str(annotations_path),
        "total_samples": n_samples,
        "valid_prediction_format_samples": valid,
        "frame_accuracy": float(acc),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate AVE inference jsonl with configurable paths.")
    parser.add_argument("--inference-jsonl", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--out-json", required=False, type=Path, default=None)
    args = parser.parse_args()

    metrics = evaluate(args.inference_jsonl, args.annotations)
    print(json.dumps(metrics, indent=2))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

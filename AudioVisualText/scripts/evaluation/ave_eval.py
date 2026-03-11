import argparse
import json
import os

import jsonlines
import re
import numpy as np
from sklearn.metrics import accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl_path",
        type=str,
        default="inference_ave.jsonl",
        help="Path to inference_ave.jsonl (predictions).",
    )
    parser.add_argument(
        "--annotations_path",
        type=str,
        default="Annotations.txt",
        help="Path to AVE Annotations.txt.",
    )
    parser.add_argument(
        "--output_metrics_path",
        type=str,
        default=None,
        help="Optional path to write metrics JSON (e.g. metrics.jsonl).",
    )
    args = parser.parse_args()

    vocab = set()
    with open(args.annotations_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        event = line.split("&")[0]
        vocab.add(event)
    vocab = list(vocab)
    print(len(vocab), vocab)
    mapping = {"none": 0}
    for i, event in enumerate(vocab):
        event = event.lower()
        mapping[event] = i + 1
    print(mapping)

    path = args.jsonl_path
    N = 402 * 10
    c = 0
    pre_labels = np.zeros(N)
    real_labels = np.zeros(N)
    nums = 0

    skipped_reasons = {
        "event_mismatch": 0,
        "no_primary_ranges": 0,
        "no_valid_primary_ranges": 0,
        "primary_range_exception": 0,
        "no_secondary_ranges": 0,
        "no_secondary_parentheses": 0,
        "secondary_event_not_in_mapping": 0,
        "secondary_exception": 0,
    }

    idx = -1
    with jsonlines.open(path, "r") as f:
        for idx, sample in enumerate(f):
            answer = sample["output"]
            pred = sample["predict"]
            matches = re.findall(r"event:(.*?)start_time", answer)
            event = matches[0].strip().lower()

            answer = answer.replace("</s>", "")
            answer = answer.strip()
            start_time = int(answer.split(" ")[-2].split(":")[-1])
            end_time = int(answer.split(" ")[-1].split(":")[-1])

            matches = re.findall(r"<event>(.*?)</event>", pred)
            if len(matches) != 1:
                print("event != 1,  idx: ", idx, "pred: ", pred)
                skipped_reasons["event_mismatch"] += 1
                continue
            event_content = matches[0].strip()
            pred_event_temp = event_content.lower()

            pred_ranges = []
            if pred_event_temp in mapping:
                pred_event = pred_event_temp
                matches = re.findall(r"<range>(.*?)</range>", pred)
                if len(matches) == 0:
                    print("idx: ", idx, " no ranges found in pred: ", pred)
                    skipped_reasons["no_primary_ranges"] += 1
                    continue
                for range_str in matches:
                    try:
                        parts = range_str.strip().split(",")
                        if len(parts) != 2:
                            raise ValueError("Invalid range format")
                        pred_start = int(parts[0].strip())
                        pred_end = int(parts[1].strip())
                        pred_ranges.append((pred_start, pred_end))
                    except Exception:
                        print("****** idx: ", idx, " exception in primary range")
                        skipped_reasons["primary_range_exception"] += 1
                        continue
                if len(pred_ranges) == 0:
                    print("idx: ", idx, " no valid ranges in primary format")
                    skipped_reasons["no_valid_primary_ranges"] += 1
                    continue
            else:
                try:
                    time_matches = re.findall(r"\(\s*(\d+)\s+(\d+)\s*\)", event_content)
                    if len(time_matches) == 0:
                        print(
                            "****** idx: ",
                            idx,
                            " no time ranges found in secondary format",
                        )
                        skipped_reasons["no_secondary_ranges"] += 1
                        continue
                    for start_str, end_str in time_matches:
                        pred_start = int(start_str)
                        pred_end = int(end_str)
                        pred_ranges.append((pred_start, pred_end))

                    first_range_match = re.search(
                        r"\(\s*\d+\s+\d+\s*\)", event_content
                    )
                    if first_range_match is None:
                        print(
                            "****** idx: ",
                            idx,
                            " no parentheses in secondary format",
                        )
                        skipped_reasons["no_secondary_parentheses"] += 1
                        continue
                    pred_event = (
                        event_content[: first_range_match.start()]
                        .strip()
                        .rstrip(",")
                        .lower()
                    )
                    if pred_event not in mapping:
                        print("==== idx: ", idx, " event not in mapping:", pred_event)
                        skipped_reasons["secondary_event_not_in_mapping"] += 1
                        continue
                except Exception:
                    print("****** idx: ", idx, " secondary exception")
                    skipped_reasons["secondary_exception"] += 1
                    continue

            nums += 1
            for i in range(10):
                if i >= start_time and i <= end_time:
                    real_labels[c] = mapping[event]
                if any(pred_start <= i <= pred_end for pred_start, pred_end in pred_ranges):
                    pre_labels[c] = mapping[pred_event]
                c += 1

    print("tot: ", idx)
    real_labels = np.array(real_labels)
    pre_labels = np.array(pre_labels)
    acc = accuracy_score(real_labels, pre_labels)
    skipped = (idx + 1) - nums if idx >= 0 else 0
    print("c: ", c, " nums: ", nums, "skipped: ", skipped, "acc: ", acc)
    print("skipped detail:", skipped_reasons)

    if args.output_metrics_path:
        metrics = {
            "jsonl_path": os.path.abspath(path),
            "annotations_path": os.path.abspath(args.annotations_path),
            "total_samples": int(idx + 1 if idx >= 0 else 0),
            "valid_samples": int(nums),
            "frames": int(c),
            "skipped": int(skipped),
            "accuracy": float(acc),
            "skipped_detail": skipped_reasons,
        }
        with open(args.output_metrics_path, "w") as f:
            json.dump(metrics, f)
        print("Wrote metrics to", args.output_metrics_path)


if __name__ == "__main__":
    main()

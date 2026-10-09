"""Collect generation stats and policy evaluations into outputs/results.json and outputs/results.md."""

import glob
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..")


def gen_stats(path):
    rows = [json.loads(line) for line in open(path)]
    out = {"attempts": len(rows), "success": sum(r["success"] for r in rows)}
    if rows and rows[0]["nav"]:
        out["nav_reached"] = sum(all(n["reached"] for n in r["nav"]) for r in rows)
        by = defaultdict(lambda: [0, 0])
        for r in rows:
            n = r["nav"][0]
            d = math.dist(n["start"][:2], n["goal"][:2])
            k = f"{int(d * 2) / 2:.1f}-{int(d * 2) / 2 + 0.5:.1f}"
            by[k][0] += r["success"]
            by[k][1] += 1
        out["by_nav_distance"] = {k: {"success": v[0], "n": v[1]} for k, v in sorted(by.items())}
    return out


def main():
    res = {"generation": {}, "policies": {}}
    for path in sorted(glob.glob(os.path.join(ROOT, "datasets", "*", "*.stats.jsonl"))):
        res["generation"][os.path.relpath(path, ROOT).replace("\\", "/")] = gen_stats(path)
    for path in sorted(glob.glob(os.path.join(ROOT, "outputs", "policies", "*eval.json"))):
        s = json.load(open(path))["summary"]
        meta_path = os.path.join(os.path.dirname(path), os.path.basename(path).split(".")[0] + ".json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        res["policies"][os.path.basename(path)] = {**s, "train_demos": meta.get("n_demos"), "train_data": meta.get("data")}
    json.dump(res, open(os.path.join(ROOT, "outputs", "results.json"), "w"), indent=1)

    lines = ["| generation run | success / attempts | rate | nav reached |", "|---|---|---|---|"]
    for k, v in res["generation"].items():
        nav = f"{v['nav_reached']}/{v['attempts']}" if "nav_reached" in v else "-"
        lines.append(f"| {k} | {v['success']}/{v['attempts']} | {v['success'] / max(v['attempts'], 1):.0%} | {nav} |")
    lines += ["", "| policy eval | train demos | eval task | episodes | success |", "|---|---|---|---|---|"]
    for k, v in res["policies"].items():
        task = "static" if "Static" in v["task"] else "far"
        lines.append(f"| {k} | {v.get('train_demos')} | {task} | {v['n']} | {v['success_rate']:.1%} |")
    md = "\n".join(lines)
    open(os.path.join(ROOT, "outputs", "results.md"), "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    sys.exit(main())

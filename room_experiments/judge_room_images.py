"""Grade generated room images with an OpenAI vision model.

For every image the judge returns whether a couch is present, whether the
target object (framed painting or coffee table) is present, and a 0-3
warping/artifact score. Each request includes four human-labeled calibration
images (judge_calibration/) for the matching object pair.

The input root must contain one directory per cell and sampler:

    <input_root>/<cell>/<sampler>/**/*.png

where <cell> is one of FC_ID, FC_OOD, NFC_ID, NFC_OOD and <sampler> is a name
such as K1 or K16. FC cells are judged for a framed painting and NFC cells for
a coffee table. Per-image results are appended to judgments.jsonl in
--output_dir as they arrive. summary.json then holds, per group, the presence
rates, the joint presence rate (both objects present), and the mean artifact
score.

Requires the OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import IntEnum
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

MODEL = "gpt-5.6-luna"
DETAIL = "high"
REASONING_EFFORT = "none"
CELLS = ("FC_ID", "FC_OOD", "NFC_ID", "NFC_OOD")
CALIBRATION_DIR = Path(__file__).resolve().parent / "judge_calibration"
TARGETS = {"FC": "framed painting", "NFC": "coffee table"}

# (image file, couch_present, target_present, artifact_score) human labels.
CALIBRATION_EXAMPLES = {
    "FC": (
        ("FC_1.png", True, True, 0),
        ("FC_2.png", True, False, 2),
        ("FC_3.png", False, True, 2),
        ("FC_4.png", False, True, 0),
    ),
    "NFC": (
        ("NFC_1.png", True, False, 1),
        ("NFC_2.png", True, True, 2),
        ("NFC_3.png", True, False, 3),
        ("NFC_4.png", False, False, 3),
    ),
}


class ArtifactScore(IntEnum):
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3


class ImageJudgment(BaseModel):
    couch_present: bool
    target_present: bool
    artifact_score: ArtifactScore
    rationale: str


def family_of(cell: str) -> str:
    return cell.split("_", maxsplit=1)[0]


def image_data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def grading_prompt(target: str) -> str:
    return f"""Judge this generated room image on three independent fields.

Object presence:
- couch_present: true only if a visible object is identifiable as a couch/sofa
  from its shape and structure. Minor distortion is acceptable.
- target_present: true only if a visible object is identifiable as a {target}
  from its shape and structure. Minor distortion is acceptable.
- Do not infer presence from the expected room contents, placement alone, or a
  vague object-like blob. If the object is severely warped, merged with another
  object, or too ambiguous to identify confidently, mark it false.

Overall image warping/artifacting:
- 0: fully coherent; no visible generation-related deformation, melting,
  duplication, broken geometry, or meaningful texture artifacts.
- 1: mild; one or a few visible but localized flaws. Objects and room geometry
  remain structurally coherent.
- 2: moderate; multiple obvious flaws, or one major malformed/merged object or
  structural inconsistency, while the overall scene remains understandable.
- 3: severe; widespread or highly disruptive corruption, badly incoherent room
  geometry, or major objects that are substantially melted or merged.

Assess artifact_score across the whole image, including room geometry and all
objects. When between two artifact scores, choose the higher score. Give a
short factual rationale (25 words or fewer)."""


def calibration_content(family: str) -> list[dict[str, Any]]:
    """The rubric and the four labeled calibration images for one object pair."""
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                grading_prompt(TARGETS[family])
                + "\n\nThe following calibration examples have authoritative "
                "human labels. Use them to calibrate the rubric; do not rescore them."
            ),
        }
    ]
    for index, (filename, couch, target_present, artifact) in enumerate(
        CALIBRATION_EXAMPLES[family],
        start=1,
    ):
        content.extend(
            [
                {
                    "type": "input_text",
                    "text": (
                        f"Calibration example {index} human label: "
                        f"couch_present={str(couch).lower()}, "
                        f"target_present={str(target_present).lower()}, "
                        f"artifact_score={artifact}. "
                        "The next image is this example."
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": image_data_url(CALIBRATION_DIR / filename),
                    "detail": DETAIL,
                },
            ]
        )
    content.append(
        {
            "type": "input_text",
            "text": (
                "Now independently judge the candidate image below and return only "
                "the requested schema."
            ),
        }
    )
    return content


def judge_one(client: OpenAI, image_path: Path,
              calibration_prefix: list[dict[str, Any]]) -> ImageJudgment:
    response = client.responses.parse(
        model=MODEL,
        reasoning={"effort": REASONING_EFFORT},
        instructions=(
            "You are a consistent image-evaluation judge. Follow the "
            "provided rubric literally and return only the requested schema."
        ),
        input=[
            {
                "role": "user",
                "content": [
                    *calibration_prefix,
                    {
                        "type": "input_image",
                        "image_url": image_data_url(image_path),
                        "detail": DETAIL,
                    },
                ],
            }
        ],
        text_format=ImageJudgment,
        max_output_tokens=256,
        store=False,
    )
    if response.output_parsed is None:
        raise ValueError(f"Model returned no parsed output: {response.output_text!r}")
    return response.output_parsed


def summarize(records: list[dict[str, Any]], groups: list[tuple[str, str]]) -> dict[str, Any]:
    summary = {}
    for cell, sampler in groups:
        rows = [r for r in records
                if r["cell"] == cell and r["sampler"] == sampler and r["status"] == "ok"]
        n = len(rows)

        def rate(predicate) -> float | None:
            return sum(predicate(row) for row in rows) / n if n else None

        summary[f"{cell}/{sampler}"] = {
            "target_object": TARGETS[family_of(cell)],
            "judged_images": n,
            "joint_present_rate": rate(lambda r: r["couch_present"] and r["target_present"]),
            "couch_present_rate": rate(lambda r: r["couch_present"]),
            "target_present_rate": rate(lambda r: r["target_present"]),
            "mean_artifact_score": rate(lambda r: r["artifact_score"]),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--samplers", nargs="+", default=["K1", "K16"],
                        help="Sampler subdirectories to judge.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Maximum simultaneous API requests.")
    args = parser.parse_args()

    groups = [(cell, sampler) for cell in CELLS for sampler in args.samplers]
    jobs = []
    for cell, sampler in groups:
        images = sorted((args.input_root / cell / sampler).rglob("*.png"))
        if not images:
            parser.error(f"no images in {args.input_root / cell / sampler}")
        print(f"  {cell}/{sampler}: {len(images)} images")
        jobs += [(cell, sampler, path) for path in images]

    client = OpenAI(max_retries=5)
    prefixes = {family: calibration_content(family) for family in TARGETS}

    def evaluate(cell: str, sampler: str, path: Path) -> dict[str, Any]:
        record = {"cell": cell, "sampler": sampler, "image_path": str(path)}
        try:
            judgment = judge_one(client, path, prefixes[family_of(cell)])
            return {**record, "status": "ok", **judgment.model_dump(mode="json")}
        except Exception as exc:
            return {**record, "status": "error", "error": f"{type(exc).__name__}: {exc}"}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with (
        (args.output_dir / "judgments.jsonl").open("w", encoding="utf-8") as out,
        ThreadPoolExecutor(max_workers=args.concurrency) as executor,
    ):
        for future in as_completed([executor.submit(evaluate, *job) for job in jobs]):
            record = future.result()
            records.append(record)
            out.write(json.dumps(record) + "\n")
            out.flush()
            if record["status"] != "ok":
                print(f"ERROR {record['image_path']}: {record['error']}", file=sys.stderr)

    summary = summarize(records, groups)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for group, values in summary.items():
        if not values["judged_images"]:
            print(f"{group:12s} no images judged successfully")
            continue
        print(f"{group:12s} joint={values['joint_present_rate']:.3f} "
              f"couch={values['couch_present_rate']:.3f} "
              f"target={values['target_present_rate']:.3f} "
              f"artifact={values['mean_artifact_score']:.2f}")
    failed = sum(r["status"] != "ok" for r in records)
    print(f"{len(records) - failed} judged, {failed} failed. Results in {args.output_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

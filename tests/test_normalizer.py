import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from normalizer import NormalizerError, normalize


EXAMPLES = [
    "kal tui ashbi na? amar bday ache, cake niye aay",
    "yaar kal office nahi jaana, chal ghumne chalte hain",
    "bhai eta ekdom perfect, thank you so much",
    "sorry, ami ektu late hobo, traffic khub bad aaj",
    "kemon acho? sob kichu thik ache?",
    "bro ei weekend e plan ki? kolkata te ghurte jabo naki?",
]


def has_required_shape(result: dict) -> bool:
    top_level_ok = all(
        key in result
        for key in ("native_script", "english_translation", "language_ratio")
    )
    if not top_level_ok or not isinstance(result.get("language_ratio"), dict):
        return False

    ratio = result["language_ratio"]
    return all(
        key in ratio
        for key in ("bengali_pct", "hindi_pct", "english_pct")
    )


def ratio_sums_to_roughly_100(result: dict) -> bool:
    ratio = result["language_ratio"]
    total = ratio["bengali_pct"] + ratio["hindi_pct"] + ratio["english_pct"]
    return 95 <= total <= 105


def run_example(text: str) -> bool:
    print(f"\nINPUT: {text}")
    result = normalize(text)
    print(f"RESULT: {result}")

    shape_ok = has_required_shape(result)
    ratio_ok = shape_ok and ratio_sums_to_roughly_100(result)
    print(f"required keys: {'PASS' if shape_ok else 'FAIL'}")
    print(f"ratio sum within 5 of 100: {'PASS' if ratio_ok else 'FAIL'}")
    return shape_ok and ratio_ok


def main() -> None:
    all_examples_ok = True
    for example in EXAMPLES:
        try:
            all_examples_ok = run_example(example) and all_examples_ok
        except NormalizerError as exc:
            print(f"\nINPUT: {example}")
            print(f"ERROR: {exc}")
            all_examples_ok = False

    try:
        normalize("")
    except NormalizerError as exc:
        print(f"\nEMPTY INPUT: PASS ({exc})")
        empty_ok = True
    else:
        print("\nEMPTY INPUT: FAIL (expected NormalizerError)")
        empty_ok = False

    try:
        result = normalize("asdf qwer zxcv 12345 !!!")
    except NormalizerError as exc:
        print(f"\nGIBBERISH INPUT: PASS (caught NormalizerError: {exc})")
        gibberish_ok = True
    else:
        print(f"\nGIBBERISH INPUT: PASS (no crash: {result})")
        gibberish_ok = True

    overall = all_examples_ok and empty_ok and gibberish_ok
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()

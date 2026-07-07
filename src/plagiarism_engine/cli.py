from evaluation import (
    Evaluator,
    load_ground_truth_from_dataset,
    get_all_suspicious_ids,
    save_metrics,
    print_metrics
)
import sys
import os
from pathlib import Path
import argparse
import json
import csv
import importlib
from typing import List, Tuple, Dict, Set, Optional, Any

src_path = Path(__file__).resolve().parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print(" Starting plagiarism_engine CLI...", file=sys.stderr, flush=True)


def print_flush(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


def input_with_default(prompt: str, default: Any = None, validator=None) -> str:
    prompt = f"{prompt}: "
    while True:
        try:
            value = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print_flush("\n Input not received. Exiting.")
            sys.exit(0)
        if not value and default is not None:
            return str(default)
        if validator is None or validator(value):
            return value
        print_flush(" Invalid input. Please try again.")


def validate_file_exists(path_str: str) -> bool:
    return Path(path_str).exists()


def validate_positive_number(value: str) -> bool:
    try:
        return float(value) >= 0
    except ValueError:
        return False


def color_text(text: str, color_code: int) -> str:
    if sys.stdout.isatty():
        return f"\033[{color_code}m{text}\033[0m"
    return text


def print_header(title: str):
    print_flush(color_text("=" * 60, 34))
    print_flush(color_text(f" {title}", 34))
    print_flush(color_text("=" * 60, 34))


def print_success(msg: str):
    print_success_msg = color_text(f" {msg}", 32)
    print_flush(print_success_msg)


def print_error(msg: str):
    print_error_msg = color_text(f" {msg}", 31)
    print_flush(print_error_msg)


def print_info(msg: str):
    print_info_msg = color_text(f"ℹ {msg}", 33)
    print_flush(print_info_msg)


def _import_module(module_name: str):
    try:
        package = __package__
        if package:
            return importlib.import_module(f".{module_name}", package=package)
    except Exception:
        pass
    try:
        return __import__(module_name, fromlist=['*'])
    except Exception as e:
        print_error(f"Error loading module {module_name}: {e}")
        sys.exit(1)


def compare_command(args):
    prep = _import_module('preprocessing')
    mh = _import_module('minhash')
    sh = _import_module('simhash')

    print_header("Compare two documents")
    print_flush(f"File A: {args.file_a}")
    print_flush(f"File B: {args.file_b}")

    try:
        with open(args.file_a, 'r', encoding='utf-8', errors='ignore') as f:
            text_a = f.read()
        with open(args.file_b, 'r', encoding='utf-8', errors='ignore') as f:
            text_b = f.read()
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        sys.exit(1)

    shingle_size = 3
    num_perm = 128

    tokens_a = prep.preprocess_text(text_a)
    tokens_b = prep.preprocess_text(text_b)

    if not tokens_a or not tokens_b:
        print_error("One of the documents is empty after preprocessing.")
        sys.exit(1)

    shingles_a = prep.shingle_preprocess(tokens_a, k=shingle_size)
    shingles_b = prep.shingle_preprocess(tokens_b, k=shingle_size)

    inter = len(shingles_a & shingles_b)
    union = len(shingles_a | shingles_b)
    jaccard = inter / union if union > 0 else 0.0

    minhash = mh.MinHash(mh.MinHashConfig(num_perm=num_perm))
    sig_a = minhash.signature(shingles_a)
    sig_b = minhash.signature(shingles_b)
    minhash_sim = minhash.similarity(sig_a, sig_b)

    simhash = sh.SimHash(sh.SimHashConfig(hash_bits=64))
    idf = simhash.inverse_document_frequency([tokens_a, tokens_b])
    hash_a = simhash.signature(tokens_a, idf)
    hash_b = simhash.signature(tokens_b, idf)
    simhash_sim = simhash.similarity(hash_a, hash_b)

    output_path = Path(args.output)
    base_dir = output_path.parent
    processed_dir = base_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    name_a = args.file_a.stem
    name_b = args.file_b.stem

    tokens_file_a = processed_dir / f"{name_a}_tokens.txt"
    tokens_file_b = processed_dir / f"{name_b}_tokens.txt"
    shingles_file_a = processed_dir / f"{name_a}_shingles.txt"
    shingles_file_b = processed_dir / f"{name_b}_shingles.txt"

    with open(tokens_file_a, "w", encoding="utf-8") as f:
        f.write(" ".join(tokens_a))
    with open(tokens_file_b, "w", encoding="utf-8") as f:
        f.write(" ".join(tokens_b))
    with open(shingles_file_a, "w", encoding="utf-8") as f:
        f.write("\n".join(shingles_a))
    with open(shingles_file_b, "w", encoding="utf-8") as f:
        f.write("\n".join(shingles_b))

    result = {
        "file_a": str(args.file_a),
        "file_b": str(args.file_b),
        "tokens_file_a": str(tokens_file_a),
        "tokens_file_b": str(tokens_file_b),
        "shingles_file_a": str(shingles_file_a),
        "shingles_file_b": str(shingles_file_b),
        "shingle_size": shingle_size,
        "jaccard_similarity": jaccard,
        "minhash_similarity": minhash_sim,
        "simhash_similarity": simhash_sim,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print_success(f"Results saved to {output_path}")
    print_flush(f"Jaccard similarity : {jaccard:.4f}")
    print_flush(f"MinHash similarity : {minhash_sim:.4f}")
    print_flush(f"SimHash similarity : {simhash_sim:.4f}")
    print_flush(color_text("=" * 60, 34))


def corpus_command(args):

    retrieval = _import_module('retrieval')
    prep = _import_module('preprocessing')
    mh = _import_module('minhash')

    print_header("Search for similar documents in corpus")

    data_root = Path(args.data)

    source_dir = data_root / "source"
    suspicious_dir = data_root / "suspicious"

    if not source_dir.exists():
        print_error(f"Missing source folder: {source_dir}")
        sys.exit(1)

    if not suspicious_dir.exists():
        print_error(f"Missing suspicious folder: {suspicious_dir}")
        sys.exit(1)

    print_info(f"Reading source documents from {source_dir}")
    print_info(f"Reading suspicious documents from {suspicious_dir}")

    shingle_size = args.shingle_size
    threshold = args.threshold

    source_documents = {}
    suspicious_documents = {}

    for file in source_dir.glob("*.txt"):

        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        tokens = prep.preprocess_text(text)

        shingles = prep.shingle_preprocess(
            tokens,
            k=shingle_size
        )

        source_documents[file.name] = shingles

    for file in suspicious_dir.glob("*.txt"):

        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        tokens = prep.preprocess_text(text)

        shingles = prep.shingle_preprocess(
            tokens,
            k=shingle_size
        )

        suspicious_documents[file.name] = shingles

    print_success(
        f"Processed {len(source_documents)} source documents"
    )

    print_success(
        f"Processed {len(suspicious_documents)} suspicious documents"
    )

    num_perm = 128
    bands = 64
    rows = 2

    lsh_candidates = {}

    minhash = mh.MinHash(
        mh.MinHashConfig(
            num_perm=num_perm
        )
    )

    source_signatures = {}

    for name, shingles in source_documents.items():

        source_signatures[name] = minhash.signature(shingles)

    for susp_name, susp_shingles in suspicious_documents.items():

        susp_sig = minhash.signature(
            susp_shingles
        )

        candidates = []

        for src_name, src_sig in source_signatures.items():

            similarity = minhash.similarity(
                susp_sig,
                src_sig
            )

            if similarity >= threshold:

                candidates.append(
                    (
                        src_name,
                        similarity
                    )
                )

        lsh_candidates[susp_name] = candidates

    output_path = Path(args.output)

    if output_path.suffix == "":
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "lsh_candidates.csv"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    retrieval.save_candidates(
        lsh_candidates,
        output_path
    )

    print_success(
        f"Candidates saved to {output_path}"
    )


def evaluate_command(args):
    print_header("Evaluate retrieval on PAN corpus")
    print_info(f"Data root: {args.data}")
    print_info(f"Candidates: {args.candidates}")
    print_info(f"Top K: {args.top_k}")
    print_info(f"Output : {args.output}")

    ground_truth = load_ground_truth_from_dataset(args.data)
    if not ground_truth:
        print_error("No ground truth found. Check your data root.")
        sys.exit(1)

    print_success(
        f"Loaded ground truth for {len(ground_truth)} suspicious documents.")

    json_dir = Path(args.data) / "processed" / "suspicious"

    if not json_dir.exists():
        print_error(f"JSON directory not found: {json_dir}")
        sys.exit(1)

    all_suspicious_ids = []

    for file in json_dir.glob("*.json"):
        all_suspicious_ids.append(file.stem)

    print_success(
        f"Found {len(all_suspicious_ids)} suspicious JSON documents."
    )

    if not all_suspicious_ids:
        print_error("No suspicious .txt files found.")
        sys.exit(1)

    print_success(f"Found {len(all_suspicious_ids)} suspicious documents.")

    print(all_suspicious_ids[:5])

    evaluator = Evaluator(ground_truth, all_suspicious_ids)

    candidates_path = args.candidates
    csv_files = []
    if candidates_path.is_file() and candidates_path.suffix == ".csv":
        csv_files = [candidates_path]
    elif candidates_path.is_dir():
        csv_files = list(candidates_path.glob("*.csv"))
    else:
        print_error(
            f"Invalid candidates path: {candidates_path} (must be .csv or a directory)")
        sys.exit(1)

    if not csv_files:
        print_error("No CSV files found.")
        sys.exit(1)

    for csv_file in csv_files:
        print("\n" + "-" * 40)
        method_name = csv_file.stem
        print(f"Evaluating {method_name}")
        print("-" * 40)

        metrics = evaluator.evaluate_candidates(csv_file, top_k=args.top_k)

        args.output.mkdir(parents=True, exist_ok=True)

        metrics_file = args.output / f"{method_name}_metrics.csv"

        if metrics_file.exists():
            print_info(f"Updating existing metrics file: {metrics_file.name}")
        else:
            print_info(f"Creating metrics file: {metrics_file.name}")

        save_metrics(metrics, metrics_file)

        print_success(f"Metrics saved to {metrics_file}")

        print_success("Evaluation complete.")


def interactive_compare():
    print_header("Compare two documents")
    file_a = input_with_default(
        "Path to first file", default="data/sample_corpus/doc_01.txt", validator=validate_file_exists)
    file_b = input_with_default(
        "Path to second file", default="data/sample_corpus/doc_02.txt", validator=validate_file_exists)
    output = input_with_default(
        "Output JSON file path", default="outputs/two_file_compare.json")

    args = argparse.Namespace(
        file_a=Path(file_a),
        file_b=Path(file_b),
        output=Path(output)
    )
    compare_command(args)


def interactive_corpus():
    print_header("Search for similar documents in corpus")
    data_root = input_with_default(
        "Corpus root path", default="data/sample_corpus")
    shingle_size = int(input_with_default(
        "Shingle size", default=3, validator=lambda x: x.isdigit() and int(x) > 0))
    threshold = float(input_with_default("Similarity threshold",
                      default=0.25, validator=validate_positive_number))
    output = input_with_default(
        "Output file or directory path", default="outputs/candidates.csv")

    args = argparse.Namespace(
        data=Path(data_root),
        shingle_size=shingle_size,
        threshold=threshold,
        output=Path(output)
    )
    corpus_command(args)


def interactive_evaluate_pan():
    print_header("Evaluate retrieval on PAN corpus")
    data_root = input_with_default(
        "PAN corpus root path",
        default="data/sample_corpus"
    )
    candidates = input_with_default(
        "Candidates CSV file or directory",
        default="outputs/candidates.csv"
    )
    top_k = int(input_with_default(
        "Top K for evaluation",
        default="10",
        validator=lambda x: x.isdigit() and int(x) > 0
    ))
    output_dir = input_with_default(
        "Output directory for metrics",
        default="outputs/cli/metrics"
    )

    args = argparse.Namespace(
        data=Path(data_root),
        candidates=Path(candidates),
        top_k=top_k,
        output=Path(output_dir)
    )
    evaluate_command(args)


def interactive_main():
    print_flush("\n Entered interactive mode.")
    while True:
        print_header("Semantic Plagiarism Engine")
        print_flush(color_text("  1. Compare two files", 33))
        print_flush(color_text(
            "  2. Search for similar documents in corpus", 33))
        print_flush(color_text("  3. Evaluate corpus", 33))
        print_flush(color_text("  4. Exit", 31))
        choice = input_with_default(
            "Choose", default="1", validator=lambda x: x in ("1", "2", "3", "4"))

        if choice == "1":
            interactive_compare()
        elif choice == "2":
            interactive_corpus()
        elif choice == "3":
            interactive_evaluate_pan()
        elif choice == "4":
            print_success("Exiting.")
            break

        cont = input_with_default(
            "Perform another operation? (yes/no)", default="no")
        if cont.lower() not in ("y", "yes"):
            print_success("Exiting.")
            break


def main():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            prog="plagiarism_engine",
            description="Semantic Plagiarism Detection Engine"
        )
        subparsers = parser.add_subparsers(dest="command", required=True)

        compare = subparsers.add_parser(
            "compare", help="Compare two documents")
        compare.add_argument("--file-a", required=True,
                             type=Path, help="Path to first file")
        compare.add_argument("--file-b", required=True,
                             type=Path, help="Path to second file")
        compare.add_argument("--output", type=Path,
                             required=True, help="Output JSON file")
        compare.set_defaults(func=compare_command)

        corpus = subparsers.add_parser(
            "corpus", help="Search for similar documents in corpus (LSH only)")
        corpus.add_argument("--data", required=True, type=Path,
                            help="Corpus root path (must contain source/ and suspicious/)")
        corpus.add_argument("--shingle-size", type=int, required=True,
                            help="Shingle size (word n‑gram length)")
        corpus.add_argument("--threshold", type=float, required=True,
                            help="Similarity threshold for filtering candidates (e.g., 0.25)")
        corpus.add_argument("--output", type=Path, required=True,
                            help="Output file or directory for candidates CSV")
        corpus.set_defaults(func=corpus_command)

        evaluate_parser = subparsers.add_parser(
            "evaluate",
            help="Evaluate candidate pairs against PAN ground truth"
        )
        evaluate_parser.add_argument("--data", required=True, type=Path,
                                     help="PAN corpus root (with source/, suspicious/, and XMLs)")
        evaluate_parser.add_argument("--candidates", required=True, type=Path,
                                     help="Path to candidates CSV file or a directory containing CSV files")
        evaluate_parser.add_argument("--top-k", type=int, default=10,
                                     help="Number of top candidates to consider (default: 10)")
        evaluate_parser.add_argument("--output", type=Path, required=True,
                                     help="Directory where metrics CSV files will be saved")
        evaluate_parser.set_defaults(func=evaluate_command)

        try:
            args = parser.parse_args()
            args.func(args)
        except Exception as e:
            print_error(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        try:
            interactive_main()
        except KeyboardInterrupt:
            print_flush("\n Program interrupted with Ctrl+C.")
            sys.exit(0)
        except Exception as e:
            print_error(f"Error in interactive mode: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()

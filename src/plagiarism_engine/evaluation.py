from typing import Dict, Set, List, Tuple
from pathlib import Path
import sys
import csv
from dataset import Dataset
src_path = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_path))


class Evaluator:
    def __init__(self, ground_truth: Dict[str, Set[str]], all_suspicious_ids: List[str]):
        self.ground_truth = ground_truth
        self.all_suspicious_ids = all_suspicious_ids

    def evaluate_candidates(
        self,
        candidate_file: Path,
        top_k: int = 10
    ) -> Dict[str, float]:
        candidates_by_query: Dict[str, List[Tuple[str, float]]] = {}
        with open(candidate_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 3:
                    continue
                susp_id, src_id, score_str = row[0], row[1], row[2]
                try:
                    score = float(score_str)
                except ValueError:
                    score = 0.0

                if susp_id not in candidates_by_query:
                    candidates_by_query[susp_id] = []
                candidates_by_query[susp_id].append((src_id, score))

        for susp_id in candidates_by_query:
            candidates_by_query[susp_id].sort(key=lambda x: x[1], reverse=True)

        total_queries = 0
        ap_sum = 0.0
        rr_sum = 0.0
        hit_count = 0

        tp_sum = 0
        fp_sum = 0
        fn_sum = 0

        precision_at_k_sum = 0.0
        recall_at_k_sum = 0.0

        for susp_id in self.all_suspicious_ids:
            true_sources = self.ground_truth.get(susp_id, set())
            candidates = candidates_by_query.get(susp_id, [])[:top_k]
            predicted_ids = [src_id for src_id, _ in candidates]
            predicted_set = set(predicted_ids)

            tp = len(predicted_set & true_sources)
            fp = len(predicted_set - true_sources)
            fn = len(true_sources - predicted_set)

            tp_sum += tp
            fp_sum += fp
            fn_sum += fn

            if true_sources:
                total_queries += 1

                precision_k = tp / len(predicted_ids) if predicted_ids else 0.0
                recall_k = tp / len(true_sources) if true_sources else 0.0
                precision_at_k_sum += precision_k
                recall_at_k_sum += recall_k

                ap = 0.0
                relevant_found = 0
                for rank, (src_id, _) in enumerate(candidates, start=1):
                    if src_id in true_sources:
                        relevant_found += 1
                        ap += relevant_found / rank
                ap = ap / len(true_sources) if true_sources else 0.0
                ap_sum += ap

                rr = 0.0
                for rank, (src_id, _) in enumerate(candidates, start=1):
                    if src_id in true_sources:
                        rr = 1.0 / rank
                        break
                rr_sum += rr

                if predicted_set & true_sources:
                    hit_count += 1

        precision = tp_sum / \
            (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0.0

        map_score = ap_sum / total_queries if total_queries > 0 else 0.0
        mrr = rr_sum / total_queries if total_queries > 0 else 0.0

        precision_at_k = precision_at_k_sum / total_queries if total_queries > 0 else 0.0
        recall_at_k = recall_at_k_sum / total_queries if total_queries > 0 else 0.0
        hit_at_k = hit_count / total_queries if total_queries > 0 else 0.0

        return {
            "queries": total_queries,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "map": map_score,
            "mrr": mrr,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "hit_at_k": hit_at_k,
            "tp": tp_sum,
            "fp": fp_sum,
            "fn": fn_sum,
        }


def get_all_suspicious_ids(processed_dir: Path) -> List[str]:
    suspicious_dir = processed_dir / "suspicious"
    if not suspicious_dir.exists():
        return []
    return [f.stem for f in suspicious_dir.glob("*.json")]


def load_ground_truth_from_dataset(data_root: Path) -> Dict[str, Set[str]]:
    dataset = Dataset(str(data_root))
    dataset.load()

    cleaned_gt = {}
    for key, sources in dataset.ground_truth.items():
        clean_key = key.replace(".txt", "")
        cleaned_gt[clean_key] = sources

    return cleaned_gt


def save_metrics(metrics: Dict[str, float], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def print_metrics(metrics: Dict[str, float], top_k: int = 10) -> None:
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Queries          : {metrics['queries']}")
    print(f"Precision        : {metrics['precision']:.4f}")
    print(f"Recall           : {metrics['recall']:.4f}")
    print(f"F1               : {metrics['f1']:.4f}")
    print(f"MAP              : {metrics['map']:.4f}")
    print(f"MRR              : {metrics['mrr']:.4f}")
    print(f"Precision@{top_k}: {metrics['precision_at_k']:.4f}")
    print(f"Recall@{top_k}   : {metrics['recall_at_k']:.4f}")
    print(f"Hit@{top_k}      : {metrics['hit_at_k']:.4f}")
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")


def main():
    data_root = Path(
        "/Users/maryam/Documents/semantic-plagiarism-engine/data/raw/pan-plagiarism-corpus-2011/pan-plagiarism-corpus-2011"
    )
    processed_dir = Path(
        "/Users/maryam/Documents/semantic-plagiarism-engine/data/processed"
    )
    candidates_dir = Path(
        "/Users/maryam/Documents/semantic-plagiarism-engine/outputs/candidates"
    )
    metrics_dir = Path(
        "/Users/maryam/Documents/semantic-plagiarism-engine/outputs/metrics"
    )
    top_k = 10

    print("=" * 60)
    print("Running Evaluation with Debug")
    print("=" * 60)

    print("\n📂 Loading ground truth...")
    ground_truth = load_ground_truth_from_dataset(data_root)
    print(f" Ground truth loaded: {len(ground_truth)} documents.")

    all_suspicious_ids = get_all_suspicious_ids(processed_dir)
    print(f"📂 Found {len(all_suspicious_ids)} suspicious JSON files.")

    common = set(all_suspicious_ids) & set(ground_truth.keys())
    print(f"🔍 Number of common IDs after cleaning: {len(common)}")
    if common:
        print(f"   Example common ID: {next(iter(common))}")
    else:
        print("    Still no common IDs. Please check your data paths.")

    evaluator = Evaluator(ground_truth, all_suspicious_ids)

    csv_files = list(candidates_dir.glob("*.csv"))
    if not csv_files:
        print(" No candidate CSV files found.")
        return

    for csv_file in csv_files:
        print("\n" + "-" * 40)
        method_name = csv_file.stem
        print(f" Evaluating {method_name}")
        print("-" * 40)

        metrics = evaluator.evaluate_candidates(csv_file, top_k=top_k)
        print_metrics(metrics, top_k)

        metrics_file = metrics_dir / f"{method_name}_metrics.csv"
        save_metrics(metrics, metrics_file)
        print(f" Metrics saved to: {metrics_file}")

    print("\n" + "=" * 60)
    print(" Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

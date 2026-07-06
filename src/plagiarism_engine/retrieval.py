from simhash import SimHash, SimHashConfig
from lsh import LSH, LSHConfig
from minhash import MinHash, MinHashConfig
from preprocessing import shingle_preprocess
from typing import List, Tuple, Dict, Set
import csv
import json
import sys
from pathlib import Path
src_path = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_path))


def load_tokens(processed_dir: Path, doc_id: str, is_source: bool = True):
    if is_source:
        file_path = processed_dir / "source" / f"{doc_id}.json"
    else:
        file_path = processed_dir / "suspicious" / f"{doc_id}.json"

    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("tokens", [])


def get_all_ids(processed_dir: Path, doc_type: str = "source") -> List[str]:
    dir_path = processed_dir / doc_type
    return [f.stem for f in dir_path.glob("*.json")]


def save_candidates(candidates: Dict[str, List[Tuple[str, float]]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["suspicious", "source", "score"])
        for susp_id, results in candidates.items():
            for src_id, score in results:
                writer.writerow([susp_id, src_id, f"{score:.6f}"])


def retrieve_lsh(
    processed_dir: Path,
    output_file: Path,
    shingle_size: int = 3,
    num_perm: int = 128,
    bands: int = 64,
    rows: int = 2,
    top_k: int = 10
) -> Dict[str, List[Tuple[str, float]]]:

    source_ids = get_all_ids(processed_dir, "source")
    print(f" Source documents: {len(source_ids)}")

    source_shingles: Dict[str, Set[str]] = {}
    source_signatures: Dict[str, List[int]] = {}

    minhash = MinHash(MinHashConfig(num_perm=num_perm))
    lsh = LSH(LSHConfig(signature_size=num_perm, bands=bands, rows=rows))

    print(" Building LSH index...")
    for idx, doc_id in enumerate(source_ids, 1):
        tokens = load_tokens(processed_dir, doc_id, is_source=True)
        if not tokens:
            continue

        shingles = shingle_preprocess(tokens, k=shingle_size)
        if not shingles:
            continue

        source_shingles[doc_id] = shingles
        sig = minhash.signature(shingles)
        source_signatures[doc_id] = sig
        lsh.index(doc_id, sig)

        if idx % 500 == 0 or idx == len(source_ids):
            print(f"  Indexed {idx}/{len(source_ids)} source documents.")

    print(f" LSH index built: {len(source_shingles)} documents.")

    suspicious_ids = get_all_ids(processed_dir, "suspicious")
    print(f" Suspicious documents: {len(suspicious_ids)}")

    results: Dict[str, List[Tuple[str, float]]] = {}

    for idx, susp_id in enumerate(suspicious_ids, 1):
        tokens = load_tokens(processed_dir, susp_id, is_source=False)
        if not tokens:
            continue

        shingles = shingle_preprocess(tokens, k=shingle_size)
        if not shingles:
            continue

        sig = minhash.signature(shingles)
        candidates = lsh.query(sig)

        if candidates:
            scores = []
            for src_id in candidates:
                if src_id in source_shingles:
                    intersect = len(shingles & source_shingles[src_id])
                    union = len(shingles | source_shingles[src_id])
                    sim = intersect / union if union > 0 else 0.0
                    scores.append((src_id, sim))
            scores.sort(key=lambda x: x[1], reverse=True)
            results[susp_id] = scores[:top_k]
        else:
            results[susp_id] = []

        if idx % 100 == 0 or idx == len(suspicious_ids):
            print(
                f"  Processed {idx}/{len(suspicious_ids)} suspicious documents.")

    save_candidates(results, output_file)
    print(f" LSH candidates saved to: {output_file}")
    print(f"   Total queries: {len(results)}")
    print(f"   Candidates found: {sum(len(v) for v in results.values())}")

    return results


def retrieve_simhash(
    processed_dir: Path,
    output_file: Path,
    top_k: int = 10
) -> Dict[str, List[Tuple[str, float]]]:

    simhash = SimHash(SimHashConfig(hash_bits=64))

    source_ids = get_all_ids(processed_dir, "source")
    suspicious_ids = get_all_ids(processed_dir, "suspicious")
    print(f" Source documents: {len(source_ids)}")
    print(f" Suspicious documents: {len(suspicious_ids)}")

    source_tokens: Dict[str, List[str]] = {}
    all_tokens: List[List[str]] = []

    for doc_id in source_ids:
        tokens = load_tokens(processed_dir, doc_id, is_source=True)
        if tokens:
            source_tokens[doc_id] = tokens
            all_tokens.append(tokens)

    suspicious_tokens_for_idf: Dict[str, List[str]] = {}
    for doc_id in suspicious_ids:
        tokens = load_tokens(processed_dir, doc_id, is_source=False)
        if tokens:
            suspicious_tokens_for_idf[doc_id] = tokens
            all_tokens.append(tokens)

    print(f" Total documents for IDF: {len(all_tokens)}")

    print(" Computing IDF from source + suspicious...")
    idf = simhash.inverse_document_frequency(all_tokens)

    print(" Computing fingerprints for source documents...")
    source_hashes: Dict[str, int] = {}
    for idx, (doc_id, tokens) in enumerate(source_tokens.items(), 1):
        fingerprint = simhash.signature(tokens, idf)
        source_hashes[doc_id] = fingerprint
        if idx % 500 == 0 or idx == len(source_tokens):
            print(f"  Processed {idx}/{len(source_tokens)} source documents.")

    print(f" SimHash index built: {len(source_hashes)} documents.")

    print(" Querying suspicious documents...")
    results: Dict[str, List[Tuple[str, float]]] = {}

    for idx, susp_id in enumerate(suspicious_ids, 1):
        tokens = load_tokens(processed_dir, susp_id, is_source=False)
        if not tokens:
            results[susp_id] = []
            continue

        susp_hash = simhash.signature(tokens, idf)

        scores = []
        for src_id, src_hash in source_hashes.items():
            sim = simhash.similarity(susp_hash, src_hash)
            scores.append((src_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        results[susp_id] = scores[:top_k]

        if idx % 100 == 0 or idx == len(suspicious_ids):
            print(
                f"  Processed {idx}/{len(suspicious_ids)} suspicious documents.")

    save_candidates(results, output_file)
    print(f"   SimHash candidates saved to: {output_file}")
    print(f"   Total queries: {len(results)}")
    print(f"   Candidates found: {sum(len(v) for v in results.values())}")

    return results


def main():
    processed_dir = Path(
        "/Users/maryam/Documents/semantic-plagiarism-engine/data/processed")
    output_dir = Path(
        "/Users/maryam/Documents/semantic-plagiarism-engine/outputs/candidates")

    lsh_output = output_dir / "lsh_candidates.csv"
    retrieve_lsh(processed_dir, lsh_output)

    print("\n" + "=" * 60 + "\n")

    sim_output = output_dir / "simhash_candidates.csv"
    retrieve_simhash(processed_dir, sim_output)

    print("\n" + "=" * 60)
    print(" All retrieval engines completed.")
    print(f" Candidates saved in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

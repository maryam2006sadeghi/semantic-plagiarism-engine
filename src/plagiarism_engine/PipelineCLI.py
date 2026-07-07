import json
from pathlib import Path
from typing import Dict

from preprocessing import preprocess_text


class PreprocessingPipeline:

    def __init__(
        self,
        corpus_root: str | Path,
        output_root: str | Path = "data/sample_corpus/processed",
        overwrite: bool = False,
    ):

        self.corpus_root = Path(corpus_root)

        self.source_dir = self.corpus_root / "source"
        self.suspicious_dir = self.corpus_root / "suspicious"

        if not self.source_dir.exists():
            raise FileNotFoundError(
                f"Source directory not found:\n{self.source_dir}"
            )

        if not self.suspicious_dir.exists():
            raise FileNotFoundError(
                f"Suspicious directory not found:\n{self.suspicious_dir}"
            )

        self.output_root = Path(output_root)

        self.output_source = self.output_root / "source"
        self.output_suspicious = self.output_root / "suspicious"

        self.overwrite = overwrite

    def run(self):

        self.output_source.mkdir(parents=True, exist_ok=True)
        self.output_suspicious.mkdir(parents=True, exist_ok=True)

        print("Processing source documents...")
        self._process_directory(
            self.source_dir,
            self.output_source
        )

        print("Processing suspicious documents...")
        self._process_directory(
            self.suspicious_dir,
            self.output_suspicious
        )

        print("Finished preprocessing.")

    def _process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
    ):

        files = sorted(input_dir.glob("*.txt"))

        for i, file in enumerate(files, start=1):

            output_file = output_dir / f"{file.stem}.json"

            if output_file.exists() and not self.overwrite:
                continue

            text = file.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            tokens = preprocess_text(text)

            result = {
                "document_id": file.stem,
                "document_name": file.name,
                "token_count": len(tokens),
                "tokens": tokens
            }

            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(
                    result,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            print(
                f"[{i}/{len(files)}] {file.name}"
            )


if __name__ == "__main__":

    pipeline = PreprocessingPipeline(
        corpus_root="data/sample_corpus",
        output_root="data/sample_corpus/processed",
        overwrite=True
    )

    pipeline.run()

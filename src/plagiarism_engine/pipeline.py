import inspect
import json
from typing import Optional, List
from preprocessing import preprocess_text
from dataset import Dataset
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


class Pipeline:
    def __init__(
        self,
        dataset: Dataset,
        output_directory: str = "/Users/maryam/Documents/semantic-plagiarism-engine/data/processed"
    ):
        self.dataset = dataset

        self.output_root = Path(output_directory)

        self.source_output = self.output_root / "source"
        self.suspicious_output = self.output_root / "suspicious"

        self.source_output.mkdir(parents=True, exist_ok=True)
        self.suspicious_output.mkdir(parents=True, exist_ok=True)

    def build_cache(self, overwrite: bool = False) -> None:
        self._process_documents(
            self.dataset.source_docs,
            self.source_output,
            overwrite
        )

        self._process_documents(
            self.dataset.suspicious_docs,
            self.suspicious_output,
            overwrite
        )

    def _process_documents(
        self,
        documents: dict,
        output_directory: Path,
        overwrite: bool
    ) -> None:

        total = len(documents)

        for index, (document_name, document_data) in enumerate(
            documents.items(), start=1
        ):
            doc_id = Path(document_name).stem
            output_file = output_directory / f"{doc_id}.json"

            if output_file.exists() and not overwrite:
                continue

            raw_text = document_data["text"]

            tokens = preprocess_text(raw_text)

            json_data = {
                "id": doc_id,
                "tokens": tokens
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)


dataset = Dataset(
    "/Users/maryam/Documents/semantic-plagiarism-engine/data/raw/pan-plagiarism-corpus-2011/pan-plagiarism-corpus-2011"
)
dataset.load()

pipeline = Pipeline(
    dataset=dataset,
    output_directory="/Users/maryam/Documents/semantic-plagiarism-engine/data/processed"
)

pipeline.build_cache(overwrite=True)

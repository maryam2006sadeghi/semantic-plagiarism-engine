from pathlib import Path
import xml.etree.ElementTree as ET


class Dataset:
    def __init__(self, root_directory):
        self.root = Path(root_directory)

        self.source_root = self.root / "external-detection-corpus" / "source-document"
        self.suspicious_root = self.root / \
            "external-detection-corpus" / "suspicious-document"

        self.source_docs = {}
        self.suspicious_docs = {}
        self.ground_truth = {}

    def load(self):
        self._load_source_documents()
        self._load_suspicious_documents()
        self._load_ground_truth()

    def _load_source_documents(self):
        if not self.source_root.exists():
            raise FileNotFoundError(
                f"Source directory not found at: {self.source_root}")

        for txt_file in self.source_root.rglob("*.txt"):
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                self.source_docs[txt_file.name] = {
                    "text": f.read(),
                    "path": txt_file
                }

    def _load_suspicious_documents(self):
        if not self.suspicious_root.exists():
            raise FileNotFoundError(
                f"Suspicious directory not found at: {self.suspicious_root}")

        for txt_file in self.suspicious_root.rglob("*.txt"):
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                self.suspicious_docs[txt_file.name] = {
                    "text": f.read(),
                    "path": txt_file
                }

    def _load_ground_truth(self):
        for xml_file in self.suspicious_root.rglob("*.xml"):

            try:

                tree = ET.parse(xml_file)
                root = tree.getroot()

            except ET.ParseError:
                continue

            suspicious_name = xml_file.stem
            sources = set()

            for feature in root.iter("feature"):

                if feature.get("name") != "plagiarism":
                    continue

                source = feature.get("source_reference")

                if source is not None:
                    sources.add(
                        source.replace(".txt", "")
                    )
            if sources:
                self.ground_truth[suspicious_name] = sources

    def get_source_text(self, filename):
        return self.source_docs[filename]["text"]

    def get_suspicious_text(self, filename):
        return self.suspicious_docs[filename]["text"]

    def get_ground_truth(self, suspicious_filename):
        return self.ground_truth.get(suspicious_filename, set())

    def source_documents(self):
        return self.source_docs.items()

    def suspicious_documents(self):
        return self.suspicious_docs.items()

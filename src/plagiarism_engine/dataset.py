from pathlib import Path
import xml.etree.ElementTree as ET


class Dataset:
    def init(self, root_directory):
        self.root = Path(root_directory)

        self.source_docs = {}
        self.suspicious_docs = {}
        self.ground_truth = {}

    def load(self):
        self._load_source_documents()
        self._load_suspicious_documents()
        self._load_ground_truth()

    def _load_source_documents(self):
        source_root = (
            self.root /
            "/Users/maryam/Documents/semantic-plagiarism-engine/data/raw/pan-plagiarism-corpus-2011/pan-plagiarism-corpus-2011/external-detection-corpus/source-document"
        )

        for txt_file in source_root.rglob("*.txt"):
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                self.source_docs[txt_file.name] = {
                    "text": f.read(),
                    "path": txt_file
                }

    def _load_suspicious_documents(self):
        suspicious_root = (
            self.root /
            "/Users/maryam/Documents/semantic-plagiarism-engine/data/raw/pan-plagiarism-corpus-2011/pan-plagiarism-corpus-2011/external-detection-corpus/suspicious-document"
        )

        for txt_file in suspicious_root.rglob("*.txt"):
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                self.suspicious_docs[txt_file.name] = {
                    "text": f.read(),
                    "path": txt_file
                }

    def _load_ground_truth(self):
        suspicious_root = (
            self.root /
            "/Users/maryam/Documents/semantic-plagiarism-engine/data/raw/pan-plagiarism-corpus-2011/pan-plagiarism-corpus-2011/external-detection-corpus/suspicious-document"
        )

        for xml_file in suspicious_root.rglob("*.xml"):

            tree = ET.parse(xml_file)
            root = tree.getroot()

            suspicious_name = root.attrib["reference"]

            sources = set()

            for feature in root.findall("feature"):

                if feature.attrib.get("name") != "plagiarism":
                    continue

                source = feature.attrib["source_reference"]
                sources.add(source)

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

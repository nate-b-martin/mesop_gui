from langchain.document_loaders import CSVLoader
from typing import List, Dict, Any
import csv

class DynamicCSVLoader(CSVLoader):
    def __init__(self, file_path: str, csv_args: Dict[str, Any] = None, encoding: str = "utf-8"):
        super().__init__(file_path, csv_args, encoding)
        self.columns = self._get_columns()

    def _get_columns(self) -> List[str]:
        with open(self.file_path, 'r', encoding=self.encoding) as f:
            csv_reader = csv.reader(f)
            return next(csv_reader, [])

    def lazy_load(self) -> List[Dict[str, Any]]:
        with open(self.file_path, newline='', encoding=self.encoding) as csvfile:
            csv_reader = csv.DictReader(csvfile, fieldnames=self.columns)
            return [row for row in csv_reader]

def csv_loader_factory(file_path: str) -> DynamicCSVLoader:
    return DynamicCSVLoader(file_path)
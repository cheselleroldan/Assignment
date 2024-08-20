import os
import re
from typing import List, Tuple



class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        os.path.join(root, file), "r", encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks
    


class SentenceTextSplitter:
    def __init__(
        self,
        chunk_size: int = 10,
        chunk_overlap: int = 1,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        # Split the text into sentences using a regex that matches sentence end punctuation.
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        
        for i in range(0, len(sentences), self.chunk_size - self.chunk_overlap):
            chunk = sentences[i : i + self.chunk_size]
            chunks.append(' '.join(chunk))
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks
    
class PageTextSplitter:
    def __init__(self, page_marker: str = "The Pmarca Blog Archives"):
        self.page_marker = page_marker

    def split(self, text: str) -> List[Tuple[int, str]]:
        # Use a regex pattern to capture the page number and the text on each page
        pattern = rf"(\d+)\s+{self.page_marker}"
        matches = re.split(pattern, text)

        pages = []
        for i in range(1, len(matches), 3):  # Skipping the text part, capturing page numbers and following texts
            page_number = int(matches[i])
            page_text = matches[i + 1].strip()  # The text after the page number
            pages.append((page_number, page_text))

        return pages


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])

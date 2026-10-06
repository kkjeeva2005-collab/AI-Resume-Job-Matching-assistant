import faiss
import numpy as np


class VectorStore:

    def __init__(self):

        self.index = None
        self.documents = []

    def create_index(self, embeddings):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

    def add_documents(
        self,
        documents,
        embeddings
    ):

        self.documents = documents

        self.create_index(
            embeddings
        )

    def search(
        self,
        query_embedding,
        top_k=5
    ):

        if self.index is None:
            return []

        query_embedding = np.asarray(
            [query_embedding],
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            min(
                top_k,
                len(self.documents)
            )
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            results.append(
                {
                    "text": self.documents[index],
                    "score": float(score)
                }
            )

        return results
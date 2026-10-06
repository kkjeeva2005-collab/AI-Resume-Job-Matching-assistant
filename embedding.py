from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingModel:
    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, normalize_embeddings=True):
        return self.model.encode(texts, normalize_embeddings=normalize_embeddings)

    def __call__(self, texts, normalize_embeddings=True):
        return self.encode(texts, normalize_embeddings=normalize_embeddings)


def create_embeddings(texts):
    model = EmbeddingModel()
    return model.encode(texts, normalize_embeddings=True)


if __name__ == "__main__":
    sentences = [
        "Python developer with machine learning experience",
        "AI engineer skilled in Python and machine learning",
        "Frontend developer using React"
    ]

    vectors = create_embeddings(sentences)

    print("Embedding shape:", vectors.shape)
    print("First embedding:", vectors[0][:5])
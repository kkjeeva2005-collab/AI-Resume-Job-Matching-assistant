from embedding import EmbeddingModel


def test_embedding_model_exists():
    model = EmbeddingModel()
    assert hasattr(model, "encode")
    vectors = model.encode(["hello world", "resume analysis"])
    assert vectors.shape[0] == 2

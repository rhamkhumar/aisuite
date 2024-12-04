from aisuite.framework.embedding import Embedding


class CreateEmbeddingResponse:
    """Used to conform to the response model of OpenAI"""

    def __init__(self):
        self.data = [Embedding()]

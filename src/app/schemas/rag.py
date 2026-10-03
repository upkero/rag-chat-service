from pydantic import BaseModel, ConfigDict, Field

from src.app.contracts.rag.rag_answer import RagAnswer


class AskRequest(BaseModel):
    # Stripped before min_length runs, so a question of spaces is a local 422 instead
    # of a trip to ops-core that comes back as a 502 and an ERROR log.
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        ...,
        min_length=1,
        max_length=2_000,
        description="User question to answer from the knowledge base.",
    )


class SourceSchema(BaseModel):
    document_title: str
    chunk_index: int
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceSchema]

    @classmethod
    def from_contract(cls, answer: RagAnswer) -> "AskResponse":
        # contract -> HTTP schema mapping (the boundary conversion the router owns).
        return cls(
            answer=answer.answer,
            sources=[
                SourceSchema(
                    document_title=chunk.document_title,
                    chunk_index=chunk.chunk_index,
                    score=chunk.score,
                )
                for chunk in answer.sources
            ],
        )

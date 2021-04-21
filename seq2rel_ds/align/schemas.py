from pydantic import BaseModel


class AlignedExample(BaseModel):
    doc_id: str
    text: str
    relations: str
    score: float

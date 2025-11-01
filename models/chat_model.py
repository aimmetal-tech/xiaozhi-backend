from pydantic import BaseModel
from typing import List, Dict, Optional, Union


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    base_url: Optional[str] = None


class ChatChoice(BaseModel):
    index: int
    delta: Dict[str, Union[str, None]]
    finish_reason: Optional[str] = None


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatChoice]
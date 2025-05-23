from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TranscriptToken:
    text: str = ''
    t0: Optional[int] = None
    t1: Optional[int] = None


@dataclass
class TranscriptSegment:
    index: int
    text: str = ''
    tokens: List[TranscriptToken] = None
    t0: Optional[int] = None
    t1: Optional[int] = None

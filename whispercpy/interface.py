from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptSegment:
    index: int
    text: str = ''
    t0: Optional[int] = None
    t1: Optional[int] = None

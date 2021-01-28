from dataclasses import dataclass
from dataclasses import field


@dataclass
class StateStat():
    duration_sec: int = 0
    price_min: int = 10 ** 5
    price_max: int = 0
    price_range: int = 0
    price_return: int = 0
    jma_min: int = 10 ** 5
    jma_max: int = 0
    jma_range: int = 0
    jma_return: int = 0
    tick_count: int = 0
    volume: int = 0
    dollars: int = 0
    tick_imbalance: int = 0
    volume_imbalance: int = 0
    dollar_imbalance: int = 0


# @dataclass
# class StateTrades():
#     date_time: list[int] = field(default_factory=list)
#     price: list[int] = field(default_factory=list)
#     volume: list[int] = field(default_factory=list)
#     side: list[int] = field(default_factory=list)
#     jma: list[int] = field(default_factory=list)

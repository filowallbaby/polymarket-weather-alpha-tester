from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass(slots=True)
class BookSide:
    levels: dict[Decimal, Decimal] = field(default_factory=dict)

    def replace(self, raw_levels: list[dict[str, Any]]) -> None:
        self.levels.clear()
        for level in raw_levels:
            price = Decimal(str(level["price"]))
            size = Decimal(str(level["size"]))
            if size > 0:
                self.levels[price] = size

    def update(self, price: str | float, size: str | float) -> None:
        p = Decimal(str(price))
        s = Decimal(str(size))
        if s <= 0:
            self.levels.pop(p, None)
        else:
            self.levels[p] = s

    def sorted(self, reverse: bool) -> list[tuple[Decimal, Decimal]]:
        return sorted(self.levels.items(), key=lambda item: item[0], reverse=reverse)


@dataclass(slots=True)
class OrderBook:
    asset_id: str
    bids: BookSide = field(default_factory=BookSide)
    asks: BookSide = field(default_factory=BookSide)
    exchange_timestamp_ms: int | None = None
    local_received_wall_ns: int | None = None
    has_snapshot: bool = False

    def reset(self) -> None:
        self.bids.levels.clear()
        self.asks.levels.clear()
        self.exchange_timestamp_ms = None
        self.local_received_wall_ns = None
        self.has_snapshot = False

    def apply_snapshot(self, message: dict[str, Any], received_wall_ns: int) -> None:
        self.bids.replace(message.get("bids") or [])
        self.asks.replace(message.get("asks") or [])
        self.exchange_timestamp_ms = _int_or_none(message.get("timestamp"))
        self.local_received_wall_ns = received_wall_ns
        self.has_snapshot = True

    def apply_change(self, change: dict[str, Any], timestamp: Any, received_wall_ns: int) -> None:
        if not self.has_snapshot:
            raise RuntimeError("Cannot apply a price change before a full book snapshot")
        side = str(change.get("side", "")).upper()
        target = self.bids if side == "BUY" else self.asks if side == "SELL" else None
        if target is None:
            raise ValueError(f"Unknown price_change side: {side!r}")
        target.update(change["price"], change["size"])
        self.exchange_timestamp_ms = _int_or_none(timestamp)
        self.local_received_wall_ns = received_wall_ns

    @property
    def best_bid(self) -> Decimal | None:
        return max(self.bids.levels) if self.bids.levels else None

    @property
    def best_ask(self) -> Decimal | None:
        return min(self.asks.levels) if self.asks.levels else None

    @property
    def mid(self) -> Decimal | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / Decimal("2")

    @property
    def is_crossed(self) -> bool:
        return bool(
            self.best_bid is not None
            and self.best_ask is not None
            and self.best_bid > self.best_ask
        )

    def normalized(self, depth: int = 25) -> dict[str, Any]:
        bids = self.bids.sorted(reverse=True)[:depth]
        asks = self.asks.sorted(reverse=False)[:depth]
        best_bid = self.best_bid
        best_ask = self.best_ask
        mid = self.mid
        return {
            "asset_id": self.asset_id,
            "exchange_timestamp_ms": self.exchange_timestamp_ms,
            "best_bid": str(best_bid) if best_bid is not None else None,
            "best_ask": str(best_ask) if best_ask is not None else None,
            "mid": str(mid) if mid is not None else None,
            "spread": str(best_ask - best_bid)
            if best_ask is not None and best_bid is not None
            else None,
            "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
            "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
            "book_synchronized": self.has_snapshot,
            "book_crossed": self.is_crossed,
        }

    def simulate_buy_notional(self, notional_usd: float) -> dict[str, Any]:
        remaining = Decimal(str(notional_usd))
        cost = Decimal("0")
        shares = Decimal("0")
        fills: list[dict[str, str]] = []
        for price, available_shares in self.asks.sorted(reverse=False):
            if remaining <= 0:
                break
            available_cost = price * available_shares
            take_cost = min(remaining, available_cost)
            take_shares = take_cost / price
            cost += take_cost
            shares += take_shares
            remaining -= take_cost
            fills.append({"price": str(price), "shares": str(take_shares), "cost": str(take_cost)})
        return {
            "requested_notional": str(Decimal(str(notional_usd))),
            "filled_cost": str(cost),
            "filled_shares": str(shares),
            "unfilled_notional": str(remaining),
            "vwap": str(cost / shares) if shares else None,
            "fill_ratio": str(cost / Decimal(str(notional_usd))) if notional_usd else "0",
            "fills": fills,
        }


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

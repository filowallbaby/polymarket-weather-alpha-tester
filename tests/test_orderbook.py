from weather_alpha.orderbook import OrderBook
import pytest


def test_snapshot_update_and_notional_fill() -> None:
    book = OrderBook("asset")
    book.apply_snapshot(
        {
            "timestamp": "1000",
            "bids": [{"price": "0.40", "size": "100"}],
            "asks": [
                {"price": "0.45", "size": "100"},
                {"price": "0.50", "size": "100"},
            ],
        },
        1,
    )
    assert str(book.best_bid) == "0.40"
    assert str(book.best_ask) == "0.45"
    fill = book.simulate_buy_notional(50)
    assert float(fill["filled_cost"]) == 50.0
    assert float(fill["vwap"]) > 0.45
    assert float(fill["fill_ratio"]) == 1.0

    book.apply_change({"side": "SELL", "price": "0.45", "size": "0"}, 1001, 2)
    assert str(book.best_ask) == "0.50"


def test_book_requires_snapshot_and_reset_invalidates_depth() -> None:
    book = OrderBook("asset")
    with pytest.raises(RuntimeError, match="before a full book snapshot"):
        book.apply_change({"side": "SELL", "price": "0.45", "size": "10"}, 1, 1)

    book.apply_snapshot({"bids": [], "asks": [{"price": "0.45", "size": "10"}]}, 2)
    assert book.normalized()["book_synchronized"] is True
    book.reset()
    assert book.normalized()["book_synchronized"] is False
    assert book.best_ask is None


def test_crossed_book_is_detected() -> None:
    book = OrderBook("asset")
    book.apply_snapshot(
        {
            "bids": [{"price": "0.60", "size": "10"}],
            "asks": [{"price": "0.50", "size": "10"}],
        },
        1,
    )

    assert book.is_crossed is True
    assert book.normalized()["book_crossed"] is True

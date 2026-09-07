//! A price-time ordered view of the orders published on the feed.
//!
//! The feed publishes limit orders but never matches them, so this module keeps
//! the book we build up from that stream: one buy side and one sell side per
//! symbol, each held in a sorted tree so the next order to consider is always
//! at the front of the set.

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};

use crate::feed::{AccountId, OrderId, OrderMessage, SYMBOLS, Side};

/// A resting limit order, stripped of the feed metadata the book does not need.
#[derive(Debug, Clone)]
pub struct BookOrder {
    pub id: OrderId,
    pub timestamp: u64,
    pub account: AccountId,
    pub price: f64,
    pub quantity: f64,
}

/// A resting buy order, ordered lowest price first.
#[derive(Debug, Clone)]
pub struct BuyOrder(pub BookOrder);

/// A resting sell order, ordered highest price first.
#[derive(Debug, Clone)]
pub struct SellOrder(pub BookOrder);

impl Ord for BuyOrder {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lowest price first, then oldest first within a price level. `id` is
        // the final tiebreak so two orders that share a price and a millisecond
        // stay distinct instead of the set silently dropping one of them.
        //
        // `total_cmp` rather than `partial_cmp().unwrap()`: it is a total order
        // over every f64, so a NaN price would sort to one end rather than
        // panicking and corrupting the tree's invariants.
        self.0
            .price
            .total_cmp(&other.0.price)
            .then_with(|| self.0.timestamp.cmp(&other.0.timestamp))
            .then_with(|| self.0.id.cmp(&other.0.id))
    }
}

impl Ord for SellOrder {
    fn cmp(&self, other: &Self) -> Ordering {
        // Highest price first (the price comparison is flipped), then oldest
        // first within a price level, then `id` as above.
        other
            .0
            .price
            .total_cmp(&self.0.price)
            .then_with(|| self.0.timestamp.cmp(&other.0.timestamp))
            .then_with(|| self.0.id.cmp(&other.0.id))
    }
}

// `PartialOrd`/`PartialEq` are defined in terms of `Ord` so the three agree,
// which `BTreeSet` relies on to place and find entries correctly.
impl PartialOrd for BuyOrder {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialOrd for SellOrder {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for BuyOrder {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl PartialEq for SellOrder {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for BuyOrder {}
impl Eq for SellOrder {}

/// The two sides of a single symbol's book.
///
/// Both sets iterate front to back in the priority order defined above, so
/// `buys.first()` is the lowest bid and `sells.first()` is the highest ask.
#[derive(Debug, Default)]
pub struct Book {
    pub buys: BTreeSet<BuyOrder>,
    pub sells: BTreeSet<SellOrder>,
}

/// Holds one [`Book`] per trading pair, built from the messages on the feed.
#[derive(Debug)]
pub struct Engine {
    books: HashMap<String, Book>,
}

impl Engine {
    /// Creates an engine with an empty book for every symbol the feed trades.
    pub fn new() -> Self {
        Engine {
            books: SYMBOLS
                .iter()
                .map(|(symbol, _)| (symbol.to_string(), Book::default()))
                .collect(),
        }
    }

    /// Records a feed message in the book.
    ///
    /// `New` orders rest on their side of the book. Cancels are ignored for
    /// now — every order is treated as still open.
    pub fn ingest(&mut self, msg: &OrderMessage) {
        let OrderMessage::New {
            id,
            timestamp,
            account,
            symbol,
            side,
            price,
            quantity,
        } = msg
        else {
            return;
        };

        let order = BookOrder {
            id: *id,
            timestamp: *timestamp,
            account: *account,
            price: *price,
            quantity: *quantity,
        };

        // `or_default` so a symbol the feed adds later still gets a book.
        let book = self.books.entry(symbol.clone()).or_default();
        match side {
            Side::Buy => {
                book.buys.insert(BuyOrder(order));
            }
            Side::Sell => {
                book.sells.insert(SellOrder(order));
            }
        }
    }

    /// Returns the book for `symbol`, if the engine has one.
    pub fn book(&self, symbol: &str) -> Option<&Book> {
        self.books.get(symbol)
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYMBOL: &str = "ETH-USDC";

    /// Builds a `New` message; quantity and account are irrelevant to ordering.
    fn new_order(id: OrderId, side: Side, price: f64, timestamp: u64) -> OrderMessage {
        OrderMessage::New {
            id,
            timestamp,
            account: 0,
            symbol: SYMBOL.to_string(),
            side,
            price,
            quantity: 1.0,
        }
    }

    fn ingest_all(msgs: &[OrderMessage]) -> Engine {
        let mut engine = Engine::new();
        for msg in msgs {
            engine.ingest(msg);
        }
        engine
    }

    fn buy_ids(engine: &Engine) -> Vec<OrderId> {
        engine
            .book(SYMBOL)
            .unwrap()
            .buys
            .iter()
            .map(|o| o.0.id)
            .collect()
    }

    fn sell_ids(engine: &Engine) -> Vec<OrderId> {
        engine
            .book(SYMBOL)
            .unwrap()
            .sells
            .iter()
            .map(|o| o.0.id)
            .collect()
    }

    #[test]
    fn sells_are_ordered_highest_price_first() {
        let engine = ingest_all(&[
            new_order(1, Side::Sell, 100.0, 10),
            new_order(2, Side::Sell, 102.0, 10),
            new_order(3, Side::Sell, 101.0, 10),
        ]);
        assert_eq!(sell_ids(&engine), vec![2, 3, 1]);
    }

    #[test]
    fn buys_are_ordered_lowest_price_first() {
        let engine = ingest_all(&[
            new_order(1, Side::Buy, 100.0, 10),
            new_order(2, Side::Buy, 102.0, 10),
            new_order(3, Side::Buy, 101.0, 10),
        ]);
        assert_eq!(buy_ids(&engine), vec![1, 3, 2]);
    }

    #[test]
    fn equal_prices_break_on_timestamp_oldest_first() {
        let engine = ingest_all(&[
            new_order(1, Side::Buy, 100.0, 30),
            new_order(2, Side::Buy, 100.0, 10),
            new_order(3, Side::Buy, 100.0, 20),
            new_order(4, Side::Sell, 100.0, 30),
            new_order(5, Side::Sell, 100.0, 10),
            new_order(6, Side::Sell, 100.0, 20),
        ]);
        assert_eq!(buy_ids(&engine), vec![2, 3, 1]);
        assert_eq!(sell_ids(&engine), vec![5, 6, 4]);
    }

    #[test]
    fn orders_sharing_a_price_and_timestamp_are_both_kept() {
        // Two orders in the same millisecond at the same price must not
        // collapse into one entry.
        let engine = ingest_all(&[
            new_order(1, Side::Buy, 100.0, 10),
            new_order(2, Side::Buy, 100.0, 10),
        ]);
        assert_eq!(buy_ids(&engine), vec![1, 2]);
    }

    #[test]
    fn sides_are_kept_in_separate_books_per_symbol() {
        let mut engine = Engine::new();
        engine.ingest(&new_order(1, Side::Buy, 100.0, 10));
        engine.ingest(&OrderMessage::New {
            id: 2,
            timestamp: 10,
            account: 0,
            symbol: "BTC-USDC".to_string(),
            side: Side::Buy,
            price: 1000.0,
            quantity: 1.0,
        });

        assert_eq!(buy_ids(&engine), vec![1]);
        assert_eq!(
            engine.book("BTC-USDC").unwrap().buys.len(),
            1,
            "the BTC order should land in its own book"
        );
        assert!(engine.book(SYMBOL).unwrap().sells.is_empty());
    }

    #[test]
    fn cancels_are_ignored_for_now() {
        let engine = ingest_all(&[
            new_order(1, Side::Buy, 100.0, 10),
            OrderMessage::Cancel {
                id: 2,
                timestamp: 20,
                account: 0,
                target_id: 1,
            },
        ]);
        assert_eq!(
            buy_ids(&engine),
            vec![1],
            "the cancel should not remove order 1"
        );
    }
}

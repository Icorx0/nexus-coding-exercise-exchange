//! A price-time ordered view of the orders published on the feed, and the
//! matching that happens as they arrive.
//!
//! The feed publishes limit orders but never matches them, so this module
//! keeps the book we build up from that stream: one buy side and one sell side
//! per symbol, each held in a sorted tree so the order to match against is
//! always at the front. Every incoming order is settled against the opposite
//! side on arrival, and each fill is recorded as a [`Trade`].

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::ops::Bound;

use serde::Serialize;
use tracing::info;

use crate::feed::{AccountId, FeedState, OrderId, OrderMessage, SYMBOLS, Side};

/// Quantities reach the feed rounded to one decimal place, so a remainder
/// smaller than this is float noise left by repeated subtraction rather than
/// real size. Treating it as zero keeps dust orders out of the book.
const QUANTITY_EPSILON: f64 = 1e-9;

/// A resting limit order, stripped of the feed metadata the book does not need.
#[derive(Debug, Clone)]
pub struct BookOrder {
    pub id: OrderId,
    pub timestamp: u64,
    pub account: AccountId,
    pub price: f64,
    pub quantity: f64,
}

impl BookOrder {
    /// A sentinel used as a range bound, not a real order.
    ///
    /// It sorts after every genuine order at `price` on *either* side, so
    /// `range((Excluded(pivot), Unbounded))` starts at the first order that
    /// strictly crosses `price` — the lowest buy above it on the buy side, the
    /// highest sell below it on the sell side.
    fn pivot(price: f64) -> Self {
        BookOrder {
            id: OrderId::MAX,
            timestamp: u64::MAX,
            account: 0,
            price,
            quantity: 0.0,
        }
    }
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

/// A match between an incoming order and one resting on the book.
#[derive(Debug, Clone, Serialize)]
pub struct Trade {
    /// The incoming order's timestamp, i.e. when the match happened.
    pub timestamp: u64,
    pub symbol: String,
    /// The resting order's price. The incoming order crossed it, so the trade
    /// prints at the price that was already on the book.
    pub price: f64,
    pub quantity: f64,
    pub buy_order: OrderId,
    pub buy_account: AccountId,
    pub sell_order: OrderId,
    pub sell_account: AccountId,
}

/// The two sides of a single symbol's book.
///
/// Both sets iterate front to back in the priority order defined above, so
/// `buys.first()` is the lowest bid and `sells.first()` is the highest ask.
#[derive(Debug, Default)]
pub struct Book {
    pub buys: BTreeSet<BuyOrder>,
    pub sells: BTreeSet<SellOrder>,
}

impl Book {
    /// The first buy in book order priced strictly above `price`: the order an
    /// incoming sell at `price` matches against.
    ///
    /// The buy side is ordered lowest first, so this is the *lowest* bid that
    /// still crosses. One tree descent, no scan.
    fn first_buy_above(&self, price: f64) -> Option<&BookOrder> {
        self.buys
            .range((
                Bound::Excluded(BuyOrder(BookOrder::pivot(price))),
                Bound::Unbounded,
            ))
            .next()
            .map(|order| &order.0)
    }

    /// The first sell in book order priced strictly below `price`: the order an
    /// incoming buy at `price` matches against.
    ///
    /// The sell side is ordered highest first, so this is the *highest* ask
    /// that still crosses.
    fn first_sell_below(&self, price: f64) -> Option<&BookOrder> {
        self.sells
            .range((
                Bound::Excluded(SellOrder(BookOrder::pivot(price))),
                Bound::Unbounded,
            ))
            .next()
            .map(|order| &order.0)
    }
}

/// Holds one [`Book`] per trading pair plus the trades matched out of them.
#[derive(Debug)]
pub struct Engine {
    books: HashMap<String, Book>,
    trades: Vec<Trade>,
}

impl Engine {
    /// Creates an engine with an empty book for every symbol the feed trades.
    pub fn new() -> Self {
        Engine {
            books: SYMBOLS
                .iter()
                .map(|(symbol, _)| (symbol.to_string(), Book::default()))
                .collect(),
            trades: Vec::new(),
        }
    }

    /// Records a feed message in the book and settles whatever it crosses.
    ///
    /// `New` orders rest on their side of the book and are then matched against
    /// the opposite side. Cancels are ignored for now — every order is treated
    /// as still open.
    ///
    /// `feed` is needed because an order that fills completely also has to stop
    /// being a candidate for a random cancel, and that pool lives on the feed
    /// state. The caller holds both locks, so the book and the feed never
    /// disagree about which orders are open.
    pub fn ingest(&mut self, msg: &OrderMessage, feed: &mut FeedState) {
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
                book.buys.insert(BuyOrder(order.clone()));
            }
            Side::Sell => {
                book.sells.insert(SellOrder(order.clone()));
            }
        }

        // The order joins the book first, then we see what it crosses.
        self.match_order(symbol, *side, order, feed);
    }

    /// Settles `incoming` against the opposite side of the book, one fill at a
    /// time, until it is exhausted or nothing over there crosses it any more.
    fn match_order(
        &mut self,
        symbol: &str,
        side: Side,
        mut incoming: BookOrder,
        feed: &mut FeedState,
    ) {
        // Split the borrow so trades can be appended while a book is held.
        let Engine { books, trades } = self;
        let Some(book) = books.get_mut(symbol) else {
            return;
        };

        // Lift the incoming order back out of the book while it is being
        // matched, so it cannot be found as its own counterparty and so its
        // shrinking quantity is tracked in one place. Whatever survives goes
        // back in below. Nothing can observe the gap: the caller holds both
        // the feed and the engine lock for this whole call.
        match side {
            Side::Buy => book.buys.remove(&BuyOrder(incoming.clone())),
            Side::Sell => book.sells.remove(&SellOrder(incoming.clone())),
        };

        while incoming.quantity > QUANTITY_EPSILON {
            // Step 2: the first order on the far side that crosses this price.
            let crossing = match side {
                Side::Buy => book.first_sell_below(incoming.price),
                Side::Sell => book.first_buy_above(incoming.price),
            };
            // Cloned to end the borrow on `book` before it is mutated below.
            let Some(resting) = crossing.cloned() else {
                break;
            };

            // Step 3: the smaller of the two is filled whole and leaves; the
            // larger stays with the difference. Equal sizes fill each other
            // exactly, so neither survives.
            let fill = incoming.quantity.min(resting.quantity);
            if fill <= QUANTITY_EPSILON {
                // Unreachable while every resting order carries real size, but
                // a zero-size fill would spin here forever holding both locks.
                break;
            }

            // Step 4: record the trade. Trades are appended as they happen, so
            // the list stays sorted by timestamp and the newest is on top.
            let (buy, sell) = match side {
                Side::Buy => (&incoming, &resting),
                Side::Sell => (&resting, &incoming),
            };
            trades.push(Trade {
                timestamp: incoming.timestamp,
                symbol: symbol.to_string(),
                price: resting.price,
                quantity: fill,
                buy_order: buy.id,
                buy_account: buy.account,
                sell_order: sell.id,
                sell_account: sell.account,
            });
            info!(
                "Matched {} {} @ {} (buy #{} acct {} / sell #{} acct {})",
                fill, symbol, resting.price, buy.id, buy.account, sell.id, sell.account
            );

            // The resting order comes out either way; it only goes back if it
            // was the larger of the two. Its price and timestamp are unchanged,
            // so it keeps its place in the queue.
            let remainder = resting.quantity - fill;
            let mut left_over = resting.clone();
            left_over.quantity = remainder;
            match side {
                Side::Buy => {
                    book.sells.remove(&SellOrder(resting.clone()));
                    if remainder > QUANTITY_EPSILON {
                        book.sells.insert(SellOrder(left_over));
                    }
                }
                Side::Sell => {
                    book.buys.remove(&BuyOrder(resting.clone()));
                    if remainder > QUANTITY_EPSILON {
                        book.buys.insert(BuyOrder(left_over));
                    }
                }
            }
            if remainder <= QUANTITY_EPSILON {
                feed.remove_cancel_candidate(resting.id);
            }

            // Step 5: round again with what is left of the incoming order.
            incoming.quantity -= fill;
        }

        // Whatever is left of the incoming order rests; if nothing is left it
        // has gone, so it can no longer be a cancel target either.
        if incoming.quantity > QUANTITY_EPSILON {
            match side {
                Side::Buy => {
                    book.buys.insert(BuyOrder(incoming));
                }
                Side::Sell => {
                    book.sells.insert(SellOrder(incoming));
                }
            }
        } else {
            feed.remove_cancel_candidate(incoming.id);
        }
    }

    /// Returns the book for `symbol`, if the engine has one.
    pub fn book(&self, symbol: &str) -> Option<&Book> {
        self.books.get(symbol)
    }

    /// Every trade matched so far, oldest first.
    pub fn trades(&self) -> &[Trade] {
        &self.trades
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

    /// Builds a `New` message on the default symbol.
    fn order(id: OrderId, side: Side, price: f64, quantity: f64) -> OrderMessage {
        order_at(id, side, price, quantity, id)
    }

    /// As `order`, but with an explicit timestamp for the tie-break tests.
    fn order_at(
        id: OrderId,
        side: Side,
        price: f64,
        quantity: f64,
        timestamp: u64,
    ) -> OrderMessage {
        OrderMessage::New {
            id,
            timestamp,
            account: id as AccountId,
            symbol: SYMBOL.to_string(),
            side,
            price,
            quantity,
        }
    }

    /// Feeds `msgs` through the engine, mirroring what the live feed does:
    /// every new order also joins the cancel-candidate pool, so the tests can
    /// check that filled orders are taken back out of it.
    fn run(msgs: &[OrderMessage]) -> (Engine, FeedState) {
        let mut engine = Engine::new();
        let mut feed = FeedState::new(1);
        for msg in msgs {
            if let OrderMessage::New { id, account, .. } = msg {
                feed.push_cancel_candidate(*id, *account);
            }
            engine.ingest(msg, &mut feed);
        }
        (engine, feed)
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

    fn candidate_ids(feed: &FeedState) -> Vec<OrderId> {
        feed.cancel_candidates().iter().map(|(id, _)| *id).collect()
    }

    // ---- ordering -------------------------------------------------------

    #[test]
    fn sells_are_ordered_highest_price_first() {
        let (engine, _) = run(&[
            order(1, Side::Sell, 100.0, 1.0),
            order(2, Side::Sell, 102.0, 1.0),
            order(3, Side::Sell, 101.0, 1.0),
        ]);
        assert_eq!(sell_ids(&engine), vec![2, 3, 1]);
    }

    #[test]
    fn buys_are_ordered_lowest_price_first() {
        let (engine, _) = run(&[
            order(1, Side::Buy, 100.0, 1.0),
            order(2, Side::Buy, 102.0, 1.0),
            order(3, Side::Buy, 101.0, 1.0),
        ]);
        assert_eq!(buy_ids(&engine), vec![1, 3, 2]);
    }

    #[test]
    fn equal_prices_break_on_timestamp_oldest_first() {
        // Same price on both sides, which under the strict crossing rule does
        // not trade, so the queue order is what is under test here.
        let (engine, _) = run(&[
            order_at(1, Side::Buy, 100.0, 1.0, 30),
            order_at(2, Side::Buy, 100.0, 1.0, 10),
            order_at(3, Side::Buy, 100.0, 1.0, 20),
            order_at(4, Side::Sell, 100.0, 1.0, 30),
            order_at(5, Side::Sell, 100.0, 1.0, 10),
            order_at(6, Side::Sell, 100.0, 1.0, 20),
        ]);
        assert_eq!(buy_ids(&engine), vec![2, 3, 1]);
        assert_eq!(sell_ids(&engine), vec![5, 6, 4]);
    }

    #[test]
    fn orders_sharing_a_price_and_timestamp_are_both_kept() {
        // Two orders in the same millisecond at the same price must not
        // collapse into one entry.
        let (engine, _) = run(&[
            order_at(1, Side::Buy, 100.0, 1.0, 10),
            order_at(2, Side::Buy, 100.0, 1.0, 10),
        ]);
        assert_eq!(buy_ids(&engine), vec![1, 2]);
    }

    #[test]
    fn sides_are_kept_in_separate_books_per_symbol() {
        let mut engine = Engine::new();
        let mut feed = FeedState::new(1);
        engine.ingest(&order(1, Side::Buy, 100.0, 1.0), &mut feed);
        engine.ingest(
            &OrderMessage::New {
                id: 2,
                timestamp: 2,
                account: 0,
                symbol: "BTC-USDC".to_string(),
                side: Side::Buy,
                price: 1000.0,
                quantity: 1.0,
            },
            &mut feed,
        );

        assert_eq!(buy_ids(&engine), vec![1]);
        assert_eq!(engine.book("BTC-USDC").unwrap().buys.len(), 1);
        assert!(engine.book(SYMBOL).unwrap().sells.is_empty());
    }

    #[test]
    fn cancels_are_ignored_for_now() {
        let (engine, _) = run(&[
            order(1, Side::Buy, 100.0, 1.0),
            OrderMessage::Cancel {
                id: 2,
                timestamp: 2,
                account: 0,
                target_id: 1,
            },
        ]);
        assert_eq!(
            buy_ids(&engine),
            vec![1],
            "the cancel must not remove order 1"
        );
    }

    // ---- matching -------------------------------------------------------

    #[test]
    fn crossing_orders_of_equal_size_fill_each_other_completely() {
        let (engine, feed) = run(&[
            order(1, Side::Sell, 100.0, 5.0),
            order(2, Side::Buy, 101.0, 5.0),
        ]);

        assert_eq!(engine.trades().len(), 1);
        let trade = &engine.trades()[0];
        assert_eq!(trade.quantity, 5.0);
        assert_eq!(trade.price, 100.0, "trades print at the resting price");
        assert_eq!(trade.buy_order, 2);
        assert_eq!(trade.sell_order, 1);
        assert_eq!(trade.buy_account, 2);
        assert_eq!(trade.sell_account, 1);

        assert!(buy_ids(&engine).is_empty(), "both orders are fully filled");
        assert!(sell_ids(&engine).is_empty());
        assert!(
            candidate_ids(&feed).is_empty(),
            "a filled order is no longer cancellable"
        );
    }

    #[test]
    fn the_incoming_order_keeps_its_remainder() {
        let (engine, feed) = run(&[
            order(1, Side::Sell, 100.0, 3.0),
            order(2, Side::Buy, 101.0, 5.0),
        ]);

        assert_eq!(engine.trades().len(), 1);
        assert_eq!(engine.trades()[0].quantity, 3.0);
        assert!(sell_ids(&engine).is_empty());

        let book = engine.book(SYMBOL).unwrap();
        let rest = &book.buys.first().unwrap().0;
        assert_eq!(rest.id, 2);
        assert_eq!(rest.quantity, 2.0);
        assert_eq!(
            candidate_ids(&feed),
            vec![2],
            "only the filled order leaves"
        );
    }

    #[test]
    fn the_resting_order_keeps_its_remainder() {
        let (engine, feed) = run(&[
            order(1, Side::Sell, 100.0, 8.0),
            order(2, Side::Buy, 101.0, 5.0),
        ]);

        assert_eq!(engine.trades().len(), 1);
        assert_eq!(engine.trades()[0].quantity, 5.0);
        assert!(buy_ids(&engine).is_empty());

        let book = engine.book(SYMBOL).unwrap();
        let rest = &book.sells.first().unwrap().0;
        assert_eq!(rest.id, 1);
        assert_eq!(rest.quantity, 3.0);
        assert_eq!(candidate_ids(&feed), vec![1]);
    }

    #[test]
    fn an_order_walks_down_several_levels_until_it_is_filled() {
        // Step 5: keep matching the remainder against the next crossing order.
        let (engine, _) = run(&[
            order(1, Side::Sell, 100.0, 2.0),
            order(2, Side::Sell, 99.0, 2.0),
            order(3, Side::Sell, 98.0, 2.0),
            order(4, Side::Buy, 101.0, 5.0),
        ]);

        let prices: Vec<f64> = engine.trades().iter().map(|t| t.price).collect();
        let sizes: Vec<f64> = engine.trades().iter().map(|t| t.quantity).collect();
        assert_eq!(
            prices,
            vec![100.0, 99.0, 98.0],
            "highest ask below the bid first"
        );
        assert_eq!(sizes, vec![2.0, 2.0, 1.0]);

        assert!(buy_ids(&engine).is_empty(), "the buy is fully filled");
        let book = engine.book(SYMBOL).unwrap();
        assert_eq!(
            sell_ids(&engine),
            vec![3],
            "only the partly filled ask is left"
        );
        assert_eq!(book.sells.first().unwrap().0.quantity, 1.0);
    }

    #[test]
    fn orders_that_do_not_cross_both_rest() {
        let (engine, _) = run(&[
            order(1, Side::Sell, 101.0, 5.0),
            order(2, Side::Buy, 100.0, 5.0),
        ]);
        assert!(engine.trades().is_empty());
        assert_eq!(sell_ids(&engine), vec![1]);
        assert_eq!(buy_ids(&engine), vec![2]);
    }

    #[test]
    fn orders_at_the_same_price_do_not_cross() {
        // Crossing is strict on both sides: an ask matches a bid *above* it and
        // a bid matches an ask *below* it, so an equal price does not trade.
        let (engine, _) = run(&[
            order(1, Side::Sell, 100.0, 5.0),
            order(2, Side::Buy, 100.0, 5.0),
        ]);
        assert!(engine.trades().is_empty());
        assert_eq!(sell_ids(&engine), vec![1]);
        assert_eq!(buy_ids(&engine), vec![2]);
    }

    #[test]
    fn an_incoming_sell_takes_the_lowest_bid_above_its_price() {
        let (engine, _) = run(&[
            order(1, Side::Buy, 102.0, 1.0),
            order(2, Side::Buy, 105.0, 1.0),
            order(3, Side::Buy, 103.0, 1.0),
            order(4, Side::Sell, 101.0, 1.0),
        ]);
        assert_eq!(engine.trades().len(), 1);
        assert_eq!(engine.trades()[0].buy_order, 1);
        assert_eq!(engine.trades()[0].price, 102.0);
        assert_eq!(buy_ids(&engine), vec![3, 2]);
    }

    #[test]
    fn an_incoming_buy_takes_the_highest_ask_below_its_price() {
        let (engine, _) = run(&[
            order(1, Side::Sell, 98.0, 1.0),
            order(2, Side::Sell, 95.0, 1.0),
            order(3, Side::Sell, 97.0, 1.0),
            order(4, Side::Buy, 99.0, 1.0),
        ]);
        assert_eq!(engine.trades().len(), 1);
        assert_eq!(engine.trades()[0].sell_order, 1);
        assert_eq!(engine.trades()[0].price, 98.0);
        assert_eq!(sell_ids(&engine), vec![3, 2]);
    }

    #[test]
    fn an_order_never_matches_against_itself() {
        // A single order resting on one side must not be found as its own
        // counterparty, whatever the price.
        let (engine, _) = run(&[order(1, Side::Buy, 100.0, 5.0)]);
        assert!(engine.trades().is_empty());
        assert_eq!(buy_ids(&engine), vec![1]);
    }

    #[test]
    fn trades_are_recorded_in_timestamp_order() {
        let (engine, _) = run(&[
            order_at(1, Side::Sell, 100.0, 1.0, 10),
            order_at(2, Side::Buy, 101.0, 1.0, 20),
            order_at(3, Side::Sell, 99.0, 1.0, 30),
            order_at(4, Side::Buy, 102.0, 1.0, 40),
        ]);
        let stamps: Vec<u64> = engine.trades().iter().map(|t| t.timestamp).collect();
        assert_eq!(stamps, vec![20, 40]);
        assert!(stamps.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn repeated_partial_fills_do_not_leave_dust_in_the_book() {
        // 0.1 is not exact in binary, so subtracting it repeatedly drifts. The
        // epsilon guard has to stop a near-zero remainder resting forever.
        let mut msgs = vec![order(1, Side::Sell, 100.0, 0.9)];
        for id in 2..=10 {
            msgs.push(order(id, Side::Buy, 101.0, 0.1));
        }
        let (engine, _) = run(&msgs);

        assert_eq!(engine.trades().len(), 9);
        assert!(
            sell_ids(&engine).is_empty(),
            "the ask filled exactly and must not linger as dust"
        );
        assert!(buy_ids(&engine).is_empty());
    }
}

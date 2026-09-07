use crate::engine::Engine;
use axum::{
    Json, Router,
    extract::{FromRef, Query, State},
    http::StatusCode,
    routing::{get, post},
};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time::sleep;
use tracing::{Level, info};
use tracing_subscriber::FmtSubscriber;

/// The trading pairs available on the feed, along with their initial mid prices.
/// Prices are quoted in the second token of the pair (e.g. USDC per ETH).
pub const SYMBOLS: [(&str, f64); 3] = [
    ("NEX-USDC", 10.0),
    ("ETH-USDC", 100.0),
    ("BTC-USDC", 1000.0),
];

/// How many recent open orders the generator remembers as candidates for
/// random cancellation.
const CANCEL_CANDIDATE_WINDOW: usize = 50;

/// The probability that any given generated message is a cancel instead of a
/// new order.
const CANCEL_PROBABILITY: f64 = 0.15;

/// Initializes and starts the order feed simulation.
/// This function sets up the logger, creates the initial feed state,
/// and starts the order generator and the API server.
pub async fn start_feed(num_accounts: u32, rate: f64) {
    // This subscriber is used for logging events.
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    // The feed state is created and wrapped in a thread-safe smart pointer.
    let shared_state = Arc::new(Mutex::new(FeedState::new(num_accounts)));

    // The order book we build from the feed is shared the same way, so the
    // generator task and the API handlers both write into it.
    let engine = Arc::new(Mutex::new(Engine::new()));

    // The order generator runs in a separate asynchronous task.
    let generator_state = Arc::clone(&shared_state);
    let generator_engine = Arc::clone(&engine);
    tokio::spawn(async move {
        produce_orders(generator_state, generator_engine, rate).await;
    });

    // The API server is started to handle incoming requests.
    run_server(shared_state, engine).await;
}

/// Represents the side of an order in the market.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// This implementation allows parsing a string into a Side.
/// It is case-insensitive and supports single-letter shorthands.
impl FromStr for Side {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "BUY" | "BID" | "B" => Ok(Side::Buy),
            "SELL" | "ASK" | "S" => Ok(Side::Sell),
            _ => Err(format!("'{}' is not a valid side", s)),
        }
    }
}

/// A type alias for account identifiers for clarity.
pub type AccountId = u32;

/// A type alias for order/message identifiers. IDs are strictly increasing
/// across all messages on the feed and start at 1.
pub type OrderId = u64;

/// Defines the types of messages published on the order feed.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum OrderMessage {
    /// A new limit order entering the market.
    New {
        id: OrderId,
        timestamp: u64,
        account: AccountId,
        symbol: String,
        side: Side,
        price: f64,
        quantity: f64,
    },
    /// A request to cancel a previously placed order.
    /// `target_id` refers to the `id` of the order to cancel.
    Cancel {
        id: OrderId,
        timestamp: u64,
        account: AccountId,
        target_id: OrderId,
    },
}

impl OrderMessage {
    /// Returns the message's own feed sequence ID.
    pub fn id(&self) -> OrderId {
        match self {
            OrderMessage::New { id, .. } => *id,
            OrderMessage::Cancel { id, .. } => *id,
        }
    }
}

/// Represents the entire state of the order feed at a given moment.
pub struct FeedState {
    /// Every message published so far, in feed order.
    pub messages: Vec<OrderMessage>,
    /// The ID to be assigned to the next message.
    next_id: OrderId,
    /// The current mid price for each symbol; each drifts as a random walk.
    mids: HashMap<String, f64>,
    /// Recently generated (order id, account) pairs, used to pick cancel targets.
    cancel_candidates: Vec<(OrderId, AccountId)>,
    /// The number of simulated accounts placing orders.
    num_accounts: u32,
}

impl FeedState {
    /// Creates a new feed state with the initial mid price for each symbol.
    pub fn new(num_accounts: u32) -> Self {
        let mut mids = HashMap::new();
        for (symbol, mid) in SYMBOLS {
            mids.insert(symbol.to_string(), mid);
        }
        FeedState {
            messages: Vec::new(),
            next_id: 1,
            mids,
            cancel_candidates: Vec::new(),
            num_accounts: num_accounts.max(1),
        }
    }
}

/// Returns the current UNIX timestamp in milliseconds.
fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Rounds a value to two decimal places (used for prices).
fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Rounds a value to one decimal place (used for quantities).
fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// A loop that publishes a new random message on the feed at a fixed rate and
/// records it in the order book.
async fn produce_orders(state: Arc<Mutex<FeedState>>, engine: Arc<Mutex<Engine>>, rate: f64) {
    let interval_ms = (1000.0 / rate.clamp(0.1, 1000.0)) as u64;
    loop {
        sleep(Duration::from_millis(interval_ms)).await;

        // The feed lock is released before the engine lock is taken, so the
        // two are never held at once and the writers cannot deadlock.
        let msg = {
            let mut state = state.lock().unwrap();
            let msg = generate_message(&mut state);
            state.messages.push(msg.clone());
            msg
        };

        info!("Publishing message: {:?}", msg);
        engine.lock().unwrap().ingest(&msg);
    }
}

/// Generates the next random feed message.
/// Most messages are new limit orders priced around a per-symbol drifting mid
/// price; occasionally a recent order is cancelled instead.
fn generate_message(state: &mut FeedState) -> OrderMessage {
    let mut rng = rand::thread_rng();
    let timestamp = now_millis();
    let id = state.next_id;
    state.next_id += 1;

    // Occasionally cancel a recent order instead of placing a new one.
    if !state.cancel_candidates.is_empty() && rng.gen_bool(CANCEL_PROBABILITY) {
        let idx = rng.gen_range(0..state.cancel_candidates.len());
        let (target_id, account) = state.cancel_candidates.swap_remove(idx);
        return OrderMessage::Cancel {
            id,
            timestamp,
            account,
            target_id,
        };
    }

    // Pick a random symbol and let its mid price drift slightly.
    let (symbol, _) = SYMBOLS[rng.gen_range(0..SYMBOLS.len())];
    let mid = state.mids.get_mut(symbol).unwrap();
    *mid *= 1.0 + rng.gen_range(-0.002..0.002);

    // Price the order randomly around the mid so that buys and sells
    // frequently cross each other.
    let price = round2(*mid * (1.0 + rng.gen_range(-0.005..0.005)));
    let quantity = round1(rng.gen_range(1.0..10.0));
    let side = if rng.gen_bool(0.5) {
        Side::Buy
    } else {
        Side::Sell
    };
    let account = rng.gen_range(0..state.num_accounts);

    // Remember this order as a potential cancel target.
    state.cancel_candidates.push((id, account));
    if state.cancel_candidates.len() > CANCEL_CANDIDATE_WINDOW {
        state.cancel_candidates.remove(0);
    }

    OrderMessage::New {
        id,
        timestamp,
        account,
        symbol: symbol.to_string(),
        side,
        price,
        quantity,
    }
}

/// The state shared with every request handler.
///
/// Handlers pull out only the piece they need through the `FromRef` impls
/// below, so a handler that just touches the feed keeps its original signature.
#[derive(Clone)]
struct AppState {
    feed: Arc<Mutex<FeedState>>,
    engine: Arc<Mutex<Engine>>,
}

impl FromRef<AppState> for Arc<Mutex<FeedState>> {
    fn from_ref(app: &AppState) -> Self {
        Arc::clone(&app.feed)
    }
}

impl FromRef<AppState> for Arc<Mutex<Engine>> {
    fn from_ref(app: &AppState) -> Self {
        Arc::clone(&app.engine)
    }
}

/// Runs the API server to handle HTTP requests.
/// The server serves the order feed and accepts order submissions.
async fn run_server(shared_state: Arc<Mutex<FeedState>>, engine: Arc<Mutex<Engine>>) {
    let app = Router::new()
        .route("/orders", get(get_orders))
        .route("/order", post(submit_order))
        .route("/cancel", post(submit_cancel))
        .route("/symbols", get(get_symbols))
        .with_state(AppState {
            feed: shared_state,
            engine,
        });

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    info!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// Defines the query parameters for the /orders endpoint.
#[derive(Deserialize)]
struct GetOrdersQuery {
    /// Return only messages with an ID strictly greater than this value.
    /// Use the highest ID you have seen so far to poll for new messages.
    since: Option<OrderId>,
    /// If provided, return only the last n messages instead.
    n: Option<usize>,
}

/// Handles GET requests for retrieving feed messages.
/// With `since`, returns all messages after the given ID (use this to poll).
/// With `n`, returns the last n messages.
async fn get_orders(
    State(state): State<Arc<Mutex<FeedState>>>,
    Query(params): Query<GetOrdersQuery>,
) -> Json<Vec<OrderMessage>> {
    let state = state.lock().unwrap();
    if let Some(n) = params.n {
        let start = state.messages.len().saturating_sub(n);
        return Json(state.messages[start..].to_vec());
    }
    let since = params.since.unwrap_or(0);
    // Messages are stored in ID order, so we can skip everything up to `since`.
    let start = state.messages.partition_point(|m| m.id() <= since);
    Json(state.messages[start..].to_vec())
}

/// Defines the request body for submitting a new order.
#[derive(Debug, Deserialize)]
struct SubmitOrderRequest {
    account: AccountId,
    symbol: String,
    side: Side,
    price: f64,
    quantity: f64,
}

/// Defines the response returned when a message is accepted onto the feed.
#[derive(Serialize)]
struct SubmitResponse {
    id: OrderId,
}

/// Handles POST requests for submitting a new order onto the feed.
/// The order is validated, assigned an ID, published like any other feed
/// message, and recorded in the order book.
async fn submit_order(
    State(state): State<Arc<Mutex<FeedState>>>,
    State(engine): State<Arc<Mutex<Engine>>>,
    Json(req): Json<SubmitOrderRequest>,
) -> Result<Json<SubmitResponse>, (StatusCode, String)> {
    if !SYMBOLS.iter().any(|(s, _)| *s == req.symbol) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("'{}' is not a valid symbol", req.symbol),
        ));
    }
    if req.price <= 0.0 || req.quantity <= 0.0 {
        return Err((
            StatusCode::BAD_REQUEST,
            "price and quantity must be positive".to_string(),
        ));
    }

    // As in the generator, the feed lock is dropped before the engine lock.
    let msg = {
        let mut state = state.lock().unwrap();
        let id = state.next_id;
        state.next_id += 1;
        let msg = OrderMessage::New {
            id,
            timestamp: now_millis(),
            account: req.account,
            symbol: req.symbol,
            side: req.side,
            price: req.price,
            quantity: req.quantity,
        };
        state.messages.push(msg.clone());
        msg
    };

    info!("Received order: {:?}", msg);
    engine.lock().unwrap().ingest(&msg);
    Ok(Json(SubmitResponse { id: msg.id() }))
}

/// Defines the request body for submitting a cancel.
#[derive(Debug, Deserialize)]
struct SubmitCancelRequest {
    account: AccountId,
    target_id: OrderId,
}

/// Handles POST requests for submitting a cancel onto the feed.
/// The feed does not check whether the target order is still open; consumers
/// of the feed decide what a cancel means for their own book.
async fn submit_cancel(
    State(state): State<Arc<Mutex<FeedState>>>,
    Json(req): Json<SubmitCancelRequest>,
) -> Json<SubmitResponse> {
    let mut state = state.lock().unwrap();
    let id = state.next_id;
    state.next_id += 1;
    let msg = OrderMessage::Cancel {
        id,
        timestamp: now_millis(),
        account: req.account,
        target_id: req.target_id,
    };
    info!("Received cancel: {:?}", msg);
    state.messages.push(msg);
    Json(SubmitResponse { id })
}

/// Handles GET requests for the list of available symbols.
async fn get_symbols() -> Json<Vec<String>> {
    Json(SYMBOLS.iter().map(|(s, _)| s.to_string()).collect())
}

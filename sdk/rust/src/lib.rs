//! # Spirit Rust SDK
//!
//! The official Rust client library for interacting with `SpiritROS` services. Provides
//! a typed, async API for inference requests, model management, session control, and
//! system monitoring.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use spirit_sdk::{SpiritClient, Config};
//!
//! # async fn example() -> Result<(), spirit_sdk::Error> {
//! let config = Config::default();
//! let client = SpiritClient::connect(config).await?;
//! # Ok(())
//! # }
//! ```

#![warn(clippy::all, clippy::pedantic)]

pub mod client;
pub mod config;
pub mod error;
pub mod types;

pub use client::SpiritClient;
pub use config::Config;
pub use error::Error;
pub use types::*;

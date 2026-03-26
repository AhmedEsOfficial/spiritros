//! Spirit Guard — security policy enforcement daemon entry point.
//!
//! Listens on a Unix domain socket and validates capability tokens
//! for all privileged operations in SpiritROS.

use spirit_security::SecurityConfig;

#[tokio::main]
async fn main() {
    let config = SecurityConfig::default();
    println!(
        "spirit-guard: starting Spirit Guard daemon (socket={})",
        config.guard_socket_path
    );
    println!(
        "spirit-guard: enforcement mode = {}",
        if config.enforce {
            "ENFORCING"
        } else {
            "PERMISSIVE"
        }
    );
    println!("spirit-guard: daemon stub exiting (implementation pending)");
}

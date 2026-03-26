//! SpiritFS — FUSE daemon entry point.
//!
//! Mounts the SpiritFS semantic filesystem and runs until signaled to stop.

use spirit_spiritfs::SpiritFsConfig;

#[tokio::main]
async fn main() {
    let config = SpiritFsConfig::default();
    println!(
        "spiritfs: starting SpiritFS daemon (mount_point={})",
        config.mount_point
    );
    println!("spiritfs: Phase 1 — FUSE-based semantic filesystem");
    println!("spiritfs: daemon stub exiting (implementation pending)");
}

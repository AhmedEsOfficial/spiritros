//! Memory pressure monitoring via Linux PSI (Pressure Stall Information).
//!
//! Reads `/proc/pressure/memory` to detect memory stalls and maps them
//! to discrete pressure levels. The memory manager uses these levels to
//! trigger graduated responses:
//!
//! | Level    | Trigger                  | Response                         |
//! |----------|--------------------------|----------------------------------|
//! | None     | some < 10%               | Normal operation                 |
//! | Low      | some 10–25%              | Log warning                      |
//! | Medium   | some > 25%               | Shrink ZONE_TENSOR scratch       |
//! | High     | full > 5%                | Evict cold KV-cache to swap      |
//! | Critical | full > 20%               | Unload least-used model          |
//!
//! // PHASE1-SHIM: In Phase 2, pressure monitoring is a kernel subsystem
//! // with interrupt-driven response instead of polling.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use tracing::{debug, info, warn};

use crate::config::PressureConfig;
use crate::types::{PressureInfo, PressureLevel};

/// Monitors memory pressure via PSI and exposes the current level.
#[derive(Debug)]
pub struct PressureMonitor {
    /// Current pressure level (atomic for lock-free reads).
    level: Arc<AtomicU8>,
    /// Whether monitoring is available and enabled.
    available: bool,
    /// Configuration.
    config: PressureConfig,
}

impl PressureMonitor {
    /// Create a new pressure monitor.
    pub fn new(config: &PressureConfig) -> Self {
        let available = config.enabled && Self::probe_psi();

        if config.enabled && !available {
            warn!("PSI not available; memory pressure monitoring disabled");
        }

        Self {
            level: Arc::new(AtomicU8::new(PressureLevel::None as u8)),
            available,
            config: config.clone(),
        }
    }

    /// Whether pressure monitoring is active.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Current pressure level (lock-free read).
    #[must_use]
    pub fn level(&self) -> PressureLevel {
        Self::u8_to_level(self.level.load(Ordering::Relaxed))
    }

    /// Read raw PSI info from procfs.
    pub fn read_psi(&self) -> Option<PressureInfo> {
        if !self.available {
            return None;
        }
        Self::parse_psi_file("/proc/pressure/memory").ok()
    }

    /// Poll once: read PSI, update level, return the new level.
    pub fn poll(&self) -> PressureLevel {
        if !self.available {
            return PressureLevel::None;
        }

        match Self::parse_psi_file("/proc/pressure/memory") {
            Ok(info) => {
                let new_level = info.level();
                let old = self.level.swap(new_level as u8, Ordering::Relaxed);
                let old_level = Self::u8_to_level(old);

                if new_level != old_level {
                    match new_level {
                        PressureLevel::None | PressureLevel::Low => {
                            debug!(
                                level = %new_level,
                                some_avg10 = info.some_avg10,
                                full_avg10 = info.full_avg10,
                                "pressure level changed"
                            );
                        }
                        PressureLevel::Medium => {
                            info!(
                                level = %new_level,
                                some_avg10 = info.some_avg10,
                                "medium memory pressure — consider shrinking \
                                 ZONE_TENSOR"
                            );
                        }
                        PressureLevel::High => {
                            warn!(
                                level = %new_level,
                                full_avg10 = info.full_avg10,
                                "high memory pressure — evicting cold \
                                 KV-cache"
                            );
                        }
                        PressureLevel::Critical => {
                            warn!(
                                level = %new_level,
                                full_avg10 = info.full_avg10,
                                "CRITICAL memory pressure — emergency \
                                 measures active"
                            );
                        }
                    }
                }

                new_level
            }
            Err(_) => PressureLevel::None,
        }
    }

    /// Spawn a background task that polls PSI at the configured interval.
    ///
    /// Returns a handle to the shared level that gets updated. The task
    /// runs until the returned `Arc` is the last reference and the
    /// `CancellationToken` is dropped.
    pub fn spawn_monitor(&self) -> Option<tokio::task::JoinHandle<()>> {
        if !self.available {
            return None;
        }

        let level = Arc::clone(&self.level);
        let interval_ms = self.config.poll_interval_ms;

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(interval_ms));
            loop {
                interval.tick().await;
                if let Ok(info) = Self::parse_psi_file("/proc/pressure/memory") {
                    level.store(info.level() as u8, Ordering::Relaxed);
                }
            }
        });

        info!(
            interval_ms,
            "started PSI pressure monitoring background task"
        );

        Some(handle)
    }

    // ── Internal ────────────────────────────────────────────────────────

    fn probe_psi() -> bool {
        std::path::Path::new("/proc/pressure/memory").exists()
    }

    /// Parse `/proc/pressure/memory`.
    ///
    /// Format:
    /// ```text
    /// some avg10=0.00 avg60=0.00 avg300=0.00 total=0
    /// full avg10=0.00 avg60=0.00 avg300=0.00 total=0
    /// ```
    fn parse_psi_file(path: &str) -> std::result::Result<PressureInfo, String> {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        Ok(Self::parse_psi_content(&content))
    }

    fn parse_psi_content(content: &str) -> PressureInfo {
        let mut info = PressureInfo::default();

        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                continue;
            }

            let is_some = parts[0] == "some";
            let is_full = parts[0] == "full";

            if !is_some && !is_full {
                continue;
            }

            for part in &parts[1..] {
                if let Some((key, val)) = part.split_once('=') {
                    let v: f64 = val.parse().unwrap_or(0.0);
                    match (is_some, key) {
                        (true, "avg10") => info.some_avg10 = v,
                        (true, "avg60") => info.some_avg60 = v,
                        (true, "avg300") => info.some_avg300 = v,
                        (false, "avg10") => info.full_avg10 = v,
                        (false, "avg60") => info.full_avg60 = v,
                        (false, "avg300") => info.full_avg300 = v,
                        _ => {}
                    }
                }
            }
        }

        info
    }

    fn u8_to_level(v: u8) -> PressureLevel {
        match v {
            0 => PressureLevel::None,
            1 => PressureLevel::Low,
            2 => PressureLevel::Medium,
            3 => PressureLevel::High,
            4 => PressureLevel::Critical,
            _ => PressureLevel::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_psi_content() {
        let content = "\
some avg10=1.50 avg60=0.80 avg300=0.20 total=12345
full avg10=0.10 avg60=0.05 avg300=0.01 total=6789
";
        let info = PressureMonitor::parse_psi_content(content);
        assert!((info.some_avg10 - 1.50).abs() < f64::EPSILON);
        assert!((info.some_avg60 - 0.80).abs() < f64::EPSILON);
        assert!((info.some_avg300 - 0.20).abs() < f64::EPSILON);
        assert!((info.full_avg10 - 0.10).abs() < f64::EPSILON);
        assert!((info.full_avg60 - 0.05).abs() < f64::EPSILON);
        assert!((info.full_avg300 - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pressure_level_from_psi() {
        let none = PressureMonitor::parse_psi_content(
            "some avg10=5.00 avg60=0.00 avg300=0.00 total=0\n\
             full avg10=0.00 avg60=0.00 avg300=0.00 total=0",
        );
        assert_eq!(none.level(), PressureLevel::None);

        let medium = PressureMonitor::parse_psi_content(
            "some avg10=30.00 avg60=0.00 avg300=0.00 total=0\n\
             full avg10=0.00 avg60=0.00 avg300=0.00 total=0",
        );
        assert_eq!(medium.level(), PressureLevel::Medium);

        let critical = PressureMonitor::parse_psi_content(
            "some avg10=50.00 avg60=0.00 avg300=0.00 total=0\n\
             full avg10=25.00 avg60=0.00 avg300=0.00 total=0",
        );
        assert_eq!(critical.level(), PressureLevel::Critical);
    }

    #[test]
    fn test_disabled_monitor() {
        let config = PressureConfig {
            enabled: false,
            poll_interval_ms: 1000,
        };
        let monitor = PressureMonitor::new(&config);
        assert!(!monitor.is_available());
        assert_eq!(monitor.level(), PressureLevel::None);
        assert!(monitor.read_psi().is_none());
    }

    #[test]
    fn test_u8_roundtrip() {
        for level in [
            PressureLevel::None,
            PressureLevel::Low,
            PressureLevel::Medium,
            PressureLevel::High,
            PressureLevel::Critical,
        ] {
            assert_eq!(PressureMonitor::u8_to_level(level as u8), level);
        }
    }

    #[test]
    fn test_parse_empty_content() {
        let info = PressureMonitor::parse_psi_content("");
        assert_eq!(info.level(), PressureLevel::None);
    }

    #[test]
    fn test_parse_malformed_content() {
        let info = PressureMonitor::parse_psi_content("garbage data\n");
        assert_eq!(info.level(), PressureLevel::None);
    }
}

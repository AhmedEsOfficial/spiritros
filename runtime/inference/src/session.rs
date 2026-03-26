//! Inference session lifecycle management.
//!
//! Each inference session owns pre-allocated buffers for token streaming.
//! Buffers are allocated once at session creation and reused across
//! multiple generation requests within the same session. This ensures
//! the hot path (token receive loop) never allocates.
//!
//! Sessions follow a state machine:
//! ```text
//! Pending → Ready → Prefilling → Decoding → Idle → (next prompt) → Prefilling
//!                                              └→ Terminated
//! ```

use std::collections::HashMap;
use std::time::Instant;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use tracing::{debug, info, instrument, warn};

use crate::error::{InferenceError, Result};
use crate::types::{
    InferencePhase, KvSnapshot, ModelId, Priority, SamplingConfig, SessionId, SessionState, Token,
};

/// A single inference session with pre-allocated buffers.
pub struct InferenceSession {
    /// Unique session identifier.
    pub id: SessionId,
    /// The model this session is using.
    pub model_id: ModelId,
    /// Model name for display.
    pub model_name: String,
    /// Current session state.
    state: SessionState,
    /// Priority level for this session.
    pub priority: Priority,
    /// Pre-allocated token buffer (reused across generations).
    token_buffer: Vec<Token>,
    /// Current position in the token buffer.
    token_pos: usize,
    /// Number of tokens in the current KV-cache.
    pub kv_token_count: usize,
    /// Total tokens generated across all generations in this session.
    pub total_tokens_generated: u64,
    /// Current inference phase.
    pub phase: InferencePhase,
    /// When this session was created.
    pub created_at: DateTime<Utc>,
    /// When this session was last active.
    pub last_active_at: Instant,
    /// Sampling configuration for this session.
    pub sampling: SamplingConfig,
    /// KV-cache snapshots taken from this session.
    pub snapshots: Vec<KvSnapshot>,
}

impl InferenceSession {
    /// Create a new session with pre-allocated buffers.
    ///
    /// The `buffer_capacity` parameter controls the size of the pre-allocated
    /// token buffer. This buffer is reused across generations to avoid
    /// heap allocation on the hot path.
    #[must_use]
    pub fn new(
        model_id: ModelId,
        model_name: String,
        priority: Priority,
        buffer_capacity: usize,
    ) -> Self {
        let mut token_buffer = Vec::with_capacity(buffer_capacity);
        // Pre-allocate with dummy tokens to ensure the Vec's backing
        // storage is actually allocated (not just capacity reserved).
        token_buffer.resize_with(buffer_capacity, || Token {
            text: String::new(),
            token_id: 0,
            logprob: None,
            done: false,
        });

        Self {
            id: SessionId::new(),
            model_id,
            model_name,
            state: SessionState::Pending,
            priority,
            token_buffer,
            token_pos: 0,
            kv_token_count: 0,
            total_tokens_generated: 0,
            phase: InferencePhase::Idle,
            created_at: Utc::now(),
            last_active_at: Instant::now(),
            sampling: SamplingConfig::default(),
            snapshots: Vec::new(),
        }
    }

    /// Get the current session state.
    #[must_use]
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Transition to Ready state (model loaded).
    pub fn mark_ready(&mut self) -> Result<()> {
        self.ensure_state(SessionState::Pending)?;
        self.state = SessionState::Ready;
        self.last_active_at = Instant::now();
        debug!(session = %self.id, "Session ready");
        Ok(())
    }

    /// Transition to Prefilling state (prompt submitted).
    pub fn mark_prefilling(&mut self) -> Result<()> {
        if !self.state.can_prompt() {
            return Err(InferenceError::InvalidSessionState {
                session_id: self.id,
                state: self.state,
                expected: vec![SessionState::Ready, SessionState::Idle],
            });
        }
        self.state = SessionState::Prefilling;
        self.phase = InferencePhase::Prefill;
        self.token_pos = 0;
        self.last_active_at = Instant::now();
        Ok(())
    }

    /// Transition to Decoding state (prefill complete, generating tokens).
    pub fn mark_decoding(&mut self) -> Result<()> {
        self.ensure_state(SessionState::Prefilling)?;
        self.state = SessionState::Decoding;
        self.phase = InferencePhase::Decode;
        self.last_active_at = Instant::now();
        Ok(())
    }

    /// Transition to Idle state (generation complete).
    pub fn mark_idle(&mut self) -> Result<()> {
        if !self.state.is_generating() {
            return Err(InferenceError::InvalidSessionState {
                session_id: self.id,
                state: self.state,
                expected: vec![SessionState::Prefilling, SessionState::Decoding],
            });
        }
        self.state = SessionState::Idle;
        self.phase = InferencePhase::Idle;
        self.last_active_at = Instant::now();
        Ok(())
    }

    /// Terminate the session.
    pub fn terminate(&mut self) {
        self.state = SessionState::Terminated;
        self.phase = InferencePhase::Idle;
        info!(session = %self.id, "Session terminated");
    }

    /// Push a token into the pre-allocated buffer.
    ///
    /// Reuses existing buffer slots to avoid allocation on the hot path.
    /// If the buffer is full, overwrites from the beginning (ring semantics).
    pub fn push_token(&mut self, token: Token) {
        let idx = self.token_pos % self.token_buffer.len();
        self.token_buffer[idx] = token;
        self.token_pos += 1;
        self.total_tokens_generated += 1;
        self.last_active_at = Instant::now();
    }

    /// Get the last N tokens from the buffer.
    #[must_use]
    pub fn last_tokens(&self, n: usize) -> Vec<&Token> {
        let count = n.min(self.token_pos).min(self.token_buffer.len());
        let start = if self.token_pos > self.token_buffer.len() {
            self.token_pos % self.token_buffer.len()
        } else {
            0
        };
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let idx = (start + self.token_pos - count + i) % self.token_buffer.len();
            result.push(&self.token_buffer[idx]);
        }
        result
    }

    /// Reset the token buffer position for a new generation.
    pub fn reset_buffer(&mut self) {
        self.token_pos = 0;
    }

    /// Duration since the session was last active.
    #[must_use]
    pub fn idle_duration(&self) -> std::time::Duration {
        self.last_active_at.elapsed()
    }

    /// Check if the session is in the expected state.
    fn ensure_state(&self, expected: SessionState) -> Result<()> {
        if self.state != expected {
            return Err(InferenceError::InvalidSessionState {
                session_id: self.id,
                state: self.state,
                expected: vec![expected],
            });
        }
        Ok(())
    }
}

// ── Session Manager ─────────────────────────────────────────────────────────

/// Manages the lifecycle of all active inference sessions.
///
/// Thread-safe: uses `parking_lot::RwLock` for minimal contention on reads.
pub struct SessionManager {
    /// Active sessions indexed by ID.
    sessions: RwLock<HashMap<SessionId, InferenceSession>>,
    /// Maximum number of concurrent sessions.
    max_sessions: usize,
    /// Pre-allocated token buffer size per session.
    buffer_capacity: usize,
}

impl SessionManager {
    /// Create a new session manager.
    #[must_use]
    pub fn new(max_sessions: usize, buffer_capacity: usize) -> Self {
        Self {
            sessions: RwLock::new(HashMap::with_capacity(max_sessions)),
            max_sessions,
            buffer_capacity,
        }
    }

    /// Create a new inference session.
    ///
    /// Returns the session ID. The session starts in `Pending` state.
    #[instrument(skip(self))]
    pub fn create_session(
        &self,
        model_id: ModelId,
        model_name: String,
        priority: Priority,
    ) -> Result<SessionId> {
        let sessions = self.sessions.read();
        let active = sessions
            .values()
            .filter(|s| s.state() != SessionState::Terminated)
            .count();
        if active >= self.max_sessions {
            return Err(InferenceError::SessionLimitReached {
                active,
                limit: self.max_sessions,
            });
        }
        drop(sessions);

        let session = InferenceSession::new(model_id, model_name, priority, self.buffer_capacity);
        let id = session.id;

        self.sessions.write().insert(id, session);
        info!(session = %id, "Session created");
        Ok(id)
    }

    /// Get a reference to a session for reading.
    pub fn with_session<F, T>(&self, id: &SessionId, f: F) -> Result<T>
    where
        F: FnOnce(&InferenceSession) -> T,
    {
        let sessions = self.sessions.read();
        let session = sessions
            .get(id)
            .ok_or(InferenceError::SessionNotFound(*id))?;
        Ok(f(session))
    }

    /// Get a mutable reference to a session for state transitions.
    pub fn with_session_mut<F, T>(&self, id: &SessionId, f: F) -> Result<T>
    where
        F: FnOnce(&mut InferenceSession) -> T,
    {
        let mut sessions = self.sessions.write();
        let session = sessions
            .get_mut(id)
            .ok_or(InferenceError::SessionNotFound(*id))?;
        Ok(f(session))
    }

    /// Destroy a session.
    #[instrument(skip(self))]
    pub fn destroy_session(&self, id: &SessionId) -> Result<()> {
        let mut sessions = self.sessions.write();
        let session = sessions
            .get_mut(id)
            .ok_or(InferenceError::SessionNotFound(*id))?;
        session.terminate();
        // Remove terminated sessions immediately
        sessions.remove(id);
        Ok(())
    }

    /// Get the number of active (non-terminated) sessions.
    #[must_use]
    pub fn active_count(&self) -> usize {
        let sessions = self.sessions.read();
        sessions
            .values()
            .filter(|s| s.state() != SessionState::Terminated)
            .count()
    }

    /// Get all session IDs.
    pub fn session_ids(&self) -> Vec<SessionId> {
        self.sessions.read().keys().copied().collect()
    }

    /// Evict idle sessions that have been inactive for longer than `max_idle`.
    ///
    /// Returns the number of sessions evicted.
    pub fn evict_idle(&self, max_idle: std::time::Duration) -> usize {
        let mut sessions = self.sessions.write();
        let before = sessions.len();
        sessions.retain(|_, s| {
            if s.state() == SessionState::Idle && s.idle_duration() > max_idle {
                info!(session = %s.id, "Evicting idle session");
                false
            } else {
                true
            }
        });
        before - sessions.len()
    }

    /// Get the maximum session limit.
    #[must_use]
    pub fn max_sessions(&self) -> usize {
        self.max_sessions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model_id() -> ModelId {
        ModelId::new()
    }

    #[test]
    fn test_session_state_machine() {
        let mut session =
            InferenceSession::new(test_model_id(), "test".into(), Priority::Normal, 256);

        assert_eq!(session.state(), SessionState::Pending);

        session.mark_ready().unwrap();
        assert_eq!(session.state(), SessionState::Ready);

        session.mark_prefilling().unwrap();
        assert_eq!(session.state(), SessionState::Prefilling);

        session.mark_decoding().unwrap();
        assert_eq!(session.state(), SessionState::Decoding);

        session.mark_idle().unwrap();
        assert_eq!(session.state(), SessionState::Idle);

        // Can go back to generating from idle
        session.mark_prefilling().unwrap();
        assert_eq!(session.state(), SessionState::Prefilling);
    }

    #[test]
    fn test_session_invalid_transition() {
        let mut session =
            InferenceSession::new(test_model_id(), "test".into(), Priority::Normal, 256);

        // Can't go from Pending to Decoding
        assert!(session.mark_decoding().is_err());
        // Can't go from Pending to Idle
        assert!(session.mark_idle().is_err());
    }

    #[test]
    fn test_session_token_buffer() {
        let mut session =
            InferenceSession::new(test_model_id(), "test".into(), Priority::Normal, 4);

        for i in 0..4 {
            session.push_token(Token {
                text: format!("tok{i}"),
                token_id: i as u32,
                logprob: None,
                done: false,
            });
        }

        assert_eq!(session.total_tokens_generated, 4);
        let last2 = session.last_tokens(2);
        assert_eq!(last2.len(), 2);
        assert_eq!(last2[0].text, "tok2");
        assert_eq!(last2[1].text, "tok3");
    }

    #[test]
    fn test_session_token_buffer_wraps() {
        let mut session =
            InferenceSession::new(test_model_id(), "test".into(), Priority::Normal, 4);

        // Push 6 tokens into a buffer of size 4 (should wrap)
        for i in 0..6 {
            session.push_token(Token {
                text: format!("tok{i}"),
                token_id: i as u32,
                logprob: None,
                done: false,
            });
        }

        assert_eq!(session.total_tokens_generated, 6);
    }

    #[test]
    fn test_session_manager_create_and_destroy() {
        let mgr = SessionManager::new(4, 256);

        let id = mgr
            .create_session(test_model_id(), "model".into(), Priority::Normal)
            .unwrap();
        assert_eq!(mgr.active_count(), 1);

        mgr.destroy_session(&id).unwrap();
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_session_manager_limit_enforced() {
        let mgr = SessionManager::new(2, 256);
        let mid = test_model_id();

        mgr.create_session(mid, "m".into(), Priority::Normal)
            .unwrap();
        mgr.create_session(mid, "m".into(), Priority::Normal)
            .unwrap();

        // Third session should fail
        let result = mgr.create_session(mid, "m".into(), Priority::Normal);
        assert!(result.is_err());
    }

    #[test]
    fn test_session_manager_with_session() {
        let mgr = SessionManager::new(4, 256);
        let mid = test_model_id();
        let id = mgr
            .create_session(mid, "m".into(), Priority::Normal)
            .unwrap();

        let state = mgr.with_session(&id, |s| s.state()).unwrap();
        assert_eq!(state, SessionState::Pending);

        mgr.with_session_mut(&id, |s| s.mark_ready().unwrap())
            .unwrap();

        let state = mgr.with_session(&id, |s| s.state()).unwrap();
        assert_eq!(state, SessionState::Ready);
    }

    #[test]
    fn test_session_manager_evict_idle() {
        let mgr = SessionManager::new(4, 256);
        let mid = test_model_id();

        let id = mgr
            .create_session(mid, "m".into(), Priority::Normal)
            .unwrap();

        // Transition to idle
        mgr.with_session_mut(&id, |s| {
            s.mark_ready().unwrap();
            s.mark_prefilling().unwrap();
            s.mark_decoding().unwrap();
            s.mark_idle().unwrap();
        })
        .unwrap();

        // Evict with 0 duration — should evict immediately
        let evicted = mgr.evict_idle(std::time::Duration::from_secs(0));
        assert_eq!(evicted, 1);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_session_terminate() {
        let mut session =
            InferenceSession::new(test_model_id(), "test".into(), Priority::Normal, 256);
        session.terminate();
        assert_eq!(session.state(), SessionState::Terminated);
    }
}

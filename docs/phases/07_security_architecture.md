# Phase 07: Security Architecture — Zero-Trust Inference

**Roadmap Alignment:** Phase 1 — cross-cutting (foundation), Phase 3 — M3.2 (full hardening)
**Estimated Duration:** Month 3–5 (Phase 1 basics), Months 22–24 (Phase 3 hardening)
**Dependencies:** Phase 01 (Foundation), Phase 03 (Inference Engine)
**Architecture Plan Sections:** §10

---

## Context & Overview

SpiritROS faces unique security challenges because its most powerful subsystem — the LLM — can generate arbitrary text (including executable code), be manipulated via prompt injection, leak sensitive data from its context window, and be used as an oracle for data exfiltration.

The security model treats **every inference output as untrusted** until validated. All access control is capability-based (unforgeable tokens granting specific permissions), with no ambient authorities. This phase implements the capability system, inference sandboxing, output filtering (Spirit Guard), and model integrity verification.

In Phase 1, the security model is enforced at the userspace level via the Spirit Runtime. In Phase 12+, it becomes a kernel-level enforcement mechanism.

---

## Goals

1. **Capability system** — Unforgeable tokens for all system permissions (inference, memory, GPU, filesystem, network)
2. **Inference sandboxing** — Per-session configurable sandbox with token limits, tool restrictions, and output filters
3. **Spirit Guard** — Output filtering pipeline (pattern matching + semantic analysis) on every inference response
4. **Model integrity** — SHA-256 verification, Ed25519 signing, Sigstore/cosign support
5. **Audit logging** — Every prompt/response pair logged with configurable retention
6. **Network security** — WireGuard for inter-node, mTLS for API access, default-deny firewall

---

## Design Decisions

### DD-07.1: Capability-Based Access Control (Not RBAC)
**Decision:** Use capability tokens (inspired by seL4 and Fuchsia/Zircon) rather than traditional role-based access control.
**Rationale:** Capabilities are more granular and composable than roles. A process that needs to call inference and read `/data` gets exactly `CAP_INFER_PROMPT` + `CAP_FS_READ(/data)` — nothing more. This prevents privilege escalation and limits blast radius.

### DD-07.2: Output Filtering Before Delivery
**Decision:** Every inference output passes through Spirit Guard *before* reaching the requesting process. This is non-optional and not bypassable (except by CAP_INFER_ADMIN holders).
**Rationale:** The LLM can generate API keys, passwords, or harmful content. Filtering at the delivery boundary ensures no process receives dangerous output accidentally. Admin override exists for debugging.

### DD-07.3: Audit Log is Append-Only
**Decision:** The audit log (`/var/log/spirit/audit.log`) is append-only. Even root cannot delete entries (enforced via filesystem attributes in Phase 1, kernel enforcement in Phase 12).
**Rationale:** Audit trails are useless if they can be tampered with. Append-only ensures a complete record for incident investigation.

### DD-07.4: Model Signing is Configurable
**Decision:** Model signing verification is configurable: `required` (reject unsigned models), `warn` (allow with warning), `off`. Default: `warn` in Phase 1, `required` in Phase 3.
**Rationale:** The model supply chain is not yet mature enough for mandatory signing (many community models are unsigned). But production deployments should require it.

### DD-07.5: Sandbox Policies are Declarative
**Decision:** Sandbox policies are defined in TOML files and attached to sessions/processes at creation time. They cannot be modified after creation (immutable once applied).
**Rationale:** Prevents privilege escalation during a session. If a process needs different capabilities, it must create a new session with the required policy.

---

## Components & Tasks

### 1. Capability System (`runtime/security/capabilities.rs`)
```rust
pub struct Capability {
    id: CapabilityId,          // Unforgeable random token
    kind: CapabilityKind,
    scope: CapabilityScope,    // e.g., path for FS, port for NET
    granted_to: ProcessId,
    granted_by: ProcessId,
    created_at: Timestamp,
    expires_at: Option<Timestamp>,
}

pub enum CapabilityKind {
    // Inference
    InferAdmin,
    InferLoadModel,
    InferCreateSession,
    InferPrompt,
    InferKvManage,
    InferBatch,
    // Memory
    MemoryAdmin,
    TensorAlloc,
    VramAccess,
    KvZoneAccess,
    // GPU
    GpuAdmin,
    GpuCompute,
    GpuExclusive,
    // Filesystem
    FsAdmin,
    FsRead(PathBuf),
    FsWrite(PathBuf),
    FsSemantic,
    // Network
    NetAdmin,
    NetListen(u16),
    NetConnect(String),
    NetMesh,
}

pub struct CapabilityManager {
    capabilities: HashMap<CapabilityId, Capability>,
}

impl CapabilityManager {
    pub fn grant(&mut self, target: ProcessId, kind: CapabilityKind, scope: CapabilityScope) -> Result<CapabilityId>;
    pub fn revoke(&mut self, cap_id: CapabilityId) -> Result<()>;
    pub fn check(&self, process: ProcessId, kind: CapabilityKind) -> bool;
    pub fn list(&self, process: ProcessId) -> Vec<&Capability>;
}
```

### 2. Inference Sandbox (`runtime/security/sandbox.rs`)
```rust
pub struct InferenceSandbox {
    policy: SandboxPolicy,
    session_id: SessionId,
    token_count: AtomicU64,
}

#[derive(Deserialize)]
pub struct SandboxPolicy {
    pub max_output_tokens: u64,
    pub allowed_tools: Vec<String>,
    pub blocked_patterns: Vec<String>,    // Regex patterns to block in output
    pub output_filter: String,            // "spirit-guard" or "none"
    pub context_isolation: bool,          // No cross-session context leakage
    pub rate_limit: Option<RateLimit>,
    pub audit: bool,
}

impl InferenceSandbox {
    pub fn new(policy: SandboxPolicy, session: SessionId) -> Self;
    pub fn check_prompt(&self, prompt: &str) -> Result<(), SandboxViolation>;
    pub fn filter_output(&self, output: &str) -> Result<String, SandboxViolation>;
    pub fn increment_tokens(&self, count: u64) -> Result<(), SandboxViolation>;
}
```

### 3. Spirit Guard — Output Filter (`runtime/security/guard.rs`)
```rust
pub struct SpiritGuard {
    pattern_rules: Vec<PatternRule>,        // Regex-based pattern matching
    classifier: Option<ClassifierModel>,     // Small model for semantic analysis
    rate_limiter: RateLimiter,
    audit_logger: AuditLogger,
}

pub struct PatternRule {
    pattern: Regex,
    action: FilterAction,  // Block, Redact, Warn
    description: String,
}

impl SpiritGuard {
    pub fn new(policy_path: &Path) -> Result<Self>;

    pub fn filter(&self, session: &SessionId, output: &str) -> FilterResult {
        // 1. Pattern matching (fast path)
        for rule in &self.pattern_rules {
            if rule.pattern.is_match(output) {
                match rule.action {
                    FilterAction::Block => return FilterResult::Blocked(rule.description.clone()),
                    FilterAction::Redact => { /* redact matched content */ }
                    FilterAction::Warn => { /* log warning, pass through */ }
                }
            }
        }

        // 2. Semantic analysis (optional, slower)
        if let Some(classifier) = &self.classifier {
            let risk_score = classifier.analyze(output);
            if risk_score > THRESHOLD {
                return FilterResult::Blocked("Semantic filter triggered".into());
            }
        }

        // 3. Rate limiting
        if !self.rate_limiter.allow(session) {
            return FilterResult::RateLimited;
        }

        FilterResult::Passed(output.to_string())
    }
}
```

### 4. Guard Policy Configuration
```yaml
# /etc/spirit/guard-policy.yaml
patterns:
  - name: api_keys
    pattern: "(sk-[a-zA-Z0-9]{32,}|AKIA[0-9A-Z]{16}|ghp_[a-zA-Z0-9]{36})"
    action: redact
    description: "API key detected in output"

  - name: passwords
    pattern: "(?i)(password|passwd|secret)\\s*[:=]\\s*\\S+"
    action: redact
    description: "Password-like pattern in output"

  - name: private_keys
    pattern: "-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"
    action: block
    description: "Private key in output"

  - name: sql_injection
    pattern: "(DROP TABLE|DELETE FROM|UPDATE .* SET|INSERT INTO).*;"
    action: warn
    description: "SQL-like statement in output"

semantic_analysis:
  enabled: true
  model: "llama-guard-3-1b"
  threshold: 0.8

rate_limiting:
  per_session: 100/min
  per_user: 500/min
  global: 2000/min

audit:
  enabled: true
  log_path: /var/log/spirit/audit.log
  retention_days: 90
  include_prompts: true
  include_responses: true
```

### 5. Model Integrity Verification
```rust
pub struct ModelVerifier {
    trust_store: TrustStore,  // Ed25519 public keys, Sigstore roots
}

impl ModelVerifier {
    pub fn verify_hash(&self, path: &Path, expected_sha256: &str) -> Result<()>;
    pub fn verify_signature(&self, path: &Path, signature: &[u8]) -> Result<VerifyResult>;
    pub fn verify_sigstore(&self, path: &Path, identity: &str) -> Result<VerifyResult>;

    pub fn on_model_load(&self, path: &Path, policy: SigningPolicy) -> Result<()> {
        match policy {
            SigningPolicy::Required => { /* reject if unsigned */ }
            SigningPolicy::Warn => { /* warn if unsigned, allow */ }
            SigningPolicy::Off => { /* no verification */ }
        }
    }
}
```

### 6. Audit Logger
```rust
pub struct AuditLogger {
    log_path: PathBuf,
    writer: AppendOnlyWriter,
}

pub struct AuditEntry {
    timestamp: DateTime<Utc>,
    session_id: SessionId,
    process_id: ProcessId,
    event_type: AuditEventType,  // Prompt, Response, CapabilityGrant, SandboxViolation
    content: String,
    metadata: HashMap<String, String>,
}

impl AuditLogger {
    pub fn log(&self, entry: AuditEntry) -> Result<()>;
    pub fn query(&self, filter: AuditFilter) -> Result<Vec<AuditEntry>>;
}
```

### 7. Network Security (Phase 1 Basics)
- Default iptables/nftables rules: deny all inbound except configured ports (11434, 50051, 8080)
- WireGuard tunnel setup for Spirit Mesh communication
- mTLS for the inference API (using `rustls` + self-signed CA or Let's Encrypt)

---

## Acceptance Criteria

- [ ] Capability system works: processes can only perform actions they have capabilities for
- [ ] A process without `CAP_INFER_PROMPT` cannot submit inference requests
- [ ] Sandbox enforces `max_output_tokens` (generation stops at limit)
- [ ] Spirit Guard blocks output containing API key patterns
- [ ] Spirit Guard redacts password-like patterns
- [ ] Audit log records every prompt and response
- [ ] Audit log is append-only (`chattr +a` or equivalent)
- [ ] Model hash verification rejects tampered model files
- [ ] Model signature verification works with Ed25519 keys
- [ ] Unsigned models produce a warning when `signing_policy = "warn"`
- [ ] Rate limiting prevents abuse (returns 429-equivalent after limit)
- [ ] Network firewall defaults to deny-all-inbound
- [ ] mTLS authentication works for the inference API
- [ ] `spirit security status` shows capability grants, sandbox policies, and guard stats

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Capability system too restrictive for development | Developer friction | Provide a "dev mode" with relaxed defaults; document how to grant capabilities |
| Spirit Guard false positives (blocks legitimate output) | User frustration | Tunable sensitivity; `warn` mode instead of `block`; easy policy editing |
| Semantic classifier model competes with primary LLM | Inference slowdown | Use tiny model (1B params); run on CPU at P4; make semantic analysis optional |
| Audit log grows unbounded | Disk full | Configurable retention; log rotation; exclude high-volume low-value events |
| Userspace capability enforcement is bypassable | Security gap | Enforce at all IPC boundaries; Phase 12 kernel enforcement closes this gap |

---

## Key References

- Architecture Plan §10 (Security Architecture — Zero-Trust Inference)
- Architecture Plan §10.1 (Threat Model)
- Architecture Plan §10.2 (Capability-Based Security)
- Architecture Plan §10.3 (Inference Sandboxing)
- Architecture Plan §10.4 (Model Integrity)
- Architecture Plan §10.5 (Spirit Guard — Output Filtering)
- Architecture Plan §10.6 (Network Security)

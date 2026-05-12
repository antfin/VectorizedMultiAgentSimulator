# Secrets and env vars

LERO needs LLM API keys at runtime. On local hosts this is just `.env`;
on OVH containers it's a Fernet-encrypted shipping channel.

## Local

```bash
echo 'OPENAI_API_KEY=sk-…' >> .env
# or ANTHROPIC_API_KEY, OVH_API_KEY (LiteLLM dispatches by model prefix)
```

`LiteLlmClient._load_env_once()` walks up from cwd to load `.env` on
first `generate()` call. The Submit page's preflight row "LLM API key
present for cfg.lero" asserts the key is in `os.environ` before
allowing submit.

## OVH — encrypted-secrets channel

OVH containers don't have access to your local `.env`. The dispatch
path Fernet-encrypts API keys into two env vars on the job spec:

- `MS_ENCRYPTED_SECRETS` — ciphertext blob (key:value pairs).
- `MS_SECRETS_PASSPHRASE` — fresh 32-byte urlsafe token per submission.

In-container, `application/secrets_priming.py::prime_os_environ_from_encrypted_secrets()`
runs BEFORE the LERO orchestrator constructs the `LiteLlmClient`:

1. Reads `MS_ENCRYPTED_SECRETS` from env.
2. Decrypts via `FernetSecretsAdapter().decrypt_from_env()`.
3. Sets each decrypted key/value into `os.environ` (preserves
   existing values — manual export wins).
4. Logs at WARNING level so the line is visible in `ovhai job logs`.

## Verification

After firing a LERO OVH job:

```bash
ovhai job get <job_id> | grep MS_
#   MS_ENCRYPTED_SECRETS: gAAAAAB...
#   MS_SECRETS_PASSPHRASE: ...

ovhai job logs <job_id> | grep "primed os.environ"
#   primed os.environ with 2 decrypted secrets (keys: ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY'])
```

If the log line is absent, `MS_ENCRYPTED_SECRETS` wasn't shipped — the
Submit page's preflight should have caught this; check
`tests/unit/application/test_submission_lero_secrets.py` if the
collection helper regressed.

## Per-submission passphrase rotation

Each submission generates a fresh 32-byte passphrase via
`secrets.token_urlsafe(32)`. Reuse would mean a leaked passphrase from
one job decrypts the next; per-submission rotation closes that window.
Pinned by `test_passphrase_is_fresh_per_submission`.

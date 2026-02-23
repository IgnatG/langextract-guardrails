# CHANGELOG

<!-- version list -->

## v1.1.1 (2026-02-23)

### Bug Fixes

- Implement OnFailAction semantics for validators and enhance validation outcome handling
  ([`b2ce56a`](https://github.com/IgnatG/langcore-guardrails/commit/b2ce56a75ade41399d6da811889da35913673e2d))


## v1.1.0 (2026-02-22)

### Features

- Introduce new validators using pydantic and OnFailAction enum
  ([`5c88c9d`](https://github.com/IgnatG/langcore-guardrails/commit/5c88c9d2de66a1f52240eb14dda206d789627766))


## v1.0.4 (2026-02-21)

### Bug Fixes

- Enhance JsonSchemaValidator to enforce additionalProperties for implicit objects
  ([`631a5b6`](https://github.com/IgnatG/langcore-guardrails/commit/631a5b6a72fda28768a7bab676ef616986893333))


## v1.0.3 (2026-02-21)

### Bug Fixes

- Update langcore dependency to langcore and adjust versioning in pyproject.toml and
  uv.lock
  ([`a4a00b9`](https://github.com/IgnatG/langcore-guardrails/commit/a4a00b972fca2c2003c213c614f1652f3add3956))


## v1.0.2 (2026-02-21)

### Bug Fixes

- Add error-only correction mode to GuardrailLanguageModel; update tests for new behavior
  ([`32e427a`](https://github.com/IgnatG/langcore-guardrails/commit/32e427a56c549208e782e51018e189587c26e796))


## v1.0.1 (2026-02-21)

### Bug Fixes

- Enhance GuardrailLanguageModel with concurrency and truncation features; add strict mode to
  JsonSchemaValidator
  ([`7410c35`](https://github.com/IgnatG/langcore-guardrails/commit/7410c352d9fff31816b363120a6dce35a420cc84))

- Update version to 1.0.0 in uv.lock and reorder imports in provider.py
  ([`df95b80`](https://github.com/IgnatG/langcore-guardrails/commit/df95b802cce64e0911bb5ec11fa08d85a0283edd))

### Chores

- Add Apache License 2.0 to the repository
  ([`c318694`](https://github.com/IgnatG/langcore-guardrails/commit/c3186947fddea26bc672fd42f816ac4c9d921ddb))


## v1.0.0 (2026-02-21)

- Initial Release

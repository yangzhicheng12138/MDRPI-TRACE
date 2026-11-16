# Contributing

Before submitting, run:

```bash
python -m pytest -q
node --test tests_frontend/*.test.mjs
python scripts/check_release.py
```

For issues, include reproduction steps, expected and actual behavior, and redacted error details. Do not upload credentials, identifiable patient information or document full texts without permission.

For UI changes, check desktop and mobile widths, light/dark themes, keyboard navigation, and English/Chinese switching. Keep user drafts, model output and backend option values intact. UI translations live in `modes/frontend/messages-en.js`; dynamic message formats and localization boundaries live in `i18n.js`. Browser code runs without a build step; frontend unit tests require Node.js 22+.

For teaching changes, check the distinction between student answers, teacher explanations and source materials, plus hint counts, revision and step progression.

For model-quality changes, document the model, cases, knowledge and web evidence conditions, and comparison method. Unit tests verify program behavior; they do not replace model-quality evaluation.

Preserve licenses for third-party code and update `THIRD_PARTY_NOTICES.md`. Preserve authors, source URLs, licenses and review scope for new teaching content.

---
doc_id: troubleshooting-login
title: Troubleshooting Sign-In Problems
tags: login, sign-in, password, locked, account, 2fa, mfa, reset, troubleshooting
updated: 2026-05-30
---

Common sign-in failures and fixes:

- **Wrong password** — use Forgot password on the sign-in screen. The reset
  link is valid for 60 minutes and can only be used once.
- **Account locked** — five failed attempts lock the account for 15 minutes.
  The lock clears automatically; support cannot shorten it.
- **Two-factor code rejected** — check that the device clock is accurate.
  Recovery codes are single-use and live in Settings, then Security.
- **SAML loop on Team plans** — the workspace admin must re-upload the
  identity provider metadata after a certificate rotation.
- **"Email not found"** — the account may have been created with a different
  address, or deleted. Deleted accounts cannot be restored after 30 days.

If none of these apply, open a support ticket with the exact error text.

<!-- ARCHIVED 2019-03-02 -->

Sign-in uses a local passphrase stored on the device. There is no password
reset, no two-factor authentication, no SAML, and no account lockout. If the
passphrase is lost, the notes cannot be recovered by any means and support
cannot help.

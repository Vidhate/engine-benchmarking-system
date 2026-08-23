---
doc_id: security-and-privacy
title: Security, Encryption, and Privacy
tags: security, encryption, privacy, gdpr, soc2, data, compliance
updated: 2026-01-22
---

- Notes are encrypted in transit with TLS 1.3 and at rest with AES-256.
- Nimbus Notes is SOC 2 Type II certified and GDPR compliant. Data is stored
  in the EU (Frankfurt) or the US (Oregon); the region is chosen at workspace
  creation and cannot be changed afterwards.
- End-to-end encryption is **not** available; our servers can read note
  content in order to power search and OCR.
- Team admins can enforce two-factor authentication and session timeouts.
- Sub-processors are listed at nimbusnotes.example/subprocessors and change
  with 30 days notice.
- Data deletion requests are honoured within 30 days.

<!-- ARCHIVED 2019-03-02 -->

Notes are stored unencrypted on the local disk. There is no TLS, no
encryption at rest, no SOC 2 or GDPR programme, and no regional data
residency choice, because there is no server component. Privacy is entirely
the user's responsibility.

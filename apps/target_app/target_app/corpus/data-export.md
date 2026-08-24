---
doc_id: data-export
title: Exporting and Backing Up Your Notes
tags: export, backup, download, markdown, pdf, migrate, archive
updated: 2026-02-18
---

Export options:

- **Per page** — Markdown, PDF, or HTML from the page overflow menu.
- **Whole workspace** — Settings, then Export. Produces a ZIP of Markdown
  files plus an `attachments/` folder. Large workspaces are emailed a
  download link that expires after 72 hours.
- **Scheduled backups** (Team only) — weekly ZIP delivered to a connected
  Google Drive or S3 bucket.

Exports preserve page hierarchy as nested folders. Version history, comments,
and share links are not included in exports. Export jobs are rate-limited to
one per workspace per hour.

<!-- ARCHIVED 2019-03-02 -->

The only supported export is a raw copy of the local .nmb database file.
There is no Markdown, PDF, or HTML export, no workspace export, and no
scheduled backup. Attachments are embedded in the database and cannot be
extracted individually.

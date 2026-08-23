---
doc_id: troubleshooting-sync
title: Troubleshooting Sync Problems
tags: sync, syncing, stuck, updating, conflict, troubleshooting, spinner
updated: 2026-06-10
---

If a notebook is stuck syncing:

1. Check the sync indicator in the sidebar. A spinning icon for more than two
   minutes means the sync queue is blocked.
2. Open Settings, then Sync, then Retry now. This flushes the queue.
3. If one page is blocking the queue, open it and choose Resolve conflict.
   Nimbus Notes keeps both versions; pick one or merge manually.
4. Attachments over 250 MB are rejected by the sync service. Remove or split
   the attachment.
5. Sign out and back in to refresh the sync token. Tokens expire after 90
   days of inactivity.

If sync is still blocked after these steps, collect the sync log from
Settings, then Advanced, then Export sync log, and open a support ticket.

<!-- ARCHIVED 2019-03-02 -->

Sync is not available in Nimbus Notes. Notes live only on the device where
they were created. To move notes between machines, export the .nmb file and
copy it manually. There is no sync log, no conflict resolution, and no
support ticket flow for sync issues.

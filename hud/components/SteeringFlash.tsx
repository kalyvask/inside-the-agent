"use client";

/**
 * SteeringFlash — turned into a no-op in v0.17b.
 *
 * Reviewer feedback: "The orange STEERING APPLIED toast overlaps the browser
 * and competes with the failure overlay. Never show steering toast and
 * verdict modal at the same time."
 *
 * The same information is already conveyed in two non-overlay places:
 *   - The intervention badge in the BrowserViewport title bar
 *     ("⚡ INTERVENTION · targeted · 2 edits") — auto-fades 3s after the
 *     steering event.
 *   - The InterventionTimeline (permanent audit log on the side panel).
 *
 * Kept as an exported stub so existing imports don't break.
 */

import type { SteeringEdit } from "@/lib/ws";

export default function SteeringFlash(_props: { edits: SteeringEdit[] }) {
  return null;
}

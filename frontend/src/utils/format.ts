// Display helpers shared by the views

/** An ISO timestamp from the API as local date and time, e.g. "11/10/2026, 9:47 PM"; "—" if missing. */
export function formatTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' })
}

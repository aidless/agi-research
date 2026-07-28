# MANIFEST -- Y1 Paper Release v3.7

> File inventory with SHA-256 checksums.
> Generated 2026-07-28 by Codex agent.

| File | Size (bytes) | SHA-256 |
|------|--------------|---------|| `CITATION.cff` | 2013 | `e313a3d659171b2097e872b81c96805d7b0f3396f0a4390e1fda22f0f4b29694` |
| `paper.md` | 38048 | `6a42613d9c5954a1131c8f0e5a074b817ee89f85b887d057143658c8ec5b6f2c` |
| `README.md` | 5264 | `f9f8dfd8de90a4d3401004b7541588bf79e7db4f201a8a5916e9366e79c68ee0` |
| `related_work_4systems.md` | 13271 | `c74d6bc65863cba803b7c0874bc329d399a5bae3b005897427d3a9c737c159fe` |
| `SUBMISSION.md` | 5977 | `db24c90c7905d583145e21db26ca56af60355e8b03b388565282966c0c30545b` |
| `y1_9hypothesis_framework.md` | 12394 | `b347e377cfd7f62039cf3dfb4417babf48080377c037147ff270fc6982166f71` |
| `figures\y1_fig1_y13_lunarlander.png` | 30195 | `34575e761627a101b33649548068033da095c6fd050806894d557c8c2aed824b` |
| `figures\y1_fig2_y13_per_seed.png` | 35693 | `ed2d7c29790836963a1e1f3b5c5895ceb8cd1fe7305fb70d8f8ba362a2559c13` |
| `figures\y1_fig3_dlr_crossenv.png` | 44633 | `5e86ccc1f6c233b1dd64ce4ceffa0ca2d4776bc08f4f9367b2f1071cb593b2e1` |
| `figures\y1_fig4_y13_lambda.png` | 44880 | `de8fb3c0ebd01ae900ac1314e6ce547618c1af8439221a40372ce5eab90e6a86` |
| `tables\y1_table1_dlr_summary.tex` | 752 | `233347ff78b2b72929e5f8aac9d27a3171cc7aaf2b0381d5b8f83adac59b52b9` |
| `tables\y1_table2_y13_summary.tex` | 827 | `e364b49808bfe5fbd7aa0bfd698273d10a570ece2d5369cd25518bf455d09568` |

---

## Total

- Files: 12
- Total size: 233947 bytes
- Paper text: ~5600 words / ~38 KB
- Figures: 4 PNG, ~155 KB total
- Tables: 2 LaTeX, ~1.6 KB total

## Regenerating checksums

`powershell
Get-ChildItem . -Recurse -File | ForEach-Object {
    e364b49808bfe5fbd7aa0bfd698273d10a570ece2d5369cd25518bf455d09568 = (Get-FileHash -Path $_.FullName -Algorithm SHA256).Hash.ToLower()
    Write-Host "$($_.Name): $($_.Length) bytes / $hash"
}
`

## What this manifest is for

The SHA-256 checksums let downstream users verify that the release
hasn't been tampered with. Anyone with the original tarball can
recompute the checksums and compare.

This is especially important given the paper's NO_SELF_DECEPTION.md
discipline: every artifact in this release has a known origin and
has not been silently modified.

---

*Generated 2026-07-28 by Codex agent.*
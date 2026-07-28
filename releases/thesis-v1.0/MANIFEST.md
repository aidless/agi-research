# MANIFEST -- Thesis v1.0 Release

> File inventory with SHA-256 checksums.
> Generated 2026-07-29 by Codex agent.

| File | Size (bytes) | SHA-256 |
|------|--------------|---------|| `README.md` | 4527 | `91a370575e74e0bbaac1d647f0579d1dc2a596ac51e57b2ab4f7a2cdc39f65e2` |
| `thesis.html` | 132038 | `96619a14eca6cce58c981c680f5839465148a20454c393f57d5508266dd2cfb5` |
| `thesis.md` | 115205 | `253c35ffdcf6a7d571f0e04f699d0cc4816a9771d4afe03221594ea35cf5f3da` |
| `thesis.pdf` | 188305 | `072363143a7e02c79bb03cdf614259c30cfe01f1d0ff2aaecda2d1c6f369e342` |

---

## Total

- Files: 4
- Total size: 440075 bytes
- Thesis text: ~3000+ lines / ~115 KB

## Regenerating checksums

`powershell
Get-ChildItem . -Recurse -File | ForEach-Object {
    072363143a7e02c79bb03cdf614259c30cfe01f1d0ff2aaecda2d1c6f369e342 = (Get-FileHash -Path $_.FullName -Algorithm SHA256).Hash.ToLower()
    Write-Host "$($_.Name): $($_.Length) bytes / $hash"
}
`

## What this manifest is for

The SHA-256 checksums let downstream users verify that the release
hasn't been tampered with. Anyone with the original tarball can
recompute the checksums and compare.

---

*Generated 2026-07-29 by Codex agent.*
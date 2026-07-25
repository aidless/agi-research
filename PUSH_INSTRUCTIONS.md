# PUSH INSTRUCTIONS — How to push this workspace to GitHub

> **Goal**: Push all 30+ commits from `E:\agi-research` to
> `https://github.com/aidless/agi-research` as a public repo.
> **Date**: 2026-07-26

---

## 0. Pre-flight check (already done)

- [x] Git author configured: `刘泽文 <aidless@users.noreply.github.com>`
- [x] Remote configured: `https://github.com/aidless/agi-research.git`
- [x] LICENSE, AUTHORS, README in place with attribution
- [x] 30+ commits ready, last commit attribution: `3e709e3`
- [x] .gitignore clean (excludes checkpoints, __pycache__, *.pt)

---

## 1. Create the empty GitHub repo

Go to https://github.com/new

**Settings**:
- Owner: `aidless` (your GitHub username)
- Repository name: `agi-research`
- Description: `Archimedes: A Self-Improving AGI Substrate. Independent 5-year AGI research program by 刘泽文 (Liu Zewen). AGI-2026-001.`
- Visibility: **Public** (essential for IP protection)
- Initialize: **NONE** (uncheck all: README, .gitignore, license — we have our own)

Click **Create repository**.

You should land on a page like:
> `Quick setup — if you've done this kind of thing before`
> ...or create a new repository on the command line
> ```
> git remote add origin https://github.com/aidless/agi-research.git
> git branch -M main
> git push -u origin main
> ```

---

## 2. Authenticate (one of three options)

### Option A: Personal Access Token (recommended, fastest)

GitHub removed password auth for HTTPS in 2021. Use a PAT.

1. https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)" (or fine-grained)
3. Note: `agi-research-push`
4. Expiration: 90 days (or your preference)
5. Scopes: `repo` (full repo access)
6. Click "Generate token"
7. **COPY THE TOKEN NOW** (you won't see it again)

### Option B: SSH key

If you already have an SSH key configured on GitHub:
```bash
git remote set-url origin git@github.com:aidless/agi-research.git
```

### Option C: GitHub CLI (if installed)

```bash
gh auth login
git push -u origin main
```

---

## 3. Push

### Option 1: User pushes directly (recommended)

You (aidless) run this in PowerShell or Git Bash:

```bash
cd E:\agi-research
git push -u origin main
```

When prompted:
- Username: `aidless`
- Password: **paste your PAT** (Option A) or use SSH key (Option B)

You'll see all commits being uploaded. Expected output:
```
Enumerating objects: ~150, done.
Counting objects: 100% (150/150), done.
Delta compression using up to 8 threads
Compressing objects: 100% (130/130), done.
Writing objects: 100% (150/150), ~500KB | 5MB/s, done.
Total 150 (delta 50), reused 0 (delta 0)
To https://github.com/aidless/agi-research.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

### Option 2: User gives PAT to Codex, Codex pushes

If you want Codex to push:
1. Provide the PAT to Codex (paste in chat)
2. Codex runs:
   ```
   cd E:\agi-research
   $env:GIT_TOKEN = "<your PAT>"
   git push -u origin main
   ```

**SECURITY**: If you give PAT to Codex, it's exposed in the chat
session log. After push, **revoke the PAT** and generate a new one
for any future use.

### Option 3: GitHub CLI auth (if available)

```bash
gh auth login --with-token < <PAT file>
git push -u origin main
```

---

## 4. Verify post-push

After push succeeds:

1. Visit https://github.com/aidless/agi-research
2. Check that:
   - [ ] README.md displays correctly with author 刘泽文
   - [ ] LICENSE shows MIT + copyright 刘泽文
   - [ ] AUTHORS file is visible
   - [ ] Commit history shows author "刘泽文"
   - [ ] 30+ commits all present

---

## 5. After push: announce

Once push is verified, Codex is ready to post (with your approval):
- Twitter announcement (4 versions drafted, ready in community/twitter_joint_ablation.md)
- Discord/Reddit post (drafted in community/discord_joint_ablation.md)

Say "推 Twitter" or "发 Discord" and Codex will execute (still
needs your login or browser cookie for actual posting — Codex
can output the post text + checkboxes you can copy-paste).

---

## 6. If push fails

| Error | Fix |
|-------|-----|
| `Repository not found` | The repo doesn't exist yet — create it on github.com first |
| `Authentication failed` | PAT expired or wrong scopes — regenerate PAT with `repo` scope |
| `Permission denied (publickey)` | SSH key not configured — use Option A (PAT) |
| `Could not resolve host` | Network issue — check internet |
| `non-fast-forward` | Remote has commits not in local — `git pull --rebase origin main` then push |

---

## 7. Quick status

```
Repo URL:    https://github.com/aidless/agi-research
Author:      刘泽文 (Liu Zewen)
Email:       aidless@users.noreply.github.com
License:     MIT (with attribution clause)
Visibility:  Public (essential for IP protection)
Commits:     30+ ready to push
Last commit: 3e709e3 (attribution headers in key files)
Branch:      main
```

---

*Document generated 2026-07-26 by Codex. Once push succeeds, this file
can be removed from the repo (it's a one-time push guide).*
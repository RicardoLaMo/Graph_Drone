# GitHub Branch Protection Setup Guide

This guide helps set up automatic PR review and merge for the Graph_Drone repository.

## What This Does

✅ Automatically runs CI/CD checks (tests, linting)
✅ Auto-approves PRs that pass all checks
✅ Enables auto-merge (SQUASH merge strategy)
✅ Posts automated review comments

## Setup Steps

### Step 1: Enable GitHub Actions (if not already enabled)
1. Go to GitHub repo: https://github.com/RicardoLaMo/Graph_Drone
2. Settings → Actions → General
3. Ensure "Allow all actions and reusable workflows" is enabled

### Step 2: Configure Branch Protection Rules

1. **Go to Settings → Branches**
2. **Add rule for `main` branch** with these settings:

#### Basic Settings
- ✅ Require a pull request before merging
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging

#### Require Status Checks
Add these required status checks:
- `test-and-lint (3.10)`
- `test-and-lint (3.11)`

#### Additional Options
- ✅ Require code reviews
  - Number of approvals: **1**
  - Dismiss stale pull request approvals: Yes
  - Require review from code owners: No
- ✅ Require conversation resolution before merging
- ✅ Allow auto-merge
  - ✓ Auto-merge method: **Squash and merge**

### Step 3: Enable GitHub App Permissions

The workflow uses GitHub Actions to auto-approve. Ensure the workflow has proper permissions:

1. Go to Settings → Actions → General → Workflow permissions
2. Set to: **Read and write permissions**
3. ✅ Allow GitHub Actions to create and approve pull requests

### Step 4: Configure Auto-Merge (Optional but Recommended)

For completely hands-off merging:

1. Go to Settings → General
2. Scroll to "Pull Requests"
3. ✅ Allow auto-merge

---

## How It Works

### When a PR is Created:

1. **GitHub Actions triggers** (`ci-auto-merge.yml`)
2. **Runs checks**:
   - ✅ Python 3.10 & 3.11 tests
   - ✅ Code linting (black, isort)
   - ✅ Unit tests (pytest)
3. **If all pass**:
   - Auto-approves with comment
   - Enables auto-merge (squash strategy)
   - Posts success comment
4. **PR merges automatically** once branch protection checks pass

### If Checks Fail:

- PR remains open (no auto-merge)
- Developer notified of failures
- Manual intervention required

---

## Current Configuration

### Workflow File
- **Location**: `.github/workflows/ci-auto-merge.yml`
- **Triggers**: `pull_request` to main/master branches
- **Python versions tested**: 3.10, 3.11
- **Timeout**: 30 minutes

### Dependencies
- pytest (testing)
- black (code formatting)
- isort (import sorting)
- All dev dependencies from `pyproject.toml`

---

## Troubleshooting

### PR doesn't auto-merge after checks pass
**Check**:
1. Branch protection rules configured ✓
2. "Allow auto-merge" enabled in repo settings ✓
3. GitHub Actions has read/write permissions ✓
4. All required status checks passing ✓

### Tests are failing
**Options**:
1. Fix the issues locally and push
2. Modify `.github/workflows/ci-auto-merge.yml` to be more lenient
3. Mark failing tests as `@pytest.mark.skip` temporarily

### Want to disable auto-merge
**To pause auto-merge**:
1. Comment out the `auto-merge` job in workflow
2. Or set `if: false` on the job
3. Commit and push

---

## Advanced Options

### Option A: More Strict (require all checks + human approval)
- Keep auto-merge disabled
- Use `APPROVE` only as comment feedback
- Manually approve and merge PRs

### Option B: More Lenient (merge even if some tests fail)
Edit `.github/workflows/ci-auto-merge.yml`:
```yaml
needs: test-and-lint
if: always()  # <-- Changes to 'always()' to proceed even on failure
```

### Option C: Auto-merge only for specific branches
Edit the workflow to add branch filters:
```yaml
on:
  pull_request:
    branches:
      - main
      - exp/pc-moe-*  # Only experimental branches auto-merge
```

---

## For This Project (PC-MoE Benchmark)

### Current Setup
- ✅ Workflow created (`ci-auto-merge.yml`)
- ⏳ **NEXT**: Configure branch protection rules manually
- ⏳ **NEXT**: Enable GitHub Actions permissions

### Quick Start
```bash
# 1. Verify workflow is committed
git add .github/workflows/ci-auto-merge.yml
git commit -m "ci: add auto-merge workflow"
git push origin main

# 2. Configure branch protection (via GitHub UI - see Step 2 above)

# 3. Re-run any existing PRs to trigger workflow
```

---

## Need Help?

- **GitHub Actions Docs**: https://docs.github.com/en/actions
- **Branch Protection**: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches
- **Auto-merge**: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/automatically-merging-a-pull-request

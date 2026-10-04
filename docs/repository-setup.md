<!-- SPDX-License-Identifier: Apache-2.0 -->

# Repository Setup

|              | Document data                                  |
|--------------| ---------------------------------------------- |
| Author       | [@pistore](https://github.com/pistore)         |
| Last-Updated | 2026-10-04                                     |
| Status       | Draft                                          |
| Approved-By  | N/A                                            |

This document describes how the GitHub repository and its external services
are configured. It is relevant to repository admins only; developers and
maintainers should follow the workflow in the [README](../README.md)
("Development model" and "Releasing").

## Branches

- `main` is the default branch and contains only released commits.
- `dev` is the integration branch.
- "Automatically delete head branches" is enabled, so feature branches are
  removed when their PR is merged. `dev` and `main` are protected against
  deletion by their rulesets.
- Anyone with write access can push feature branches and open PRs into
  `dev`; only repository admins can update `main` (see Rulesets).
- `hotfix/**` and `backport/**` branches are not covered by a ruleset.

## Rulesets

Configured under Settings → Rules → Rulesets.

### Protect dev (`refs/heads/dev`)

- Restrict deletions; block force pushes; require linear history.
- Require a pull request, 0 approvals, merge method **squash** only;
  dismiss stale approvals on push.
- Required status checks (branch must be up to date): `Version has +dev
  marker`, `Test with Python 3.12` — both from `CI (dev)`.

### Protect main (`refs/heads/main`)

- Restrict updates, so only bypass actors can change `main` — PR merges
  included.
- Restrict deletions; block force pushes. Linear history is **not**
  required.
- Require a pull request, 0 approvals, merge method **merge** only.
- Required status checks (branch must be up to date): `SPDX headers`,
  `Dependency audit`, `Version has no +dev marker`, `Test with Python 3.12`,
  `Test with Python 3.13`, `Test with Python 3.14` — all from `CI (release)`.

`main` accepts merge commits only because squashing or rebasing the
`dev → main` PR would create commits unrelated to `dev`'s own, detaching `dev`
from `main`'s history; every later `dev → main` diff would then re-show
already-released commits.

No approvals are required because only admins can merge into `main`, and
GitHub never counts a PR author's own approval.

### Protect release tags (`refs/tags/v*`)

- Restrict creations, updates and deletions; block force pushes. Only
  bypass actors can create, move or delete a release tag.

### Bypass

All three rulesets have one bypass actor: the **Repository admin** role,
with bypass mode **Always**. Admins are therefore the only ones who can:

- push directly to `dev` (release preparation, starting the next
  development cycle);
- update `main`, including merging the release PR and hotfix PRs;
- create, move or delete `v*` tags.

The bypass applies to every repository admin.

## CI workflows

| Workflow | File | Triggers |
| -------- | ---- | -------- |
| `CI (dev)` | `.github/workflows/ci-dev.yml` | push / PR to `dev` |
| `CI (release)` | `.github/workflows/ci-release.yml` | push / PR to `main`, `hotfix/**`, `backport/**`; manual dispatch |
| `Publish to PyPI` | `.github/workflows/publish.yml` | a GitHub Release is published |

## PyPI publishing

- `civic-digital-twins` on PyPI uses a [Trusted
  Publisher](https://docs.pypi.org/trusted-publishers/) bound to this
  repository, workflow `publish.yml`, environment `publish`. No PyPI token is
  stored in GitHub.
- The `publish` environment (Settings → Environments) has a required
  reviewer, so each publish run waits for manual approval before uploading.

## Codecov

- Coverage is uploaded by `CI (release)` on pushes to `main`, `hotfix/**` and
  `backport/**` (Python 3.12 job only), using the `CODECOV_TOKEN` repository
  secret.
- Coverage thresholds are defined in `codecov.yml`: project coverage may not
  drop by more than 1%, and new lines in a change must be at least 90%
  covered (with a 5% tolerance).
- The README badge shows coverage of the latest upload on `main`.

## Dependabot

Configured under Settings → Code security.

- Dependabot alerts: enabled.
- Dependabot security updates: enabled — Dependabot opens a PR when a
  dependency has a known vulnerability. These PRs are never merged
  automatically.
- Dependabot version updates: not configured (no `.github/dependabot.yml`).

Dependabot scans the default branch, so alerts and security-update PRs are
based on `main`'s `uv.lock` and target `main`. Since `main` only receives
released commits, apply the fix on `dev` instead and close the Dependabot PR.

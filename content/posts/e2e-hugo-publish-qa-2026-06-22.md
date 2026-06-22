---
title: "E2E Hugo Publish QA 2026-06-22"
description: ""
draft: true
slug: "e2e-hugo-publish-qa-2026-06-22"
tags:
---

# E2E Hugo Publish QA 2026-06-22

This short end-to-end QA post verifies a publishing workflow where Cowrite creates a reviewable [GitHub pull request](https://docs.github.com/en/pull-requests) for a [Hugo](https://gohugo.io/) blog repository, rather than deploying changes directly.

## What this validates

* Cowrite can publish via a [GitHub App](https://docs.github.com/en/apps/overview) connection.
* The user can target their own blog repository.
* Publishing creates a pull request for review instead of immediately deploying.
* Performance tracking should start only after the pull request is published and the live canonical URL is provided.

## Expected follow-up

After merge and deployment, the user should paste the final live URL into Cowrite and connect the matching [Google Search Console](https://search.google.com/search-console/about) property.

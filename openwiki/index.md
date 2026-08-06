---
okf_version: "0.1"
---

# Files

- [Architecture](architecture.md) - Core pipeline design of the Pattern CNN-based stock trend prediction system: Pydantic config, data loading, image rendering, CNN model, training loop, and backtest engine.
- [Domain & Strategy](domain.md) - Financial strategy and research findings for the Pattern CNN signal: headline results, cross-sectional slicing, trash-tier intersection, regime overlays, trading-cost analysis, liquid-grid pivot, ETF deployment failures, the $5 price-floor and return-convention corrections (Addenda J-K), and the Tracker paper book.
- [Operations](operations.md) - Operational reference for the Pattern pipeline: YAML configs, run directory layout, image cache, hardware setup, daily publishing job, webapp deployment, data files, and reproducibility.
- [Pattern — OpenWiki Quickstart](quickstart.md) - Entry point for the Pattern repository wiki. Covers what the project is, repository layout, prerequisites, quick-start commands, and links to all documentation sections.
- [Testing](testing.md) - Test coverage for the Pattern pipeline: pixel-exact renderer tests, labeling tests, split tests, what is not tested, and change-oriented guidance for future modifications.
- [Workflows](workflows.md) - End-to-end workflows for the Pattern pipeline: Bloomberg data building, debug and production training, multi-GPU orchestration, backtesting, live scoring, price splice, Tracker paper book, and daily publishing job.

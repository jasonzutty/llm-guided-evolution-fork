Guide for setting up Unit Testing UI
====================================

Following the guide at: https://drive.google.com/drive/folders/1rBBtos9HKWPo1zWpU71wcISqkbQsKbjm?usp=sharing

Report History Publishing Architecture
======================================

Purpose
-------

This workflow publishes CI test report history to GitHub Pages after the
``Submit Slurm Job`` workflow completes. It collects the generated XML test
report, optional documentation suggestions, run metadata, and stores them on
the ``gh-pages`` branch so each CI run has a permanent report record.

Architecture Overview
---------------------

::

   Submit Slurm Job Workflow
            |
            v
   Publish XML Report History Workflow
            |
            +--> Download junit-report artifact
            |
            +--> Download optional doc-suggestions artifact
            |
            +--> Collect run metadata
            |
            v
   Create per-run folder in gh-pages
            |
            v
   Update runs/runs_list.json
            |
            v
   Commit and push to gh-pages
            |
            v
   GitHub Pages report history

Main Components
---------------

Trigger
   Runs automatically when the ``Submit Slurm Job`` workflow completes.

Artifact Downloader
   Uses the GitHub REST API to download the ``junit-report`` artifact and the
   optional ``doc-suggestions`` artifact from the triggering workflow run.

Metadata Collector
   Records the run time, actor, run ID, run number, commit SHA, commit title,
   commit message, Actions URL, and commit URL.

Pages Publisher
   Copies ``report.xml``, ``doc_suggestions.json``, and ``metadata.json`` into
   a per-run directory under ``gh-pages/runs/``.

Index Builder
   Updates ``runs/runs_list.json`` so the report viewer can discover available
   historical runs.

Git Publisher
   Commits and pushes the updated report history to the ``gh-pages`` branch.

External Interactions
---------------------

GitHub Actions
   Provides the workflow completion event and run metadata.

GitHub REST API
   Provides access to artifacts from the completed workflow run.

GitHub Pages
   Hosts the historical report files from the ``gh-pages`` branch.

Self-hosted Runner
   Executes the publishing workflow.

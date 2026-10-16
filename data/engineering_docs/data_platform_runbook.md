---
title: Data Platform — Pipeline Operations Runbook
access: engineering, platform
revision: 2026-05
---

# Data Platform — Pipeline Operations Runbook

## Telemetry ingestion pipeline
Machine telemetry is published to the Kafka topic machine-telemetry, keyed by machine serial number.
The ingestion job writes raw events to the bronze table telemetry_raw, partitioned by event date.
The hourly Airflow DAG telemetry_hourly builds the silver table telemetry_clean after the hour is complete.

## When the telemetry_hourly DAG fails
First check the Airflow task log for the failed task and the consumer lag of the ingestion job.
All tasks are idempotent and overwrite their hourly partition, so clearing the failed task and rerunning it is safe.
If the source data was wrong, fix it in bronze and backfill the affected hours with max_active_runs set to 2.

## Service level objectives
The silver table telemetry_clean must be complete within 30 minutes after each hour on 99 percent of hours.
Alerts go to the platform on-call channel when the freshness check fails twice in a row.

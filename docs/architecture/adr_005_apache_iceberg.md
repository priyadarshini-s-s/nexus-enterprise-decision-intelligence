# ADR-005: Apache Iceberg as the NEXUS Table Format

## Status

Accepted

## Context

NEXUS requires a reliable table-management layer on top of columnar data storage.

The project uses Apache Parquet for efficient columnar storage, but raw Parquet files alone do not provide a complete table-level transaction and metadata-management abstraction.

NEXUS also requires support for:

- Atomic table updates
- Snapshot history
- Time travel
- Schema evolution
- Metadata-driven file management
- Compatibility with Apache Spark
- A path toward scalable object-storage deployment

## Decision

NEXUS will use Apache Iceberg as the table format for curated and analytical datasets where table-level transactional and historical capabilities are required.

Apache Parquet remains the underlying physical file format.

The development environment uses an Iceberg Hadoop catalog backed by the local NEXUS warehouse.

The production architecture is expected to use object storage with an appropriate Iceberg catalog.

## Why Iceberg

### 1. Snapshot-based table management

Iceberg maintains table snapshots instead of treating a directory of Parquet files as the complete source of truth.

This allows NEXUS to inspect table history and reproduce previous table states.

### 2. Time travel

Historical snapshots can be queried without manually maintaining separate copies of the dataset.

This is useful for:

- Reproducible analytics
- Debugging
- Model-training datasets
- Auditability
- Rollback investigations

### 3. Separation of table and file management

Parquet provides the physical storage format.

Iceberg provides the table abstraction and metadata layer.

This separation allows NEXUS to manage large analytical datasets without coupling business logic directly to individual files.

### 4. Spark integration

NEXUS already uses Apache Spark for distributed data processing.

Iceberg integrates directly with Spark SQL and DataFrame workflows, allowing the same processing engine to write and query analytical tables.

## Evidence from NEXUS

The Olist Orders Silver dataset contains 99,441 real order records.

The first Iceberg snapshot contained 99,441 records.

A second Iceberg snapshot was subsequently created through an overwrite operation.

The table retained both snapshots and allowed the first snapshot to be queried through Iceberg time travel.

The current and historical versions both returned 99,441 records.

The table uses Iceberg format version 2 with Parquet data files and ZSTD compression.

## Alternatives Considered

### Plain Parquet

Advantages:

- Simple
- Lightweight
- Excellent analytical performance
- Widely supported

Disadvantages:

- No native table-level snapshot management
- No built-in time-travel abstraction
- More application responsibility for safe updates
- Metadata management becomes increasingly complex

Decision:

Rejected as the primary table-management layer, but retained as the physical file format underneath Iceberg.

### Delta Lake

Advantages:

- ACID transactions
- Time travel
- Strong Spark integration
- Mature lakehouse capabilities

Disadvantages:

- Would introduce another table-format ecosystem
- Iceberg better aligns with the project's goal of demonstrating an open table-format architecture

Decision:

Not selected for NEXUS.

### Apache Hudi

Advantages:

- Strong incremental-processing capabilities
- Upserts and record-level data management
- Good support for streaming-oriented workloads

Disadvantages:

- More specialized than required for the current NEXUS analytical workloads

Decision:

Not selected as the primary table format.

## Consequences

### Positive

- Historical table states become queryable
- Better reproducibility of analytical datasets
- Clear separation between storage and table management
- Strong Spark integration
- Provides a foundation for scalable lakehouse architecture

### Negative

- Adds metadata and catalog complexity
- Requires Iceberg-compatible tooling
- Local development requires additional configuration
- Snapshot maintenance and table optimization must eventually be managed

## Development Configuration

Current development stack:

- Apache Spark 4.1.1
- Apache Iceberg 1.12.0
- PySpark
- Parquet
- HadoopCatalog
- Local filesystem warehouse

## Production Direction

The local Hadoop catalog is a development configuration.

A production NEXUS deployment should use:

- Object storage
- A production-grade Iceberg catalog
- Catalog-level access control
- Snapshot retention policies
- Compaction/data-file maintenance
- Monitoring and lineage integration

## Decision Summary

NEXUS adopts Apache Iceberg as the table-management layer for curated analytical datasets, while retaining Parquet as the underlying columnar storage format.

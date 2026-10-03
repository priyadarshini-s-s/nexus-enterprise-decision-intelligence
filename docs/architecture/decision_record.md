# ADR-001 — Multi-Dataset Real-Data Strategy

## Status
Accepted — v0.1

## Decision

NEXUS will use multiple legitimate real-world datasets rather than forcing all capabilities into one dataset.

### Primary business spine
Olist Brazilian E-Commerce Public Dataset.

### Specialized domains
- dunnhumby Complete Journey — marketing / campaign-response analysis
- RetailRocket — digital journey / behavioral recommendation
- OTTO — large-scale session-based recommendation

## Why

No single selected public dataset currently provides the full set of capabilities required by NEXUS while preserving legitimate business semantics.

The architecture therefore separates source domains and unifies them at the semantic/decision layer.

## Rejected alternatives

### One-dataset architecture
Rejected because it would force unsupported assumptions or remove important capabilities.

### Artificial cross-dataset customer joins
Rejected because similar-looking identifiers do not establish identity.

### Synthetic records to fill missing capabilities
Rejected because the project requirement is to use real observed data.

### Big Data technology everywhere
Rejected because distributed processing should be justified by workload and scale.

## Engineering principle

Use the simplest technology that is appropriate for the actual workload, while keeping the platform extensible for genuinely large datasets.

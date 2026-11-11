# Data provenance, access and verification scope

The repository contains analysis code and protocols only. Frozen observations, GT files, organization-level outcomes, source-response caches and per-host plans are excluded. No third-party access to the study-specific inputs is currently offered. Redistribution rights for the archived third-party observations have not been established.

## Input lineage
Asset-projection metrics use GT names, IP presence, ports and CPEs. The gt_sources field marks port-contributing services; it is not a complete provenance ledger. The code supports multiple domain-discovery sources, DNS, InternetDB and Censys enrichment. These code paths do not establish historical execution or permission. Some merged ports lack attribution in the retained service-specific fields.

Localization and alert recall use injected-host truth labels. Candidate and score features use archived evidence events, claims, decisions and the discovery manifest. Offline planning also requires DAG records, action mappings and cost configuration. Author-created derived statistics are distinct from provider responses; their sharing status must be evaluated for the actual artifact and applicable agreements.

## Provider access
Shodan InternetDB documentation: https://blog.shodan.io/introducing-the-internetdb-api/
Shodan terms: https://static.shodan.io/legal/terms.html
Censys research access: https://docs.censys.com/docs/research-access-to-censys-data
Censys contact: https://www.censys.com/contact/
These are provider access/information channels, not repositories of the exact frozen study inputs. Current documentation does not establish historical rights. Research access and permission to redistribute must be distinguished.

## What can be checked
Readers can inspect the sensitivity-analysis logic, declared budgets and protocols. This add-on requires original project modules that are not included, plus private frozen inputs. It cannot independently reproduce candidate coverage, detector scores, alert recall, asset metrics, planner performance or the numerical CI. No result dataset or synthetic replacement is included. Upstream re-querying is a new collection and is not guaranteed to reproduce the retained snapshot.

A separate reduced aggregate artifact has been evaluated locally for the paired mean difference and empirical-bootstrap CI. It is not included, published, approved for release, or available on request through this repository. No access permission is implied by this README.

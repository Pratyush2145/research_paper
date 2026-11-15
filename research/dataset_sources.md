# Dataset Sources and Provenance Specification

This document records the official provenance, retrieval metadata, licensing, and cryptographic verification for all research datasets used in this project.

---

## 1. Dataset A — LegitPhish (Primary Dataset)

- **Official Dataset Title:** LegitPhish Dataset
- **Official Repository:** Mendeley Data
- **Official Source URL:** https://data.mendeley.com/datasets/hx4m73v2sf/2
- **Direct Download URL:** `https://data.mendeley.com/public-files/datasets/hx4m73v2sf/files/0ed51e4d-d160-4047-b650-3f0801eea4aa/file_downloaded`
- **DOI:** [10.17632/hx4m73v2sf.2](https://doi.org/10.17632/hx4m73v2sf.2)
- **Contributors / Authors:** Rachana Potpelwar, Uday Kulkarni, Jaishri Waghmare
- **Institution:** Shri Guru Gobind Singhji Institute of Engineering and Technology
- **Publication / Release Date:** 22 May 2025 (Version 2)
- **Verified License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
  - Verified via Mendeley Data schema metadata and linkset headers (`http://creativecommons.org/licenses/by/4.0`).
  - Attribution requirement: Cite dataset creators and DOI.
- **Download Date:** 2026-11-14
- **Local Storage Path:** `data/raw/legitphish/url_features_extracted1.csv`
- **Original Downloaded Filename:** `url_features_extracted1.csv`
- **File Size:** 8,963,838 bytes (~8.55 MiB / 8.96 MB)
- **Cryptographic SHA-256 Checksum:** `8685e7901c3b4e73eb43c106cb166cc932dd14f76960a032c713c2c32f09dbcd`
  - *Verification:* Matches the official SHA-256 digest published in the Mendeley Data public API metadata.
- **Download / Extraction Notes:** Downloaded directly from the official Mendeley Data public API storage endpoint without modification.

---

## 2. Dataset B — PhiUSIIL (External Validation Dataset)

- **Official Dataset Title:** PhiUSIIL Phishing URL (Website)
- **Official Repository:** UCI Machine Learning Repository
- **Repository Identifier:** UCI Dataset ID 967
- **Official Source URL:** https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
- **Direct Download URL:** `https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip`
- **Associated Publication DOI:** [10.1016/j.cose.2023.103545](https://doi.org/10.1016/j.cose.2023.103545)
- **Associated Paper:** Arvind Prasad and Shalini Chandra (2024). *"PhiUSIIL: A diverse security profile empowered phishing URL detection framework."* Computers & Security, 137, 103545.
- **Verified License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
  - Verified via official UCI dataset portal metadata (`https://creativecommons.org/licenses/by/4.0/legalcode`).
  - Attribution requirement: Cite paper and UCI repository.
- **Date Donated to UCI:** 2024-03-04
- **Download Date:** 2026-11-14
- **Local Storage Path:** `data/raw/phiusil/`
- **Original Downloaded Archive:**
  - Filename: `phiusiil+phishing+url+dataset.zip`
  - Archive Size: 15,376,007 bytes (~14.66 MB)
  - Archive SHA-256: `0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a`
- **Extracted Raw Data File:**
  - Filename: `PhiUSIIL_Phishing_URL_Dataset.csv`
  - Uncompressed Size: 56,854,345 bytes (~54.22 MB)
  - Uncompressed SHA-256: `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`
- **Download / Extraction Notes:** The zip archive was retrieved directly from UCI servers and extracted once into `data/raw/phiusil/`. The raw CSV file remains unmodified.

---

## 3. Provenance and Storage Integrity Policy

1. **Immutability of Raw Data:** All files under `data/raw/` are strictly read-only for research pipelines. No cleaning, preprocessing, or transformation scripts are permitted to overwrite or mutate files in `data/raw/`.
2. **Processed Data Separation:** Any filtered subsets, deduplicated variants, or split datasets created in future phases will be written strictly to `data/processed/`.
3. **No External Network Calls to URLs:** Under no circumstances will any URL contained in either dataset be queried, resolved via DNS, or visited over HTTP/HTTPS.


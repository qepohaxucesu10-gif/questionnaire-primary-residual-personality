# Questionnaire-Primary Residual Personality Prediction

This repository contains reproducibility code and non-sensitive derived
materials for the study:

“Testing a Dual-Process-Inspired Questionnaire-Primary Residual Framework for
Personality Prediction: An Exploratory Study with Static Line Drawings.”

The reported analyses did not find stable incremental predictive value from
the static line-drawing modality under the present design.

## Repository contents

- src/: analysis, model, preprocessing, and summary code
- configs/: final analysis configuration
- folds/: outer-fold definitions
- selected_items/: fold-specific selected-item information
- derived_results/: non-sensitive derived analysis outputs
- docs/: input-schema and reproduction documentation
- environment/: package requirements and recorded software versions
- examples/synthetic_smoke_test/: synthetic pipeline execution checks

## Data availability

Participant-level questionnaire responses, static drawings, and teacher-provided
observer records are not included in this repository because of privacy,
consent, and ethical restrictions.

The repository provides:

- analysis code
- preprocessing code
- fold definitions
- selected-item lists
- analysis configuration
- non-sensitive derived outputs
- input schema documentation
- synthetic smoke-test data

Requests for restricted participant-level data can only be considered subject
to the applicable ethical approval, consent conditions, and institutional
requirements.

## Model summary

The Questionnaire Tower uses 31–32 fold-specific questionnaire responses, 48
hidden units, and 16 normalized factor predictions.

The TinyCNN visual residual model has 19,648 trainable parameters. The complete
architecture has 21,968–22,016 trainable parameters, depending on whether 31 or
32 questionnaire items are selected in a fold. Visual residual learning uses
cross-fitted questionnaire residual targets within 5×5 repeated
participant-level cross-validation.

## Environment

The retained analysis used Python 3.10.19, PyTorch 2.0.0+cu117, and torchvision
0.15.0+cu117. Other recorded requirements are listed in
environment/requirements.txt; recorded software information is in
environment/software_versions.md.

The restricted-data input schema is documented in docs/private_input_schema.md.
It specifies the governed inputs required for a restricted rerun and does not
include participant-level records.

## Quick start

Install the recorded dependencies:

    pip install -r environment/requirements.txt

Run the synthetic smoke tests:

    python examples/synthetic_smoke_test/smoke_test.py
    python examples/synthetic_smoke_test/analysis_smoke_test.py

Synthetic example data are not study data and do not reproduce the reported
numerical manuscript results; they are provided only to test pipeline execution.

## Reproducibility note

Neural-network retraining may exhibit small hardware-dependent numerical
variation. An independent end-to-end rerun reproduced the substantive findings
with only minor numerical differences in the visual branch and nested
scale-selection frequency.

No license has been selected for this release.

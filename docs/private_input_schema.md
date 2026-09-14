# Private input schema

`study_id` is a publication-only pseudonymous identifier. Numeric source IDs must not be treated as anonymous, and the private crosswalk must not be published.

## Questionnaire data

One row per participant with `study_id`, response columns `X_Q1` through `X_Q187`, and reference columns `Y_A`, `Y_B`, `Y_C`, `Y_E`, `Y_F`, `Y_G`, `Y_H`, `Y_I`, `Y_L`, `Y_M`, `Y_N`, `Y_O`, `Y_Q1`, `Y_Q2`, `Y_Q3`, and `Y_Q4`. Response encoding is A=2, B=1, C=0. Reference scores are Sten values from 1 through 10.

## Image mapping

A CSV with `study_id` and `image_file`. `image_file` is a relative filename within the supplied image directory. Allowed extensions are `.png`, `.jpg`, `.jpeg`, `.tif`, and `.tiff`. The primary model input is the 128×128 clean-gray PNG produced by the released preprocessing procedure.

## Teacher criterion

A CSV with `study_id` and `teacher_trait_set`, where the latter contains three semicolon-separated factor labels from the fixed 16-factor order. Do not include teacher names, class names, or private linkage fields.

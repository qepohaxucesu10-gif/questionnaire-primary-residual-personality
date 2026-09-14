# Reproduction instructions

Install the documented environment, prepare governed restricted inputs using `private_input_schema.md`, and use the released outer-fold file. Run `src/run_nested_scale_analysis.py` to produce participant-level predictions, fold-local selected-item lists, fixed-scale results, and nested scale selections. Run `src/summarize_primary_results.py` on that prediction file to obtain the primary manuscript statistics. Run `src/run_teacher_external_analysis.py` with the same predictions and the governed teacher-criterion file.

Example: `python src/run_nested_scale_analysis.py --questionnaire-data <file> --image-dir <directory> --image-map <file> --fold-assignments folds/outer_fold_assignments.csv --output-dir <analysis_output>`

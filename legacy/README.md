# Original course submission

`transformer_sml_ppt.ipynb` and `NASA-C_MAPSS.pdf` are the project as first submitted. They are kept for the
record. Their numbers come from a pipeline with these defects, all fixed in `cmapss/`:

- Unit numbers restart in every subset, and the notebook concatenated FD002–FD004 before grouping by unit.
  609 engines became 260, so 17,101 of 126,988 training windows (13.5%) spliced two engines together.
- The train/validation split was random over windows. Every validation window came from an engine that was
  also in training, so the validation score measured memorisation.
- Targets were uncapped (up to 542 cycles) and sensors were scaled globally across six operating conditions.
- Test engines shorter than the 50-cycle window were skipped (47 across FD002–FD004), and the cell that
  produced the reported test MAE has no code, so those numbers cannot be reproduced.

Use the results in `results/` instead.

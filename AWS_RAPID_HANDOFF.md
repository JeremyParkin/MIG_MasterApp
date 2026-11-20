# Staged AWS handoff for Rapid Labeling

Use the tested follow-on commit SHA supplied with this handoff. Its parent is `194d39f`. The three existing release commits remain in its history. Deploy the same revision for both AWS stages and record the Git SHA, Python version, smoke results, and resource measurements in the deployment ticket.

## Community Cloud smoke

Use the full tested revision with Python 3.13 and `MIG_ENABLE_RAPID_LABELING=1` in the app process environment. An unset switch also enables Rapid. With approved coverage and provider access, confirm the Rapid route, both AI passes, independent Spot Checks decisions, Save & Load, Insights defaults, and the selected fields in a rebuilt XLSX. This host check can run ahead of the AWS enabled stage.

## AWS stage 1 · Rapid disabled

1. Verify Python 3.13.1, synchronize from the committed Pipenv lock, and run `python -m pip check` in the app environment.
2. Deploy the full tested revision with `MIG_ENABLE_RAPID_LABELING=0` in the Streamlit service environment and restart. Check HTTP health, established Sentiment and Tagging navigation, and that the direct `/Rapid_Labeling` route cannot render Rapid.
3. Restore an approved snapshot containing established and Rapid work. Build XLSX and Word. Confirm established labels and narratives are present and Rapid sheets, fields, and narratives are omitted. Verify the saved Rapid decisions are still in the snapshot.
4. Take a representative large input through cleaning, established review, and XLSX export. Record input size, baseline and peak RSS, peak CPU, any worker restart or OOM event, and export completion. Confirm the worker stayed up. Restart the service and recheck health and established navigation before considering stage 2.

## AWS stage 2 · Rapid enabled

Run only after stage 1 passes and the pilot owner authorizes a small sample and live provider calls. Set `MIG_ENABLE_RAPID_LABELING=1` on the same revision and restart. Verify HTTP health, navigation, the direct route, and access to the existing OpenRouter and OpenAI credentials.

1. Prepare an approved report with Five-way and All applicable selected. Run Jev and Luna on a small sample. Confirm usable labels and review flags, and check for provider errors.
2. In Spot Checks, save different three-way and five-way decisions for the same story and an all-applicable tag correction. Save a session file and reopen it. Confirm both sentiment decisions survived, tag selection persisted, and Insights opens on Five-way and All applicable.
3. Build a standard XLSX and check the selected final sentiment, final tags, relevance, and model-derived spread in `RAPID LABELING SAMPLE` and every matching in-scope `CLEAN TRAD` row. Check the selected modes in metadata. Build with audit and verify the selected values match and the alternate outputs, model evidence, and human decisions are present.
4. Change a report preference and confirm Download withholds the old file until rebuilt. If using a scope-incompatible saved sample, verify Download omits Rapid outputs while the saved work remains intact.

If the pilot must stop, set `MIG_ENABLE_RAPID_LABELING=0` on the same SHA, restart, and verify health, the blocked direct route, and established exports again.

## Open limitations

- Local workflow verification used mocked provider responses. AWS credentials, provider behavior, resource headroom and restart behavior still need host evidence.
- Three-way and five-way decisions can intentionally differ. Numeric scoring policy and reconciliation are separate product decisions. The matched analyst pilot still needs quality, review-time, and cost measurements.
- Download checks Rapid Word freshness and prompts analysts to regenerate stale observations. Established narrative inputs remain outside that check. Word is outside the enabled-stage pilot gate.
- Reset clears human decisions. Save a checkpoint first. A saved Rapid sample from a different scope requires restoring its original scope or explicitly preparing a replacement.

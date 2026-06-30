# Publication Checklist

This checklist tracks what is ready for public release and what still needs author confirmation before journal submission.

## Completed in This Release Package

- Clean manuscript package created in `manuscript/`.
- LaTeX source, bibliography, latest PDF, and figure assets copied into the repository.
- Visible empty TODO subsections were removed from the release copy of `main.tex`.
- Acknowledgments placeholder was replaced with a concrete dataset/challenge acknowledgment.
- Generative AI / AI-assisted technology declaration was added to the release copy.
- Repository README was rewritten for public visitors.
- Citation metadata was added in `CITATION.cff`.
- LaTeX, model checkpoint, dataset, cache, and output patterns were added to `.gitignore`.
- Draft cover letter was added as `manuscript/cover_letter.md`.

## Must Confirm Before Journal Submission

- Target journal and article type.
- Latest author instructions for formatting, reference style, figure placement, and required statements.
- Whether the journal requires the official template, blinded review, separate title page, or separate figure files.
- Whether the special issue named in the original source comments is still open and appropriate.
- Final public code URL and optional archival DOI.
- Final data availability statement, especially if split files or preprocessing scripts will be released.
- Final license for the repository.

## Manuscript Content Risks to Review

- The reported validation split is random patch-level. The discussion already notes that spatially adjacent patches may cross train/validation boundaries; keep this limitation visible.
- The official C2Seg test labels are not public. Avoid language implying official test-set performance.
- Several references are recent 2024-2025 papers. Verify DOI, publication status, and bibliographic metadata before submission.
- If qualitative maps, CAM, t-SNE, or per-class IoU figures are required, add them as complete sections with figure files and captions. Empty headings should not be restored.
- Confirm that the PMRG implementation named in the paper maps to the intended release config, likely `cxup_4b_BW_PMRG_v2_loss.yml`.

## Code Release Tasks

- Add a `LICENSE` file after choosing the project license.
- Check PaddleSeg/PaddlePaddle license compatibility and keep upstream notices.
- Remove or document machine-specific absolute paths in configs before public release.
- Provide train/validation split files if allowed by the dataset license, or document how they were generated.
- Consider publishing trained weights separately if file size is large.
- Tag a release after the manuscript, README, and citation metadata are final.

## Suggested Release Flow

1. Choose and add a license.
2. Update dataset paths in configs or add example configs with placeholder paths.
3. Run a smoke test for import, config loading, and one validation command if weights are available.
4. Compile `manuscript/main.tex` from inside `manuscript/`.
5. Run a final citation/reference cross-check.
6. Create a GitHub release and archive it with Zenodo if a DOI is needed.
7. Update `CITATION.cff` with the final software DOI or article DOI.

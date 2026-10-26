# Rejected candidate artifacts

These initial fits contained nonfinite coefficients from unsanitized legacy
feature exports. Their ranking metrics and search comparisons are invalid and
must not satisfy a release gate. Returned paths still require independent replay.
The corrected candidate is `../feature-candidate-finite.json`, trained with
finite, bounded and scaled features. Production models were never replaced.

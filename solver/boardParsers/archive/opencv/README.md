# Archived OpenCV experiments

These unfinished experiments were removed from active parser selection on
October 22, 2026. They are retained for research; production and the solver CLI
never import them. `openCVBoardParser.py` assumes exactly 6×6 and fixed color
thresholds, and `learnCV.py` is an introductory scaffold. Neither has a validated
screenshot corpus or a supported fallback integration.

Daily puzzles now use `apps/api/puzzle_catalog.py`. Gemini handles screenshot
fallbacks only after capture-date validation in the API. Standalone screenshot
CLI input uses Gemini directly.

To revisit these experiments, use an isolated environment with `opencv-python`
and `numpy`; adapt grid dimensions, colors, and confidence handling before
considering promotion. The archived import of `fallbackBoardParser` reflects the
old standalone layout and is not supported. Do not add OpenCV to runtime
requirements merely to run these archived files.

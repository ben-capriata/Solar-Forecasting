OVERLEAF UPLOAD INSTRUCTIONS
============================

1. Overleaf -> New Project -> Upload Project -> select this .zip
2. Set the main document to main.tex (Menu -> Main document)
3. Compiler: pdfLaTeX (the default). Do not switch to XeLaTeX or LuaLaTeX.
4. Press Recompile. It should build to exactly 5 pages with no errors:
   4 pages of content + 1 page of references, as the FAQ requires.

If the bibliography shows [?] instead of numbers, hit Recompile a second
time. Overleaf needs two passes to resolve BibTeX citations.


FILL THESE IN BEFORE SUBMITTING
===============================

Five placeholders remain in main.tex. Search for the square brackets.

  Line  20   [YOUR NAME] and [ROLL NO]
  Line  24   Trimester [7/8/9]
  Line  27   [your-email]@op.iitg.ac.in
  Line ~438  \url{[GITHUB URL]}     <- repo must be PUBLIC before you submit
  Line ~442  \url{[YOUTUBE URL]}    <- set to Public or Unlisted, never Private


FILE MANIFEST
=============

  main.tex        The report. The only file you need to edit.
  refs.bib        11 references. All real, refereed sources.
  spconf.sty      The IIT Guwahati / IEEE two-column style. Do not edit.
  IEEEbib.bst     Bibliography style. Do not edit.
  figures/        The five figures used, at 150 dpi.
                    fig4_seasonal.png    -> Fig. 1, RMSE by month
                    fig2_cloudy_day.png  -> Fig. 2, most clouded test day
                    fig1_clear_day.png   -> Fig. 3, clearest test day
                    fig3_error_by_hour.png -> Fig. 4, MAE by hour
                    fig6_loss_curve.png  -> Fig. 5, training history


NOTES
=====

The original template loaded siunitx, stfloats and tikz. This report uses
none of them, so those \usepackage lines were removed. amssymb was added
because the report uses \mathbb. Everything else follows the template.

Do not let the report exceed 4 pages of content. If you add text, cut
elsewhere. The FAQ states the limit as a hard maximum.

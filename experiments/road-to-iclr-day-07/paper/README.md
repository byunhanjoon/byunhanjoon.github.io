# ICLR 2027 submission source

This directory uses the official ICLR 2027 style files downloaded from:

https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip

The anonymous source is `main.tex`; `\\iclrfinalcopy` must remain disabled for
submission. The official 2027 author guidelines require at most nine pages of
main text, double-blind anonymity, and an AI-use statement:

https://iclr.cc/Conferences/2027/AuthorGuidelines

Compile from this directory:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The checked `main.pdf` was compiled with Tectonic 0.17.0. It has 10 total
pages: references begin on page 8 and the appendix begins on page 9, leaving
the main text inside the nine-page submission limit. The build has no undefined
citations, undefined references, or overfull boxes.

The style and bibliography-style files are unmodified official assets.
`official_template.tex` is retained as a provenance copy.

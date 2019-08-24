#!/bin/bash
set -e

pandoc \
    -t beamer slides/slides.md \
    -V theme:metropolis \
    --filter pandoc-citeproc \
    --bibliography slides/bibliography.bib \
    -o slides.pdf

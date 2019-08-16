#!/bin/bash
set -e

cd slides
pandoc -t beamer slides.md -V theme:metropolis -o ../slides.pdf

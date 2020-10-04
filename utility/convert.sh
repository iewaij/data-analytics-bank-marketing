#!/bin/bash

for filename in code/*.ipynb
do 
jupytext --to md --update-metadata '{"jupytext": {"notebook_metadata_filter":"-all", "cell_metadata_filter":"-all"}}'\
 -o $(echo "docs/content/$filename" | sed 's_/code__g'| sed 's_.ipynb_.md_g') \
 $filename
done

# Uncomment below to use standard template, some variables in manual.yaml won't be used.
pandoc docs/content/*.md -o docs/manual.pdf --pdf-engine=xelatex --metadata-file=docs/format/manual.yaml --template=docs/template/manual.latex

# Uncomment below to use eisvogel template, some variables in manual.yaml won't be used.
pandoc docs/content/*.md -o docs/manual_eisvogel.pdf --pdf-engine=pdflatex --metadata-file=docs/format/manual.yaml --template=docs/template/eisvogel.latex

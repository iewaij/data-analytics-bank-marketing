#!/bin/bash

cd ../code
for filename in *.ipynb
do 
jupytext --to md --update-metadata '{"jupytext": {"notebook_metadata_filter":"-all", "cell_metadata_filter":"-all"}}'\
 -o $(echo "../doc/content/$filename" | sed 's/.ipynb/.md/g') \
 $filename
done

cd ../doc

# Uncomment below to use standard template, some variables in manual.yaml won't not used.
pandoc content/*.md -o manual.pdf --pdf-engine=xelatex --metadata-file=format/manual.yaml --template=template/manual.latex

# Uncomment below to use eisvogel template, some variables in manual.yaml won't not used.
# pandoc content/manual.md -o manual.pdf --pdf-engine=pdflatex --metadata-file=format/manual.yaml --template=template/eisvogel.latex

#!/bin/bash

cd docs/content
pandoc *.md --pdf-engine=xelatex --metadata-file=../format/manual.yaml --template=../template/manual.latex -o ../bank_marketing.pdf 
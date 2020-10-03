#!/bin/bash

jupytext --to md --update-metadata '{"jupytext": {"notebook_metadata_filter":"-all", "cell_metadata_filter":"-all"}}' \
../code/1_data_preparation.ipynb \
-o ../doc/1_data_preparation.md

jupytext --to md --update-metadata '{"jupytext": {"notebook_metadata_filter":"-all", "cell_metadata_filter":"-all"}}' \
../code/2_exploratory_data_analysis.ipynb \
-o ../doc/2_exploratory_data_analysis.md

jupytext --to md --update-metadata '{"jupytext": {"notebook_metadata_filter":"-all", "cell_metadata_filter":"-all"}}' \
../code/3_tree_based_models.ipynb \
-o ../doc/3_tree_based_models.md

jupytext --to md --update-metadata '{"jupytext": {"notebook_metadata_filter":"-all", "cell_metadata_filter":"-all"}}' \
../code/4_nerual_network.ipynb \
-o ../doc/4_nerual_network.md

jupytext --to md --update-metadata '{"jupytext": {"notebook_metadata_filter":"-all", "cell_metadata_filter":"-all"}}' \
../code/5_support_vector_machine.ipynb \
-o ../doc/5_support_vector_machine.md

cd ../doc
cat *.md > manual.md
pandoc manual.md -o manual.pdf --pdf-engine=xelatex

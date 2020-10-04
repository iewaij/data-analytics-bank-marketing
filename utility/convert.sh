#!/bin/bash

jupytext --to md --update-metadata '{"jupytext": {"notebook_metadata_filter":"-all", "cell_metadata_filter":"-all"}}' \
../code/1_data_preparation.ipynb \
-o ../doc/content/1_data_preparation.md

jupytext --to md --update-metadata '{"jupytext": {"notebook_metadata_filter":"-all", "cell_metadata_filter":"-all"}}' \
../code/2_exploratory_data_analysis.ipynb \
-o ../doc/content/2_exploratory_data_analysis.md

jupytext --to md --update-metadata '{"jupytext": {"notebook_metadata_filter":"-all", "cell_metadata_filter":"-all"}}' \
../code/3_tree_based_models.ipynb \
-o ../doc/content/3_tree_based_models.md

jupytext --to md --update-metadata '{"jupytext": {"notebook_metadata_filter":"-all", "cell_metadata_filter":"-all"}}' \
../code/4_nerual_network.ipynb \
-o ../doc/content/4_nerual_network.md

jupytext --to md --update-metadata '{"jupytext": {"notebook_metadata_filter":"-all", "cell_metadata_filter":"-all"}}' \
../code/5_support_vector_machine.ipynb \
-o ../doc/content/5_support_vector_machine.md

cd ../doc/content
cat 1_data_preparation.md 2_exploratory_data_analysis.md 3_tree_based_models.md 4_nerual_network.md 5_support_vector_machine.md > manual.md
cd ../
pandoc content/manual.md --pdf-engine=xelatex --metadata-file=format/manual.yaml --template=template/manual.latex -o manual.pdf

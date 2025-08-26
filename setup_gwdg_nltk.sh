#!/bin/bash -i
module load miniforge3

# Main script execution
check_conda_installed
check_conda_env

set -e

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

echo "Activating conda environment 'dnlp'..."
conda activate dnlp

echo $CONDA_DEFAULT_ENV

# Install packages
conda install -y "nltk" 

# Download model on login-node
python - <<EOF
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk import pos_tag as treebank_pos_tag


nltk.download('punkt')
nltk.download('punkt_tab') ####
nltk.download('averaged_perceptron_tagger_eng') ####
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('sentiwordnet')
EOF


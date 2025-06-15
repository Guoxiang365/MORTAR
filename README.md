# MORTAR
MORTAR: Multi-turn Metamorphic Testing for LLM-based Dialogue Systems

This repository is the implementation of article "MORTAR: Multi-turn Metamorphic Testing for Dialogue Systems". 

Following procedures are implemented:

1. Pre-extract information from original dialogue dataset
2. Add dialogue-level perturbation and check context information to form MT dataset
3. Run testing on SUTs
4. Result analysis 


Requirements:
- Original dataset
- Groq API Key (Can be obtained on Groq, link: https://groq.com/)
- python env


## python env config

### Step 1: Create and Activate Python Environment

Create a new Conda environment with Python 3.9:
```bash
conda create -n py39 python=3.9
conda activate py39
```

### Step 2: Install Other Required Packages

Install the following packages using pip:
```bash
pip install transformers
pip install -U sentence-transformers
pip install groq
pip install spacy==3.7.5
pip install matplotlib
pip install nltk
pip install networkx
pip install matplotlib-venn
```

### Step 3: Download SpaCy Models

Download the required SpaCy models:
```bash
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_trf
```

### Step 4: Install Coreferee

Follow these steps to install Coreferee from source

1. Clone the Coreferee repository:
```bash
git clone https://github.com/richardpaulhudson/coreferee
```
2. Navigate to the coreferee folder:
```bash
cd coreferee
```
3. Modify the `setup.cfg` file to increase the range of supported SpaCy versions to 3.8.
```ini
[options]
install_requires =
    spacy>=3.0.0,<3.8.0
```

4. Modify the `./coreferee/lang/en/config.cfg` file for the English language by adding the following to the bottom:
```ini
[trf_3_7_0]
model:                  core_web_trf
from_version:           3.7.0
to_version:             3.8.0
train_version:          3.7.0
vectors_model:          core_web_lg
```
5. Install Coreferee from the modified local code:
```bash
pip install .
```
6. Install the English language:
```bash
python3 -m coreferee install en
```

7. Install LLM-required packages:

```bash
pip install protobuf #for mistral-7B:
pip install mistral_inference #for mistral-7B:
pip install 'accelerate>=0.26.0' # for gemma-9B
```

## MT Dataset Generation
Test data generation might be time-consuming and will incur token costs. Varied MT datasets would be generated in different attempts. The generated dataset in the data directory can be used directly for testing. It is worth noting that multiple generations might be capable of revealing more unique bugs of SUTs. 

### Lower-level perturbations (baseline from METAL)
- synonym_replacement
- introduce_typos
- add_words
- to_leet

These perturbations are used with the MR template: Equivalence MRT, assuming the perturbations are all semantic-preserving, requiring the SUT to generate semantically similar output when given the perturbed input.

### Dialogue-level perturbations
- DRS
- DRR
- DRD
- DSR
- DSD

These perturbations are dynamically matched with the 4 dialogue-system MRs, and each perturbation can be used multiple times in different MRs.

### Context information check
Check if the question is given sufficient information from the question itself, the context, and the story of CoQA dataset.

### Perturbation-MR Matching
If the context of question is equialently informative when compared with the original dataset, MR1 and MR3 will be matched. In other cases, MR2 and MR4 will be used in accordance to the condition.

## Run Testing on LLM-based Dialogue Systems
Build dialogue system pipeline with following LLMs:
- qwen2_0B5
- qwen2_1B5
- qwen2_7B
- llama3_8B
- gemma2_9B
- mistral03_7B

## Result analysis
### RQ1: Effectiveness


### RQ2: Bug quality


### RQ3: Component contribution

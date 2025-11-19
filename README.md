# QAOA
## Quantum Approximate Optimization Algorithm

## How to run on IBM-Yonsei
### Step 1.
Create an API key on https://quantum.cloud.ibm.com/.

### Step 2.
```shell
git clone https://github.com/Jiwhan-Kim/QAOA-Simple.git
cd ./QAOA-Simple

# Creates an environment file.
vim ./.env

# Create an environment
conda env create -f ./environment.yml
conda activate qiskit312

# Run the code
# Optional - Response takes a lot of time
tmux new -s qaoa
python main.py
```

## How to simulate QAOA
### Step 1.

### Step 2.

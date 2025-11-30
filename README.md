# QAOA
## Quantum Approximate Optimization Algorithm
Two methods to run QAOA: via a QPU of Yonsei (`ibm_yonsei`) or a CPU.

The example here is for searching 'max-cut' partition for a graph with 15 nodes and 14 edges.
The maximum cut is 14(Partitioning into 1010...0101 or 0101...1010).

## Init.
```shell
git clone https://github.com/Jiwhan-Kim/QAOA-Simple.git
cd ./QAOA-Simple

conda env create -f ./environment.yml
conda activate qiskit312
```

## How to run on IBM-Yonsei
### Step 1.
Create an API key on https://quantum.cloud.ibm.com/.

```shell
touch .env
```

Then write your `IBM_TOKEN` and `IBM_INSTANCE` to `.env` file.

```text
IBM_TOKEN="YOUR_IBM_TOKEN"
IBM_INSTANCE="YOUR_IBM_INSTANCE"
```

### Step 2.
```shell
# Save the Account to your local computer.
python ibm_setup.py

# Run the code
# Optional - Response takes a lot of time
tmux new -s qaoa

# You can change the number of qubits and layers to your own.
python stoch_graph_partition.py --qubits 15 --layers 2 --env qpu
```

## How to simulate QAOA
### Step 1.
# You can change the number of qubits and layers to your own.
```shell
# You can change the number of qubits and layers to your own.
python stoch_graph_partition.py --qubits 15 --layers 2 
```

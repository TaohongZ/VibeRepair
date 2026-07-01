# VibeRepair

## VibeRepair for Defects4J



This repository provides scripts for dataset preparation, environment setup, and patch generation.

---

### 1. Prerequisites

Before running VibeRepair, please complete the following setup steps.

#### 1.1 Install Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements_d4j.txt
```

#### 1.2 Install Defects4J

Please follow the official Defects4J installation guide:  
https://github.com/rjust/defects4j


#### 1.3 Install Joern


Installation instructions can be found at:  
https://github.com/joernio/joern


---

### 2. Dataset Preparation

You need to download the **bug-fix example dataset** used for retrieval.

1. Download the dataset from:  
   https://sites.google.com/view/repairingredients

2. Place the downloaded dataset into the following directory:

```text
/VibeRepair_Defects4J_RWB_eval/retrieval_base/
```

Make sure the directory structure remains unchanged.

---

### 3. Plausible Patch Generation

#### 3.1 Default Repair Process

The default repair process runs VibeRepair **without the reasoning component**.

From the `VibeRepair_Defects4J_RWB_eval` directory, execute:

```bash
python vibeRepair_sf.py --dataset D4JV1.0
```

**Note**:  
You must manually add your **OpenAI API key** inside `vibeRepair_sf.py` before running the script.

---

#### 3.2 Repair with Reasoning Component Enabled

To enable the reasoning component, please follow these steps carefully.

##### Step 1: Specify Target Bugs

Edit `get_bugList.py` and add the bug IDs that should be repaired using the reasoning component.

##### Step 2: Checkout Defects4J Projects

Run the following script to checkout the required Defects4J projects:

```bash
python repo_d4j.py
```

##### Step 3: Start the Joern Server

Start the Joern server:

```bash
./joern.sh
```

Ensure Joern is running correctly before proceeding.

##### Step 4: Start the RAG Search Server

Launch the retrieval-augmented generation (RAG) server:

```bash
python rag_search_server.py
```

##### Step 5: Run VibeRepair with Reasoning Component

Finally, run the reasoning-enabled repair process:

```bash
python vibeRepair_sf_reasoning_component.py --dataset D4JV1.0
```

**Note**:  
You must manually add your **OpenAI API key** inside `vibeRepair_sf_reasoning_component.py` before execution.

---

# VibeRepair for ConDefects

Install the required dependencies first from the `VibeRepair` directory:

```bash
pip install -r requirements.txt
```

Then run the following scripts to obtain the VibeRepair ablation study results on ConDefects:

```bash
python Viberepair_ConDefects_eval/batch_repair_specification_module_PFL.py
python Viberepair_ConDefects_eval/batch_repair_specification_module_ochiaiFL.py
python Viberepair_ConDefects_eval/batch_repair_specification_feedback_module_Ochiai.py
python Viberepair_ConDefects_eval/batch_repair_specification_feedback_reasoning_module_Ochiai.py
```

# BugSkillScripts

BugSkillScripts is the benchmark we constructed to explore the applicability of APR tools to skill-script repair in LLM-agent systems.

Run the following scripts to obtain the results of VibeRepair and SpecRover on this benchmark:

```bash
python BugSkillScripts/eval_VibeRepair/run_viberepair_skillsbench_eval.py
python BugSkillScripts/eval_specrover/run_specrover_skillsbench_eval.py
```

# Note

Before running the scripts, export the required API keys:

```bash
export ANTHROPIC_API_KEY=your_anthropic_api_key
export OPENAI_API_KEY=your_openai_api_key
```

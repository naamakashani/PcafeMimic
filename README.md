# **PCAFE**

**PCAFE** is a Python library designed for feature selection (FS) in electronic health record (EHR) datasets.

## **Key Contributions**

## **Personalized Cost-Aware Feature Selection (FS):**  
  A novel method tailored for EHR datasets, addressing challenges such as:  
  - Multiple data modalities  
  - Missing values  
  - Cost-aware decision-making
  - Personalized and Iterative FS 

## **Building Benchmarks**

To generate the benchmark datasets:  
- **MIMIC-III Numeric**  
- **MIMIC-III with Costs**  
- **MIMIC-III Multi-Modal Dataset**  

Navigate to the **`Dataset_Creation`** directory for instructions.

> **Important:**  
> The MIMIC-III data itself is not provided. You must acquire the data independently from [MIMIC-III on PhysioNet](https://mimic.physionet.org/).

## **Running the Code**

To execute the main scripts:  
1. Run **`embedder_guesser.py`**  
2. Run **`main_robust.py`**

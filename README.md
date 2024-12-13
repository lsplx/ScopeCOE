# ScopeCOE
This repository contains the source code for the paper What Knowledge Dominates? Characterizing and Exploring Chain of Evidence in Imperfect External Knowledge

# Overview
Incorporating external knowledge into large language models (LLMs) has emerged as a promising approach to mitigate outdated knowledge of LLMs.
However, external knowledge is often imperfect, containing irrelevant or conflicting information that can impair LLM response reliability.
To address this challenge, existing studies explore LLMs' knowledge preferences and retrieval approaches for handling imperfect external knowledge.
Yet these studies lack clear characterization of preferred knowledge features and primarily focus on single-hop QA scenarios, leaving gaps in understanding complex multi-hop QA scenarios.
In this paper, we characterize LLMs' preferred knowledge under imperfect contexts and explore the dominance of such knowledge in handling complex scenarios.
Inspired by forensic science's Chain of Evidence (CoE) theory, we characterize how evidence pieces form logical chains in external knowledge and propose an automated CoE discrimination approach to identify CoE from external knowledge.
After that, we explore the dominance of CoE in multi-hop QA through three aspects (Effectiveness, Faithfulness, and Robustness) and design a retrieval strategy guided by CoE for knowledge-enhanced frameworks.
Evaluation reveals that CoE enhances LLM performance through more accurate answer generation, stronger knowledge faithfulness, better robustness against conflicting information, and improved retrieval effectiveness in the knowledge-augmentation scenario.

# Environment
- pip install requirements.txt

# Dataset
There are two data sets in total, namely 2WikiMultihopQA_CoE.json and HotpotQA_CoE.json.  
Each dataset contains the following elements:  
question, answer, CoE, Senp_Non_CoE, Word_Non_CoE, irrelevant_info, wrong_ans, external_knowledge, external_judge

# Running

**RQ1_effectiveness** 
```
python RQ1_effectiveness.py 
```   


**RQ2_faithfulness**
```
RQ2_faithfulness.py  
```


**RQ3_robustness**
```
python RQ3_robustness.py  
```



**RQ4_Application**
```
python RQ4_application.py  
```




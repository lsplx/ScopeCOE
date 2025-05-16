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
Accordingly, we propose an automated CoE discrimination approach and evaluate LLMs' effectiveness, faithfulness and robustness with CoE, including its application in the Retrieval-Augmented Generation (RAG). 
Tests on five LLMs show CoE improves generation accuracy, answer faithfulness, robustness to knowledge conflicts, and boosts the performance of existing approaches in three practical RAG scenarios.

The example of CoE features and CoE:
<p align="center">
  <img src="https://github.com/lsplx/ScopeCOE/blob/main/fig/CoE_inference.png" width="700"/>
</p>

# Environment
- pip install requirements.txt

# Dataset
There are two data sets in total, namely 2WikiMultihopQA_CoE.json and HotpotQA_CoE.json.  
Each dataset contains the following elements:  
question, answer, CoE, SenP_Non_CoE, WordP_Non_CoE, irrelevant_info, wrong_ans, external_knowledge, external_judge

The example of SenP_Non_CoE and WordP_Non_CoE:
<p align="center">
  <img src="https://github.com/lsplx/ScopeCOE/blob/main/fig/incomplete_gen.png" width="500"/>
</p>



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




## Analyzing the Impact of Template-Based Prompting on Text-to-World Simulation Quality

## Method Overview
We design a **user-centered prompt template** for Text-to-World generation and evaluate how template factors affect overall video quality.

**Template components**
- Task Instruction
- Prompt Conditions
- Target Scene
- Prompt Style
- Query (user intent)

# Evaluation
**Model:** COSMOS-predict2 Text2World (generation), GPT-4o (prompt generator)  
**Dataset:** PhyGenBench (physics commonsense scenarios)

**Metrics**
- DOVER (Perceptual realism / technical quality)
- LPIPS (Dynamics / temporal consistency proxy)
- BLIP-ITM (prompt-video alignment / controllability)
- PhyGenEval (VQA-based physical plausibility)

# References
We used the code from following repositories: [PhyGenBench](https://github.com/OpenGVLab/PhyGenBench), [DOVER](https://github.com/VQAssessment/DOVER), [LPIPS](https://github.com/richzhang/PerceptualSimilarity), [LAVIS](https://github.com/salesforce/LAVIS)

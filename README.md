# NLP-Study

## About

I implemented bag-of-words, RNN, and BERT using PyTorch. Also, our team did a replication experiment in the don't-stop-pretraining paper.

## Replication Approach

This paper used the following method when pre-training the language model. First, they pre-trained RoBERTa using a huge corpus including different domains. Second, they applied domain adaptive pre-training to pre-train language model again only in domain-specific data. Third, they used task adaptive pre-training to only focus on task-related data which is a subset of domain corpus. Fourth, they used additional task-relevant data for training that humans found. Lastly, they experimented with Automated Data Selection for TAPT.

<img src="https://user-images.githubusercontent.com/87184009/147098351-938d0668-fdcd-456e-93e3-72323c7f2f8b.png" alt="drawing" width="53%"/><img src="https://user-images.githubusercontent.com/87184009/147098446-26b2b38a-8571-40b9-bd71-620627c188ce.png" alt="drawing" width="47%"/>

## Replication Result

We can see that the results of the replication experiment and the actual paper results are similar. In the 500NN-TAPT experiment, we didn't have enough computing resources and model pre-training time, so the result came out differently. The authors of the original paper used Google Cloud v3-8 TPU, but we only used Nvidia GeForce GPU in the replication experiments.

![image](https://user-images.githubusercontent.com/87184009/147098990-7b4c3f55-36c5-435f-9b0e-8be7e405f187.png)

## Final Presentation and Report

Final Presentation: https://github.com/itemgiver/NLP-Study/blob/main/NLP_Final_Presentation.pdf \
Final Report: https://github.com/itemgiver/NLP-Study/blob/main/NLP_Final_Report.pdf \
It is a team project with four KAIST students.

## References

Original Paper: https://github.com/itemgiver/NLP-Study/blob/main/Don%E2%80%99t%20Stop%20Pretraining%20Adapt%20Language%20Models%20to%20Domains%20and%20Tasks.pdf
Github Code: https://github.com/allenai/dont-stop-pretraining/tree/266269faca8645482eef2e710d916607ea2c71d2

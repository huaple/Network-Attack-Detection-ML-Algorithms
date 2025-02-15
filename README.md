# Network-Attack-Detection-ML-Algorithms
This project is focused on constructing detection models for classifying network attack data. To keep the scenario close to the real world and account for the evolving nature of network technology, the UNSW-NB15 network data set, which includes both normal and current low-level attack trades, is used. It applies logistic regression and decision tree models for binary classification and a KNN model for categorizing the multi-class. The decision tree achieved an impressive 99.99% accuracy, vastly surpassing logistic regression's 78.15%. However, the KNN model achieved an average testing accuracy of around 23% for ten category classifiсation.

The details of the UNSW-NB15 data set as used in this project are detailed in the following papers:

Moustafa, Nour, and Jill Slay. "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)."Military Communications and Information Systems Conference (MilCIS), 2015. IEEE, 2015.
Moustafa, Nour, and Jill Slay. "The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set." Information Security Journal: A Global Perspective (2016): 1-14.

The dataset can be found here: https://www.unsw.adfa.edu.au/australian-centre-for-cyber-security/cybersecurity/ADFA-NB15-Datasets/
weights: The directory where the model parameters (attn_weights and importance matrix) are stored
Main.py: The main folder is used to train the model output and store the model parameters (attn_weights and importance matrix).
Tfsicnet.py: Implementation of the model TFSICNet.
evoked_vis.py: Visualize evoke based on EEG data.
Pca.py: Reduces dimension and visualizes model parameters (attn_weights and importance matrix) PCA.
RSA.py: Generates an RDM matrix based on model parameters (attn_weights and importance matrix) and compares it with the RDM matrix of EEG data.
vis.py: Some other visualizations generated based on model parameters (attn_weights and importance matrix).

The code runs as follows:
First run the main file to train the model and save the relevant model parameters to weights_f.txt or weights_g.txt.
The parameters are then moved to the corresponding positions in the weights directory according to different data sets and different actions.
Finally, run PCA, evoked_vis, RSA, and vis files to generate visualizations, changing different data sets and motion imagination actions by changing some of the values in these files.
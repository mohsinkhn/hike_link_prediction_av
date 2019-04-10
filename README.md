# hike_link_prediction_av
Codes for link prediction competition hosted my Hike on Analytics Vidhya


### Approach:

#### Validation strategy:
- Startified KFold cross-validation

#### Feature engineering:
* User activity features - All features activity features for both node1 and node2 users were used. All neural networks all of these features were simply divided by 31 to bring them in 0-1 scale.
* Node-Node features (Graph features) - Common link prediction indicators like (node counts(both node1 and node2), common neighbors, jaccard coefficient, adamic aldar, resource allocation, preferential attachment, total neighbors, neighbor distance) were calculated for 4 different graphs. The four graphs were as follows:
    1. undirected graph of all node pairs in both train and test (this will capture phonebook clusters)
    2. directed graph from node1 --> node2 for all pairs provided in train and test
    3. undirected graph of nodes who had chatted (is_chat == 1). The features from these graphs were mapped in cross-validation fashion to avoid information leakage
    4. directed graph of nodes who had chatted. Again, these were calculated in cross-validation fashion
    Additionally, during analysis it was found that some users had self edges (as in both users were same nodes). features were recalculated removing those nodes
* Swap nodes - If node1 chatted with node2, there was very high change of node2 chatting with node1. feature was generated in cross-validation fashion
* Node2vec - https://snap.stanford.edu/node2vec; it was run 

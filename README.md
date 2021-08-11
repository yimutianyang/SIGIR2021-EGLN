# SIGIR2021-EGLN
The implement of papar "Enhanced Graph Learning for Collaborative Filtering via Mutual Information Maximization"

Neural graph based Collaborative Filtering (CF) models learn user and item embeddings based on the user-item bipartite graph structure, and have achieved state-of-the-art 
recommendation performance. In the ubiquitous implicit feedback based CF, users’ unobserved behaviors are treated as unlinked edges in the user-item bipartite graph. 
As users’ unobserved behaviors are mixed with dislikes and unknown positive preferences, the fixed graph structure input is missing with potential positive preference links. 
In this paper, we study how to better learn enhanced graph structure for CF. We argue that node embedding learning and graph structure learning can mutually enhance each other 
in CF, as updated node embeddings are learned from previous graph structure, and vice versa (i.e., newly updated graph structure are optimized based on current node embedding 
results). Some previous works provided approaches to refine the graph structure. However, most of these graph learning models relied on node features for modeling, which
are not available in CF. Besides, nearly all optimization goals tried to compare the learned adaptive graph and the original graph from a local reconstruction perspective, 
whether the global properties of the adaptive graph structure are modeled in the learning process is still unknown. To this end, in this paper, we propose an enhanced
graph learning network (EGLN ) approach for CF via mutual information maximization. The key idea of EGLN is two folds: First, we let the enhanced graph learning module and the 
node embedding module iteratively learn from each other without any feature input. Second, we design a local-global consistency optimization function to capture the global 
properties in the enhanced graph learning process. Finally, extensive experimental results on three real-world datasets clearly show the effectiveness of our proposed model.

Prerequisites
-------------
* Tensorflow 1.15.0
* Python 3.7.9

Usage
-----
* Dataset:<br>
Under the data folder(cd ./datasets)
* Item Recommendation Task:<br>
* Run model for amazon dataset:<br>
python code/amazon_code/egln.py<br>


Citation
--------
If you find this useful for your research, please kindly cite the following paper:<br>
```
@article{EGLN2021,
  title={Enhanced Graph Learning for Collaborative Filtering via Mutual Information Maximization},
  author={Yonghui Yang, Le Wu, Richang Hong, Kun Zhang and Meng Wang}
  jconference={44nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2021}
}
```

Author contact:
--------------
Email: yyh.hfut@gmail.com

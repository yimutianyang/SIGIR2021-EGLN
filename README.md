# SIGIR2021-EGLN
The implement of paper "Enhanced Graph Learning for Collaborative Filtering via Mutual Information Maximization"
![](https://github.com/yimutianyang/SIGIR2021-EGLN/blob/main/figure/model.jpg)

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
* Run model for amazon dataset:<br>
cd ./code/amazon_code
python egln.py<br>


Citation
--------
If you find this useful for your research, please kindly cite the following paper:<br>
```
@inproceedings{yang2021enhanced,
  title={Enhanced Graph Learning for Collaborative Filtering via Mutual Information Maximization},
  author={Yang, Yonghui and Wu, Le and Hong, Richang and Zhang, Kun and Wang, Meng},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={71--80},
  year={2021}
}
```
This work focus on graph structure learning via graph mutual infomax. If you are also interested in graph node attributes learning, you can refer to the following paper:<br>
```
@inproceedings{wu2020joint,
  title={Joint item recommendation and attribute inference: An adaptive graph convolutional network approach},
  author={Wu, Le and Yang, Yonghui and Zhang, Kun and Hong, Richang and Fu, Yanjie and Wang, Meng},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={679--688},
  year={2020}
}
```
Author contact:
--------------
Email: yyh.hfut@gmail.com

# networkx-to-lava-nc

This is a libary of Spiking Neural Network algorithms (SNNs), along with their
regular/normal/Neumann implementation. to their default/Neumann implementations.
The user can specify an SNN and "normal" algorithm which take as input a
networkx graph, and compute some graph property as output. The output of the
SNN is then compared to the "normal" algorithm as "ground truth", in terms of:

- Score: How many results the SNN algorithm computed correctly (from a set of input
  graphs).
- Runtime
  In theory, the score should always be 100% for the SNN, as it should be an
  exact SNN implementation of the ground truth algorithm. This comparison is
  mainly relevant for the additions of brain adaptation and simulated radiation.

Different SNN implementations may use different encoding schemes, such as
sparse coding, population coding and/or rate coding. 

## Parent Repository

These algorithms can be analysed using
[this parent repository](https://github.com/a-t-0/snncompare).
Together, these repos can be used to investigate the effectivity of various
brain-adaptation mechanisms applied to these algorithms, in order to increase
their radiation robustness. You can run it on various backends, as well as on
a custom LIF-neuron simulator.

## Algorithms

The algorithms in this library can be enhanced with 
[brain-adaptation](https://github.com/a-t-0/snnadaptation) mechanisms
to increase their radiation robustness for space applications. These [simulated
radiation effects](https://github.com/a-t-0/snnradiation) can also be applied 
to the SNNs in these algorithms. The algorithms can be simulated on multiple
backends.

 Hence, an overview is included of the implemented SNN algorithms and their
respective compatibilities with adaptation and radiation implementations:

| Algorithm                                	| Encoding 	| Adaptation 	| Radiation    	| Backend                      	|
|------------------------------------------	|----------	|------------	|--------------	|------------------------------	|
| Minimum Dominating Set Approximation 	    | Sparse   	| Redundancy 	| Neuron Death 	| - networkx LIF<br>- Lava LIF 	|
|                                          	|          	|            	|              	|                              	|
|                                          	|          	|            	|              	|                              	|

### Minimum Dominating Set Approximation

This is an implementation of the distributed algorithm presented by Alipour et al.

- *Input*: Non-triangle, planar Networkx graph. (Non triangle means there
  should not be any 3 nodes that are all connected with each other (forming a
  triangle)). Planar means that if you lay-out the graph on a piece of paper, no
  lines intersect (that you can roll it out on a 2D plane).
- *Output*: A set of nodes that form a dominating set in the graph.

*Description:* The algorithm basically consists of `k` rounds, where you can
choose `k` based on how accurate you want the approximation to be, more rounds
(generally) means more accuracy. At the start each node `i` gets 1 random
number `r_i`. This is kept constant throughout the entire algorithm. Then for
the first round:

- Each node `i` computes how many neighbours (degree) `d_i` it has.
- Then it adds `r_i+d_i=w_i`.
  In all consecutive rounds:
- Each node `i` "computes" which neighbour has the highest weight `w_j`, and
  gives that node 1 mark/point. Then each node `i` has some mark/score `m_i`.
  Next, the weight `w_i=r_i+m_i` is computed (again) and the next round starts.
  This last round is repeated until `k` rounds are completed. At the end, the
  nodes with a non-zero mark/score `m_i` are selected to form the dominating set.

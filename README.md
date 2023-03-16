# Loopy Belief Propagation algorithm in MATLAB
Matlab implementation of [Loopy Belief Propagation](https://en.wikipedia.org/wiki/Belief_propagation) algorithm for image recognition.

## Description
Loopy Belief Propagation (LBP) is a message passing algorithm used for probabilistic graphical models, which allows to perform approximate inference on factor graphs with loops. This algorithm is widely used in the field of artificial intelligence, machine learning, and computer vision.

The basic idea of LBP is to iteratively update belief values of each node in the factor graph by exchanging messages between nodes. These messages are calculated by taking into account the beliefs of neighboring nodes and passing them on to connected nodes in the graph.

In the LBP algorithm, each node in the factor graph represents a random variable, and each edge represents a factor or conditional probability distribution that connects the variables. The messages passed between the nodes represent the belief or probability distribution of the variable given the evidence and the beliefs of its neighboring variables.

The LBP algorithm starts with initializing the belief values of each node to some arbitrary values. Then, the algorithm iteratively updates the belief values of each node by passing messages between the neighboring nodes. The message passing is done in two stages: the forward pass and the backward pass. In the forward pass, messages are passed from the nodes to their neighbors, and in the backward pass, messages are passed from the neighbors back to the nodes.

The algorithm continues until the belief values converge or the maximum number of iterations is reached. At the end of the algorithm, the belief values represent the approximate probabilities of the variables given the evidence.

### Prerequisites

The following software need to be installed.

- MATLAB

## Authors

[Errikos Streviniotis](https://www.linkedin.com/in/errikos-streviniotis/)

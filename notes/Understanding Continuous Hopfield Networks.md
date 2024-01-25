This is an explanation and dissection of the paper [Hopfield Networks is All You Need](https://arxiv.org/pdf/2008.02217.pdf) due to Ramsauer et al. (2020) and continuous Hopfield networks in general.

# Binary Hopfield Networks

(Binary) Hopfield networks are one of the earliest forms of neural network that had, for the most part, been forgotten in the new age of deep learning until relatively recently. They were first proposed as a classical model of biological memory, and so operate somewhat uniquely in the neural network scene, using an energy evolution to converge on stable states rather than the propagation of information through a layered structure.

They live in the subfield of **associative memory** wherein the goal is to store and later retrieve patterns in the structure of the network itself. In the case of binary Hopfield networks, the 'pattern' is some binary string whose length matches the number of neurons, and the goal is to somehow adjust the weights between neurons to 'store' these patterns within the network. Later, when we wish to retrieve a stored pattern, we provide a partial cue -- an obscured, jumbled, or otherwise disrupted form of the stored pattern -- to be reconstructed. This cue is often referred to as a **query**.

	Side note: this paper intentionally looks to create an overlap between the terminolgoy used in Hopfield networks and attention-based mechanisms (e.g. queries and states) to emphasise their equivalence in this setting.

An **update rule** is used to adjust the values attributed to the neurons via the weighted connections. If the weights are suitable, the application of the update rule (perhaps several rounds of updating) will converge the network's neurons to the associated stored pattern. We will discuss updating and energy later on.

## Capacity restrictions

These networks have a very small capacity for storing patterns, particularly as those patterns are binary strings that may appear identical to one another under some masking or obscuration. For example, you can imagine that `101101` and `110101` would be hard, or impossible, to distinguish if we provide a partial que of `___101`. As such, these binary Hopfield networks are practically useless, particularly with real-world data.

So-called "modern" Hopfield networks adjust the update rule to increase this capacity. We are more interested in this paper's proposal of *continuous* Hopfield networks because they enable deep learning (see [Demircigil et al. (2017)](https://arxiv.org/pdf/1702.01929.pdf) for the details concerning their modified energy function -- we will see it soon).

# Continuous Hopfield Networks

We can now do away with binary string patterns and generalise to a continuous (real) vector input (which, for now, is operationally identical to a string of real values). Surprisingly, and amazingly, this paper reports that the storage capacity of these continuous networks is exponential in the dimensionality of the input vector space.

## The energy function and update rule

Let's now be more specific about what it means to retrieve a pattern. The update rule is essentially a search over the energy landscape of the network, and as you might imagine, the goal is to minimise the energy. As such, every Hopfield network is associated with some **energy function** that broadly describes the network's state with respect to the stored patterns (i.e. to what degree they concur, with a high energy being indicative of a poorly representative state).

Importantly, we never work directly with the energy function, because of course that gives us direct access to the stored pattern. Instead, we use an update rule which is formulated so as to minimise this energy function (possibly over several steps).

The modern (binary) Hopfield networks are characterised by the energy function

$$
-\exp(\text{lse}(1, \mathbf{\xi X}^T))\ ,
$$

where $\mathbf{\xi}$ is our query (input) vector, $\mathbf{X}=(\mathbf{x}_1,\mathbf{x}_2,\dots,\mathbf{x}_N)$ is a matrix of $N$ stored patterns, and $\text{lse}(\cdot)$ denotes the log-sum-exp function. The exponentiation is brought about by a generalisation proposed by [Demircigil et al. (2017)](https://arxiv.org/pdf/1702.01929.pdf) to give an exponential storage capacity of $N=2^{d/2}$ for binary patterns, where $d$ is the dimensionality of our binary vectors (strings).

A new continuous energy function is proposed to retain the properties of the modern Hopfield network (i.e. exponential capacity and fast convergence) while generalising to continuous queries;

$$
-\text{lse}(\beta,\mathbf{\xi X}^T)+\frac{1}{2}\mathbf{\xi}^T\mathbf{\xi}+\beta^{-1}\log N+\frac{1}{2}M^2\ ,
$$

for $\beta>0$, with the quadratic term ensuring that the norm of the state vector $\mathbf{\xi}$ remains finite and the energy is bounded (which comes as a natural consequence of continuity), and where $M=\max_i\|\mathbf{x}_i\|$ is the largest stored pattern. Specifically, the bounding of the energy is $0\leq E\leq 2M^2$.

The paper goes on to define the novel update rule from their energy function as

$$
\mathbf{\xi}^\text{new}=\mathbf{X}\ \text{softmax}(\beta\mathbf{X}^T\mathbf{\xi})\ ,
$$

which they immediately then prove to converge globally. This can be shown to be equivalent to a transformer's attention-based mechanism

$$
\mathbf{V}\ \text{softmax}\bigg(\frac{1}{\sqrt{d_k}}\mathbf{QK}^T\bigg)\ ,
$$

which allows us to consider attention mechanisms as a continuous Hopfield network.


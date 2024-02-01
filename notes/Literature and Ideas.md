
# Ideas

#compression 
I am very interested in the idea that two sets may be associated to one another (as per the Hopfield is all you need paper), as this might be a way to associate some compression of an image, text, audio, etc. with the original, decompressed version. This lives very close to what Hopfield networks always have been, but I am curious as to how similar the partial cue (i.e. the compression) has to be to retrieve the original. For example, could we use a hashing function (not necessarily cryptographic) to process images into small, fixed sized objects (e.g. real vectors) to be associated with the original images.

#cryptography 
Following from the above, could an approximation of a cryptographic hash function be determined by continuous Hopfield networks learning the associations between cyphertext and plaintext? Maybe not at large scale, but say with 8 bits?

#classification #denoising
Following from the idea that noisy cues of stored patterns may be reconstructed, can continuous Hopfield layers serve as a pre-processing step in classification tasks to act as an intelligent approach to denoising?

#classification #explainability
The Hopfield is all you need paper refer to Hopfield Pooling layers, which makes me think that continuous Hopfield networks could be used in place of pooling layers in a CNN-style or LLM. In addition to improving performance, this could be a way to introduce layers that can provide explainable output at intermediate steps in the model. For example, a Hopfield layer might be employed following a convolutional layer to explain the learned kernel (e.g. by associating the kernel with a stored pattern, which would itself be learned, or by recalling a concept in connection with the kernel as a query). It could be potentially useful for a Hopfield layer to perform several low-level classifications on low-level feature in an image to produce a 'report' on the final classification outcome (e.g. several object detection steps followed by a classification on the scene an image might be depicting -- tennis ball, net, racket -> tennis court / grass, goal post -> football pitch).

#segmentation
Hopfield networks may be naturally perfect for image segmentation. It could easily be argued that a raw image is a partial cue and the layers or interesting portion of the image is the pattern to be recognised, and it is essentially shrouded in noise. A critical application of this is for medical applications; e.g. [lesion segmentation](https://paperswithcode.com/task/lesion-segmentation).

#theory
The Hopfield is all you need paper has been informally criticised online for its lack of addressing exactly *how* memories should be created automatically (since Hopfield networks are auto-associative). They seem to suggest that every content-based memory lookup in a neural network, with some non-linearity, is some sort of one-step Hopfield Network, which is to say that the networks are converging on learned states. However, the authors do not spend much time describing how these states come to be learned, other than the general spookiness of neural networks doing what we'd like them to.

#theory
Hopfield networks may be an interesting way to *preemptively* bias a network's weights based on the input. For example, in a classification, we might want to use some sort of small pre-classification network based on Hopfield layers to provide a (soft) weight mask to overlay onto the main classifying networks weights to bias them towards a particular type of classification. The pre-classifying model may be able to influence the choice of classification style or reduce the likelihood for overfitting by forming a more intelligent way of doing dropout. I haven't seen any work on this -- not that I have looked -- but I like this idea, and it fits with our other ideas about pre-emptive AI for deep learning applications. This might even be a path into general AI that is able to select pathways of computation based on input, giving it the ability to do arbitrarily many things based on the input it is given. 

---

# Literature

#compression
State-of-the-art image compression is reported by [Hillar et al. (2014)](https://redwood.berkeley.edu/wp-content/uploads/2018/01/hillar2014hopfield.pdf) using modern Hopfield networks. We might be able to smash it out the park with a hybrid architecture between a continuous Hopfield network and autoencoder.

#LLM
I read somewhere that protein classification (e.g. [Håndstad et al., (2007)](https://link.springer.com/article/10.1186/1471-2105-8-23)) can be brought to state-of-the-art using LSTMs, and then LSTMs can be exchanged like-for-like for continuous Hopfield networks (as they are equivalent to an attention mechanism with the added generality of iterability, as useless as that may be).

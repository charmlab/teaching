/* lessons.js — the course lessons, how they group into Learning Map modules,
   and the free YouTube lecture clips for each lesson section.
   Clip data comes from the "Concept map" sheet (lesson section ↔ video). Regenerate, don't hand-edit LESSONS.
   Loaded by index.html and module.html (data only) and by each lesson page (adds clips + prev/next). */
(function () {
"use strict";

var LESSONS = {"knn":{"n":2,"title":"k-Nearest Neighbours","blurb":"Data representation, distance, Voronoi boundaries, choosing k, and where KNN breaks.","sec":{"0":[{"id":"FnIWxC1PLVA","code":"V2.01","t":"Overview","d":"3:41","p":0},{"id":"762JnO92Rls","code":"V2.06","t":"Nearest Neighbors Overview","d":"5:28","p":0}],"1":[{"id":"fjK1eZk6Rl4","code":"V2.02","t":"What Is Data?","d":"5:39","p":0},{"id":"_C4VpM3dFyM","code":"V2.03","t":"Redundancies in Data","d":"9:34","p":0},{"id":"mPVIIbTfjxU","code":"V2.04","t":"How Is Data Represented?","d":"2:17","p":0},{"id":"84t0xhV_g0A","code":"V2.05","t":"Datasets of Training and Testing","d":"3:55","p":0},{"id":"hPHPkVtiCA8","code":"V2.07","t":"Lp Norms and Cosine Similarity","d":"15:00","p":1}],"2":[{"id":"hPHPkVtiCA8","code":"V2.07","t":"Lp Norms and Cosine Similarity","d":"15:00","p":1},{"id":"xr8C77fQoqk","code":"V2.08","t":"Nearest Neighbor Algorithm","d":"4:20","p":0}],"3":[{"id":"do6FuAlwLP8","code":"V2.09","t":"KNN Algorithm","d":"10:51","p":0}],"4":[{"id":"kY7Lww19zaU","code":"V2.10","t":"Cross-Validation & Generalization","d":"11:19","p":0}],"5":[{"id":"xNVSE4wVFxA","code":"V2.11","t":"KNN Pitfalls & Summary","d":"6:36","p":0}],"6":[{"id":"xNVSE4wVFxA","code":"V2.11","t":"KNN Pitfalls & Summary","d":"6:36","p":1}]}},"decision-trees":{"n":3,"title":"Decision Trees","blurb":"Tree anatomy, decision regions, entropy, information gain, greedy learning, overfitting, and pruning.","sec":{"0":[{"id":"sZ78cuaB1cw","code":"V3.01","t":"Introduction","d":"2:28","p":0}],"1":[{"id":"e2w4oUuPpSA","code":"V3.02","t":"Tree Overview","d":"11:30","p":0},{"id":"GIK0iANK1Ts","code":"V3.03","t":"Tree Decision Boundaries","d":"2:52","p":0},{"id":"LbOoQpyvALk","code":"V3.04","t":"Discrete Trees","d":"1:43","p":0}],"2":[{"id":"kPQxGRfM3Co","code":"V3.05","t":"Tree Expressiveness","d":"2:19","p":0}],"3":[{"id":"ibkavLCGL_M","code":"V3.07","t":"Uncertainty Quantification, Entropy, & Information Gain","d":"14:37","p":1}],"4":[{"id":"ibkavLCGL_M","code":"V3.07","t":"Uncertainty Quantification, Entropy, & Information Gain","d":"14:37","p":1}],"5":[{"id":"_wtFccgBxHQ","code":"V3.06","t":"Intuition for Learning Decision Trees","d":"7:44","p":0},{"id":"x0lZhXQFV_0","code":"V3.08","t":"Learning Trees in Action","d":"8:09","p":0}],"6":[{"id":"U6lFne1nmYs","code":"V3.09","t":"Tree Summary","d":"11:46","p":1}],"7":[{"id":"U6lFne1nmYs","code":"V3.09","t":"Tree Summary","d":"11:46","p":1}]}},"linear-regression":{"n":4,"title":"Linear Regression","blurb":"Models, losses, optimization, gradient descent, overfitting, and ridge regression.","sec":{"0":[{"id":"O-P7Amn95kw","code":"V5.01","t":"Motivation","d":"5:21","p":1}],"1":[{"id":"O-P7Amn95kw","code":"V5.01","t":"Motivation","d":"5:21","p":1}],"2":[{"id":"ypAx7842X2o","code":"V5.02","t":"A Modular Approach to Learning","d":"4:16","p":0}],"3":[{"id":"HPSCDknVM5I","code":"V5.03","t":"Regression Step 1: Defining the Model","d":"9:52","p":1}],"4":[{"id":"HPSCDknVM5I","code":"V5.03","t":"Regression Step 1: Defining the Model","d":"9:52","p":1}],"5":[{"id":"s7WVlATduUM","code":"V5.04","t":"Loops vs. Vectorized Code","d":"8:25","p":1}],"6":[{"id":"cGviVXcjVJs","code":"V5.05","t":"Regression Step 2: Defining the Loss","d":"9:40","p":0}],"7":[{"id":"cGviVXcjVJs","code":"V5.05","t":"Regression Step 2: Defining the Loss","d":"9:40","p":1}],"8":[{"id":"mROEx-9Fdjs","code":"V5.06","t":"Regression Step 3: Optimizing the Loss","d":"4:18","p":0},{"id":"_Kr1IIrG6cg","code":"V5.07","t":"Analytical Solution to Regression","d":"6:18","p":0}],"9":[{"id":"13KGIf9Mky4","code":"V5.08","t":"Gradient Descent for Linear Regression","d":"17:00","p":0}],"10":[{"id":"mXm5D6VvAN0","code":"V5.09","t":"Polynomial and Regularized Regression","d":"11:40","p":0}],"12":[{"id":"kan0WA2IWQE","code":"V5.10","t":"Summary & Discussion","d":"6:46","p":0}]}},"linear-classification":{"n":5,"title":"Linear Classification","blurb":"Scores, thresholds, decision boundaries, precision and recall, the perceptron, and multi-class strategies.","sec":{"0":[{"id":"mPQqvI2mDsU","code":"V6.01","t":"Motivation","d":"6:49","p":0}],"1":[{"id":"Ud81jvNSIKk","code":"V6.02","t":"Formulating Linear Classifiers","d":"13:16","p":1}],"2":[{"id":"Ud81jvNSIKk","code":"V6.02","t":"Formulating Linear Classifiers","d":"13:16","p":1}],"3":[{"id":"iUZt1urk7zo","code":"V6.03","t":"Loss Functions for Linear Classification","d":"15:00","p":0}],"4":[{"id":"r39zJmdq4bc","code":"V6.04","t":"Precision and Recall","d":"25:00","p":0}],"5":[{"id":"aOmpIJMQfro","code":"V6.05","t":"The Perceptron Algorithm","d":"8:33","p":0},{"id":"N9mP8MMaHr8","code":"V6.06","t":"On the Existence and Identification of Nonlinear Transformations","d":"4:48","p":1}],"6":[{"id":"eh9ZsaJ0sjs","code":"V6.07","t":"Multiclass Classification & Discussion","d":"15:00","p":0}],"7":[{"id":"eh9ZsaJ0sjs","code":"V6.07","t":"Multiclass Classification & Discussion","d":"15:00","p":1}]}},"logistic-regression":{"n":6,"title":"Logistic Regression","blurb":"The four tries, the sigmoid, probabilistic outputs, cross-entropy, gradient descent, regularization, and softmax.","sec":{"0":[{"id":"0rxqrUAW5YY","code":"V7.01","t":"Review of Linear Regression and Linear Classification","d":"7:41","p":1}],"1":[{"id":"hH8jaaiLiKA","code":"V7.02","t":"Linear Classification via Thresholded Activations","d":"11:13","p":0},{"id":"mb_BUPRn69E","code":"V7.03","t":"Linear Classification via Sigmoidal Activations","d":"13:45","p":1},{"id":"VAld71hjLB4","code":"V7.04","t":"Summary of Attempts at Linear Classification","d":"3:17","p":0}],"2":[{"id":"mb_BUPRn69E","code":"V7.03","t":"Linear Classification via Sigmoidal Activations","d":"13:45","p":1}],"3":[{"id":"sP4MYEqapvI","code":"V7.05","t":"Probabilistic Interpretation of Logistic Regression","d":"6:20","p":0}],"4":[{"id":"BS4MS6Vzg8M","code":"V7.06","t":"MLE for Logistic Regression","d":"19:00","p":0}],"7":[{"id":"9upL1TPS_-Y","code":"V7.07","t":"Summary & Discussion","d":"9:10","p":0}]}},"bias-variance-ensembles":{"n":7,"title":"Bias, Variance & Ensembles","blurb":"The decomposition, majority voting, bagging, random forests, AdaBoost, and gradient boosting.","sec":{"0":[{"id":"mMp0jt1Q7Do","code":"V4.01","t":"Motivation","d":"5:15","p":0}],"1":[{"id":"7OhJ1Xv8TdM","code":"V4.02","t":"The Bias-Variance Tradeoff","d":"25:00","p":1}],"2":[{"id":"7OhJ1Xv8TdM","code":"V4.02","t":"The Bias-Variance Tradeoff","d":"25:00","p":1}],"3":[{"id":"TIj_icxhNHs","code":"V4.03","t":"Overview of Ensembling Methods & Discussion","d":"20:00","p":0}],"4":[{"id":"owy_nAKh-wU","code":"V4.04","t":"Bagging","d":"7:53","p":0}],"5":[{"id":"by089gffP_Y","code":"V4.05","t":"Boosting","d":"15:00","p":0}],"6":[{"id":"RtV2rDZhDig","code":"V4.06","t":"Summary & Discussion","d":"7:14","p":0}]}},"svm":{"n":8,"title":"Support Vector Machines","blurb":"Max-margin classification, support vectors, soft margins and slack, feature maps, and the kernel trick.","sec":{"0":[{"id":"FCRYWkZ2DS8","code":"V8.02","t":"Motivation: Good Linear Classifiers","d":"2:43","p":0}],"1":[{"id":"jC98TV79Enc","code":"V8.03","t":"Intuition for Max-Margin Classification","d":"4:32","p":0},{"id":"cJNTk7QeWs8","code":"V8.04","t":"Formulating Max-Margin Classification","d":"13:19","p":0}],"2":[{"id":"KBmCnG2wcgs","code":"V8.06","t":"Learning Linear SVMs","d":"8:18","p":1}],"3":[{"id":"eCyZYFo-tgU","code":"V8.05","t":"Duality and Lagrangian Multipliers","d":"14:35","p":0}],"4":[{"id":"DMZ2_6ZdgX0","code":"V8.07","t":"SVMs with Slack Variables","d":"4:59","p":0}],"5":[{"id":"ixUDS6OIzc8","code":"V8.08","t":"Nonlinear SVMs","d":"11:17","p":0}],"6":[{"id":"q3MPacCo8qk","code":"V8.09","t":"Summary","d":"4:47","p":0}]}},"neural-networks":{"n":9,"title":"Neural Networks","blurb":"The XOR wall, feature maps, representation learning, MLPs, expressivity, and a hand-built network that solves XOR.","sec":{"0":[{"id":"EOspFL02pCo","code":"V9.01","t":"Motivation","d":"5:15","p":0}],"1":[{"id":"5Kb035wbT3Y","code":"V9.02","t":"Linear Separability of NOT, AND, & XOR","d":"16:00","p":0}],"2":[{"id":"YcKB8zzmU8s","code":"V9.03","t":"Hand-Designed Feature Engineering","d":"14:47","p":0}],"3":[{"id":"vS7uDSeOAJc","code":"V9.06","t":"Hierarchical Feature Learning","d":"10:55","p":0}],"4":[{"id":"mbd0WTtzzH0","code":"V9.04","t":"Biological, Computational, and Artificial Neurons","d":"7:49","p":0},{"id":"Ts9chOhkc80","code":"V9.05","t":"Feed-Forward Neural Networks","d":"11:16","p":0}],"5":[{"id":"olZCp15cPqg","code":"V10.02","t":"On Neural Network Expressive Power","d":"12:11","p":0}],"6":[{"id":"YeCXSbkqHgc","code":"V10.03","t":"Solving XOR Using Neural Networks","d":"10:40","p":0}],"7":[{"id":"HohZUYXzJBo","code":"V9.07","t":"Discussion","d":"3:26","p":1}],"8":[{"id":"HohZUYXzJBo","code":"V9.07","t":"Discussion","d":"3:26","p":1}]}},"backpropagation":{"n":10,"title":"Backpropagation","blurb":"The chain rule on a computation graph, error signals, SGD, and the training tricks that make deep networks work.","sec":{"0":[{"id":"liMgPtL8RbI","code":"V10.01","t":"Mid-Class Review","d":"2:18","p":1}],"1":[{"id":"9j7ruUbEYsk","code":"V10.04","t":"Backpropagation Overview","d":"12:08","p":1}],"2":[{"id":"UHi5dplK8o0","code":"V10.06","t":"Multivariate Chain Rule","d":"4:26","p":0}],"3":[{"id":"2V8WiTZdnyQ","code":"V10.05","t":"Backpropagation Example 1","d":"7:30","p":0},{"id":"gICP2jQGmoM","code":"V10.07","t":"Backpropagation Example 2","d":"2:12","p":0}],"4":[{"id":"6Ck61vF8Nkw","code":"V10.08","t":"Backpropagation Algorithm","d":"6:21","p":0}],"6":[{"id":"QrnnmoVp53k","code":"V12.06","t":"Training Tips & Tricks","d":"1:46","p":1}],"7":[{"id":"qUuF74G1ad8","code":"V10.09","t":"Discussion","d":"6:41","p":0}]}},"cnn":{"n":11,"title":"Convolutional Neural Networks","blurb":"Locality, weight sharing, the sliding dot product, padding and stride, filter banks, ReLU, pooling, and growing receptive fields.","sec":{"0":[{"id":"_QdefKDJF-M","code":"V11.01","t":"Introduction","d":"11:43","p":0}],"1":[{"id":"WuYwFCSnKIU","code":"V11.02","t":"Neural Networks for Computer Vision","d":"10:47","p":0}],"2":[{"id":"WuYwFCSnKIU","code":"V11.02","t":"Neural Networks for Computer Vision","d":"10:47","p":1}],"3":[{"id":"yqPktGNNckM","code":"V11.03","t":"1D Convolutions","d":"8:56","p":0},{"id":"Ju8dFj_-yy8","code":"V11.04","t":"2D Convolutions","d":"8:00","p":0},{"id":"eHx56A4fekk","code":"V11.05","t":"Intuition for Convolution Operations","d":"8:38","p":0},{"id":"VWmTNTSqtGo","code":"V11.08","t":"Example Convolutions & Toeplitz Matrices","d":"11:44","p":0}],"4":[{"id":"-JoMXlOFj4w","code":"V11.07","t":"Convolutions as Layers","d":"1:23","p":0},{"id":"lyBtBxU7F0o","code":"V11.09","t":"Convolution Hyperparameters","d":"7:21","p":0}],"5":[{"id":"BMN6_HYM0L8","code":"V11.10","t":"Pooling Layers","d":"11:29","p":0}],"6":[{"id":"kPgHx1PbF0M","code":"V11.06","t":"Mid-Class Review","d":"3:32","p":1}]}},"cnn-part-2":{"n":12,"title":"CNNs in Practice","blurb":"1x1 convolutions, large curated datasets, GPUs and scaling, the LeNet-to-ResNet architecture lineage, transfer learning, interpretability, adversarial examples, and dense prediction.","sec":{"0":[{"id":"17FJpII9WAw","code":"V12.01","t":"Motivation","d":"7:32","p":0}],"1":[{"id":"aYoHcOlmGjA","code":"V11.11","t":"1x1 Convolutions","d":"11:13","p":0}],"2":[{"id":"DgGIGz0VZiw","code":"V12.02","t":"Data","d":"7:25","p":0}],"3":[{"id":"4IFwcBe4M8M","code":"V12.03","t":"Compute","d":"4:22","p":0}],"4":[{"id":"Ax_yEYQqIJE","code":"V12.04","t":"History of Neural Architectures","d":"9:28","p":0},{"id":"M9H-ltH3pSw","code":"V12.05","t":"Comparing Performance","d":"5:11","p":0}],"8":[{"id":"qyUyfZLBHcU","code":"V12.07","t":"Discussion","d":"7:58","p":0}]}},"language-models":{"n":13,"title":"Language Models","blurb":"The chain rule, n-grams and data sparsity, tokenization, one-hot vectors, word embeddings and analogies, neural language models, decoding, and perplexity.","sec":{"0":[{"id":"iPjaXGIxsc8","code":"V13.01","t":"Motivation","d":"6:14","p":0}],"1":[{"id":"Hw_OPbj9myc","code":"V13.02","t":"N-Gram Language Models","d":"25:00","p":1}],"2":[{"id":"Hw_OPbj9myc","code":"V13.02","t":"N-Gram Language Models","d":"25:00","p":0}],"3":[{"id":"f6cdyNkJS8E","code":"V13.03","t":"Distributional Embeddings","d":"21:00","p":1}],"4":[{"id":"f6cdyNkJS8E","code":"V13.03","t":"Distributional Embeddings","d":"21:00","p":0}],"5":[{"id":"IhjfI5Znjxk","code":"V13.04","t":"Neural N-Grams","d":"12:09","p":0}],"7":[{"id":"zmGmxcaUsso","code":"V13.05","t":"Evaluations","d":"2:05","p":0}],"8":[{"id":"hi1XXTrMKPk","code":"V13.06","t":"Discussion","d":"9:18","p":0}]}},"sequence-models":{"n":14,"title":"Sequence Models","blurb":"Recurrent networks, hidden state as memory, backpropagation through time, vanishing and exploding gradients, gradient clipping, LSTMs and GRUs, bidirectional and stacked cells, and the seq2seq bottleneck that motivates attention.","sec":{"0":[{"id":"BVkxQbkmb90","code":"V14.01","t":"Review & Overview","d":"12:08","p":0}],"1":[{"id":"GvRjvvkU3Qg","code":"V14.02","t":"RNN Example 1","d":"9:35","p":0},{"id":"mt5jLOkB6XQ","code":"V14.03","t":"RNN Example 2","d":"8:10","p":0}],"2":[{"id":"bRptw-TS1jo","code":"V14.04","t":"RNNs over Words & Characters","d":"10:48","p":0}],"3":[{"id":"zFRNVd_l-ds","code":"V14.05","t":"Exploding and Vanishing Gradients","d":"17:00","p":0}],"4":[{"id":"wkzI4PKDGKA","code":"V14.06","t":"LSTMs","d":"5:41","p":0}],"5":[{"id":"nic7ogaGn0s","code":"V14.07","t":"Chaining Recurrent Neural Cells","d":"7:05","p":0}],"6":[{"id":"qyfyrXqlBr0","code":"V14.08","t":"Transformers and Their Applications","d":"8:39","p":1}],"7":[{"id":"AZKhVpa4-aQ","code":"V14.09","t":"Summary","d":"7:10","p":0}]}},"attention-transformers":{"n":15,"title":"Attention and Transformers","blurb":"Queries, keys, and values, multi-head attention, the transformer block, architecture families, the LLM training pipeline, token economics, context windows, and scaling laws.","sec":{"3":[{"id":"qyfyrXqlBr0","code":"V14.08","t":"Transformers and Their Applications","d":"8:39","p":1}],"4":[{"id":"qyfyrXqlBr0","code":"V14.08","t":"Transformers and Their Applications","d":"8:39","p":1}]}},"mle-map-bayesian":{"n":16,"title":"MLE, MAP & Bayesian Prediction","blurb":"Probability versus likelihood, maximum likelihood estimation, priors and posteriors, MAP, conjugacy, and fully Bayesian prediction.","sec":{"0":[{"id":"O02H2rGZoQE","code":"V15.01","t":"Introduction","d":"4:26","p":0}],"1":[{"id":"tuIVkan8Fxg","code":"V15.02","t":"Probability vs. Likelihood","d":"9:14","p":0}],"2":[{"id":"gv5Y4m5QNRo","code":"V15.03","t":"Maximum Likelihood Estimation for Bernoulli Distributions","d":"12:40","p":0},{"id":"1O8BzOrULp8","code":"V15.04","t":"Maximum Likelihood Estimation for Gaussian Distributions","d":"10:48","p":0}],"3":[{"id":"b7nLWNTupRE","code":"V15.05","t":"MLE in Past Models","d":"3:11","p":1}]}},"gaussian-bayes":{"n":17,"title":"Gaussian Bayes Models","blurb":"The Bayes classifier, fitting Gaussians by MLE, the multivariate Gaussian, Gaussian discriminant analysis, Naive Bayes, and the QDA/LDA decision-boundary geometry.","sec":{"1":[{"id":"1JIZNKvm7lI","code":"V15.06","t":"Discriminative vs. Generative Models","d":"11:58","p":0}],"2":[{"id":"f1A1TbNvm_U","code":"V15.07","t":"Bayes Classifier Overview","d":"11:03","p":0},{"id":"DrqMT-rbkPY","code":"V15.08","t":"Risk of a Classifier","d":"2:36","p":0}],"3":[{"id":"1O8BzOrULp8","code":"V15.04","t":"Maximum Likelihood Estimation for Gaussian Distributions","d":"10:48","p":1}],"4":[{"id":"4ErW42CvFls","code":"V15.09","t":"Gaussian Discriminant Analysis: Intuition","d":"14:42","p":1}],"5":[{"id":"4ErW42CvFls","code":"V15.09","t":"Gaussian Discriminant Analysis: Intuition","d":"14:42","p":0},{"id":"fvOPipNh9MA","code":"V15.10","t":"Gaussian Discriminant Analysis: Learning, Inference, and Decision Boundaries","d":"10:02","p":0}],"6":[{"id":"ZOa_bCDlTyo","code":"V15.11","t":"Naive Bayes and Summary","d":"13:49","p":0}],"7":[{"id":"fvOPipNh9MA","code":"V15.10","t":"Gaussian Discriminant Analysis: Learning, Inference, and Decision Boundaries","d":"10:02","p":1}],"8":[{"id":"ZOa_bCDlTyo","code":"V15.11","t":"Naive Bayes and Summary","d":"13:49","p":1}]}},"pca":{"n":18,"title":"Principal Component Analysis","blurb":"Dimensionality reduction, eigenvectors of the covariance matrix, the PCA algorithm, the variance/reconstruction duality, scree plots, compression and denoising, random projections, and supervised PCA.","sec":{"2":[{"id":"nZNnTEExG_c","code":"V17.03","t":"Vectors, Spaces, Norms, Orthogonality, and Independence","d":"9:24","p":1}]}},"autoencoders":{"n":19,"title":"Autoencoders and VAEs","blurb":"From PCA's fixed encoder/decoder to learned, nonlinear, and deep autoencoders, sparse and denoising variants, the latent-space failure modes, and the VAE with the reparameterization trick and KL regularization.","sec":{}},"clustering":{"n":20,"title":"Clustering & Density Estimation","blurb":"K-means, soft K-means, the generative view, Gaussian mixture models, latent variables, and the EM algorithm.","sec":{"2":[{"id":"NombCwKymEk","code":"V16.04","t":"K-Means Objective, Loss, and Challenges","d":"14:20","p":0}],"5":[{"id":"iofLQlFeKgc","code":"V16.06","t":"Intuition for Mixture of Gaussians","d":"8:58","p":0}]}},"reinforcement-learning":{"n":21,"title":"Reinforcement Learning","blurb":"Multi-armed bandits and the exploration-exploitation dilemma, Markov decision processes, value functions, Bellman equations, value and policy iteration, and Q-learning, with live agents you can train.","sec":{"0":[{"id":"wo6WWHXKjKg","code":"V18.04","t":"Examples of Reinforcement Learning Systems","d":"13:40","p":1}],"3":[{"id":"_9aKumf0oRg","code":"V18.03","t":"Formulating Reinforcement Learning (Part 2: Value Functions)","d":"12:06","p":0}]}}};

/* Map module → its lessons, in reading order. Only modules listed here get a lesson page;
   the others keep the old popup until their lessons are wired in. */
var MODULE_LESSONS = {
  "c-ml":     ["knn", "decision-trees", "bias-variance-ensembles", "mle-map-bayesian", "gaussian-bayes", "svm"],
  "c-nn":     ["linear-regression", "linear-classification", "logistic-regression", "neural-networks", "backpropagation"],
  "x-unsup":  ["pca", "autoencoders", "clustering"],
  "a-vision": ["cnn", "cnn-part-2"],
  "a-nlp":    ["language-models", "sequence-models", "attention-transformers"],
  "x-rl":     ["reinforcement-learning"]
};
var MODULE_TITLES = { "c-ml": "Classical ML Models", "c-nn": "Linear to Neural Models", "x-unsup": "Unsupervised Learning",
  "a-vision": "Computer Vision", "a-nlp": "Language Models", "x-rl": "Reinforcement Learning" };

function lessonModule(slug) {
  for (var k in MODULE_LESSONS) if (MODULE_LESSONS[k].indexOf(slug) >= 0) return k;
  return null;
}
function toSec(d) { var p = d.split(":"); return (+p[0]) * 60 + (+p[1]); }
/* unique clips and total minutes for a lesson */
function lessonStats(slug) {
  var L = LESSONS[slug], seen = {}, n = 0, s = 0;
  if (!L) return { clips: 0, minutes: 0 };
  Object.keys(L.sec).forEach(function (k) {
    L.sec[k].forEach(function (v) { if (!seen[v.id]) { seen[v.id] = 1; n++; s += toSec(v.d); } });
  });
  return { clips: n, minutes: Math.round(s / 60) };
}
window.LESSONS = LESSONS;
window.MODULE_LESSONS = MODULE_LESSONS;
window.MODULE_TITLES = MODULE_TITLES;
window.lessonModule = lessonModule;
window.lessonStats = lessonStats;

/* ---------------- lesson-page enhancements ---------------- */
function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
function track(ev, p) { try { if (window.gtag) gtag("event", ev, p || {}); } catch (e) {} }

var CSS = '\
.lv{margin:6px 0 22px;display:flex;flex-direction:column;gap:8px}\
.lv-row{display:flex;gap:12px;align-items:center;width:100%;text-align:left;font:inherit;cursor:pointer;\
 background:#fff;border:1px solid var(--line,#e5e5e5);border-radius:10px;padding:7px 12px 7px 7px;transition:border-color .15s,box-shadow .15s}\
.lv-row:hover{border-color:#cc0000;box-shadow:0 2px 10px rgba(0,0,0,.06)}\
.lv-thumb{position:relative;flex:0 0 auto;width:112px;aspect-ratio:16/9;border-radius:6px;overflow:hidden;background:#111}\
.lv-thumb img{width:100%;height:100%;object-fit:cover;display:block}\
.lv-thumb::after{content:"";position:absolute;left:50%;top:50%;width:30px;height:21px;margin:-10.5px 0 0 -15px;border-radius:6px;\
 background:#cc0000 url("data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 10 10%27%3E%3Cpath d=%27M3 2l5 3-5 3z%27 fill=%27white%27/%3E%3C/svg%3E") center/12px no-repeat;opacity:.92}\
.lv-meta{display:flex;flex-direction:column;gap:2px;min-width:0}\
.lv-kicker{font-family:var(--sans,sans-serif);font-size:10.5px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--muted,#777)}\
.lv-title{font-family:var(--sans,sans-serif);font-size:14.5px;font-weight:600;color:var(--text,#222);line-height:1.3}\
.lv-sub{font-family:var(--sans,sans-serif);font-size:12px;color:var(--faint,#999)}\
.lv-player{position:relative;width:100%;aspect-ratio:16/9;border-radius:10px;overflow:hidden;background:#000}\
.lv-player iframe{position:absolute;inset:0;width:100%;height:100%;border:0}\
.lv-close{align-self:flex-end;font:600 12px var(--sans,sans-serif);color:var(--muted,#777);background:none;border:0;cursor:pointer;padding:2px 4px}\
.lv-strip{display:inline-flex;align-items:center;gap:8px;margin:14px 0 0;font:600 13.5px var(--sans,sans-serif);color:#cc0000;text-decoration:none;border:1px solid #f0c8c8;background:#fff6f6;border-radius:999px;padding:7px 14px}\
.lv-strip:hover{background:#ffecec}.lv-dot{font-size:11px}\
.lv-toc{color:#cc0000;font-size:10px;margin-left:2px}\
.lv-where{font:600 12.5px var(--sans,sans-serif);color:var(--muted,#777);text-align:center;margin:12px 0 0}.lv-where a{color:inherit}\
@media (max-width:560px){.lv-thumb{width:92px}.lv-title{font-size:13.5px}}';

function clipRow(v) {
  var b = document.createElement("button");
  b.type = "button"; b.className = "lv-row"; b.dataset.id = v.id;
  b.innerHTML = '<span class="lv-thumb"><img loading="lazy" alt="" src="https://i.ytimg.com/vi/' + v.id + '/mqdefault.jpg"></span>' +
    '<span class="lv-meta"><span class="lv-kicker">Lecture clip</span><span class="lv-title">' + esc(v.t) + '</span>' +
    '<span class="lv-sub">' + v.d + (v.p ? ' · overlaps other sections' : '') + '</span></span>';
  return b;
}
function playInPlace(row, slug) {
  var id = row.dataset.id, box = document.createElement("div");
  box.className = "lv-player";
  box.innerHTML = '<iframe src="https://www.youtube-nocookie.com/embed/' + id + '?autoplay=1&rel=0&modestbranding=1" ' +
    'title="Lecture clip" allow="autoplay; encrypted-media; picture-in-picture; fullscreen" allowfullscreen></iframe>';
  var close = document.createElement("button");
  close.type = "button"; close.className = "lv-close"; close.textContent = "Close video ×";
  close.addEventListener("click", function () { box.replaceWith(row); close.remove(); });
  row.replaceWith(box); box.after(close);
  track("lesson_clip_play", { lesson: slug, video_id: id });
}

function enhance() {
  var parts = location.pathname.split("/").filter(Boolean);
  var i = parts.indexOf("c-ml"), slug = i >= 0 ? parts[i + 1] : null;
  var L = LESSONS[slug];
  if (!L) return;
  var root = "../../../";

  var st = document.createElement("style"); st.textContent = CSS; document.head.appendChild(st);

  /* clips under each section heading */
  document.querySelectorAll(".lesson-body section > h2").forEach(function (h2) {
    var sn = h2.querySelector(".secnum");
    var num = sn ? parseInt(sn.textContent, 10) : NaN;
    var vids = L.sec[num];
    if (!vids) return;
    var wrap = document.createElement("div"); wrap.className = "lv";
    vids.forEach(function (v) {
      var row = clipRow(v);
      row.addEventListener("click", function () { playInPlace(row, slug); });
      wrap.appendChild(row);
    });
    h2.after(wrap);
    /* mark the section in the table of contents */
    var sec = h2.parentNode;
    var tocA = sec.id && document.querySelector('.toc a[href="#' + sec.id + '"]');
    if (tocA) tocA.insertAdjacentHTML("beforeend", ' <span class="lv-toc" title="has lecture clips">&#9654;</span>');
    if (!document.getElementById("lv-first")) wrap.id = "lv-first";
  });

  /* one line under the intro so the clips are discoverable */
  var stt = lessonStats(slug), lede = document.querySelector(".lesson-header .lede") || document.querySelector(".lesson-header h1");
  if (stt.clips && lede) {
    var strip = document.createElement("a");
    strip.className = "lv-strip"; strip.href = "#lv-first";
    strip.innerHTML = '<span class="lv-dot">&#9654;</span> ' + stt.clips + ' lecture clips &middot; ' + stt.minutes +
      ' min &mdash; each one sits under the section it covers';
    var holder = document.createElement("div"); holder.appendChild(strip); lede.after(holder);
  }

  /* back link + prev/next inside the module */
  var mod = lessonModule(slug);
  if (!mod) return;
  var list = MODULE_LESSONS[mod], idx = list.indexOf(slug), title = MODULE_TITLES[mod];
  var back = document.querySelector(".backlink");
  if (back) { back.href = root + "module.html?id=" + mod; back.innerHTML = "&larr; " + esc(title); }
  var url = function (s) { return "../" + s + "/index.html"; };
  var prev = idx > 0 ? list[idx - 1] : null, next = idx < list.length - 1 ? list[idx + 1] : null;
  var modUrl = root + "module.html?id=" + mod;
  var link = function (href, kicker, t) {
    return '<a href="' + href + '"><div class="nav-kicker">' + kicker + '</div><div class="nav-title">' + esc(t) + '</div></a>';
  };
  var html =
    (prev ? link(url(prev), "&larr; Previous lesson", LESSONS[prev].title) : link(modUrl, "&larr; Back to", title)) +
    '<span class="spacer"></span>' +
    (next ? link(url(next), "Next lesson &rarr;", LESSONS[next].title) : link(root + "index.html", "Module complete &rarr;", "Back to the map"));
  /* reuse the lesson's own prev/next bar (it follows the old lecture order) */
  var nav = document.querySelector(".lesson-nav");
  if (!nav) { nav = document.createElement("nav"); nav.className = "lesson-nav"; document.querySelector(".lesson-body").appendChild(nav); }
  nav.innerHTML = html;
  var where = document.createElement("div"); where.className = "lv-where";
  where.innerHTML = 'Lesson ' + (idx + 1) + ' of ' + list.length + ' in <a href="' + modUrl + '">' + esc(title) + '</a>';
  nav.after(where);
}

if (typeof document !== "undefined" && document.querySelector) {
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", enhance); else enhance();
}
})();

'''
Question 1: A Simplified Decision Tree
You are to implement a decision-tree learner for classification. To simplify your work, this will not be a general
purpose decision tree. Instead, your program can assume that
• each item has two continuous features x ∈ R2
• the class label is binary and encoded as y ∈ {0, 1}
• data files are in plaintext with one labeled item per line, separated by whitespace:
x11 x12 y1
...
xn1 xn2 yn
Your program should implement a decision tree learner according to the following guidelines:
• Candidate splits (j, c) for numeric features should use a threshold c in feature dimension j in the form of
xj ≥ c.
• c should be on values of that dimension present in the training data; i.e. the threshold is on training points,
not in between training points. You may enumerate all features, and for each feature, use all possible values
for that dimension.
• You may skip those candidate splits with zero split information (i.e. the entropy of the split), and continue
the enumeration.
• The left branch of such a split is the “then” branch, and the right branch is “else”.
• Splits should be chosen using information gain ratio. If there is a tie you may break it arbitrarily.
• The stopping criteria (for making a node into a leaf) are that
– the node is empty, or
– all splits have zero gain ratio (if the entropy of the split is non-zero), or
– the entropy of any candidates split is zero
• To simplify, whenever there is no majority class in a leaf, let it predict y = 1
'''
import numpy as np
import matplotlib.pyplot as plt

class DecisionTree:
    def __init__(self, data, labels):

        # Initialize the tree, data, labels, and depth
        self.data = data
        self.labels = labels
        self.tree = self.build_tree(data, labels)
        self.depth = self.get_depth(self.tree)
    
    def build_tree(self, data, labels):
        # Recursively build the tree based on maximum gain ratio
        if len(data) == 0:
            return None
        elif self.all_same(labels):
            return labels[0]
        else:
            best_feature, best_threshold = self.choose_feature(data, labels)
            if best_feature == None or best_threshold == None:
                if len(labels[labels == 1]) >= len(labels[labels == 0]):
                    return 1
                else:
                    return 0
            tree = [best_feature, best_threshold, [], []]
            left_data, left_labels, right_data, right_labels = self.split_data(data, labels, best_feature, best_threshold)
            tree[2] = self.build_tree(left_data, left_labels)
            tree[3] = self.build_tree(right_data, right_labels)
            return tree

    def choose_feature(self, data, labels):
        # Choose the feature that maximizes the gain ratio
        best_feature = None
        best_threshold = None
        best_gain = 0
        for feature in range(len(data[0])):
            for threshold in data[:, feature]:
                left_data, left_labels, right_data, right_labels = self.split_data(data, labels, feature, threshold)
                gain = self.gain_ratio(labels, left_labels, right_labels)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def split_data(self, data, labels, feature, threshold):
        # Split the data into left and right based on the feature and threshold
        left_data = []
        left_labels = []
        right_data = []
        right_labels = []
        for i in range(len(data)):
            if data[i][feature] >= threshold:
                left_data.append(data[i])
                left_labels.append(labels[i])
            else:
                right_data.append(data[i])
                right_labels.append(labels[i])
        return np.array(left_data), np.array(left_labels), np.array(right_data), np.array(right_labels)
    
    def gain_ratio(self, labels, left_labels, right_labels):
        # Compute the gain ratio
        if self.split_info(left_labels, right_labels) == 0:
            return 0
        else:
            return (self.entropy(labels) - self.split_entropy(left_labels, right_labels)) / self.split_info(left_labels, right_labels)

    def split_info(self, left_labels, right_labels):
        # Compute the split info
        p = float(len(left_labels)) / (len(left_labels) + len(right_labels))
        if p == 0 or p == 1:
            return 0
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    
    def split_entropy(self, left_labels, right_labels):
        # Compute the split entropy
        p = float(len(left_labels)) / (len(left_labels) + len(right_labels))
        return p * self.entropy(left_labels) + (1 - p) * self.entropy(right_labels)
    
    def entropy(self, labels):
        # Compute the entropy
        if len(labels) == 0:
            return 0
        p = float(len(labels[labels == 1])) / len(labels)
        if p == 0 or p == 1:
            return 0
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    
    def all_same(self, labels):
        # Check to see if all labels are the same
        return np.all(labels == labels[0])
    
    def get_depth(self, tree):
        # Get the maximum depth of the tree
        if tree == None or type(tree) != list:
            return 0
        return 1 + max(self.get_depth(tree[2]), self.get_depth(tree[3]))
    
    def predict(self, data):
        # Predict an array of data
        return np.array([self.predict_one(d) for d in data])

    def predict_one(self, data):
        # Predict one data point
        tree = self.tree
        while type(tree) == list:
            feature, threshold = tree[0], tree[1]
            if data[feature] >= threshold:
                tree = tree[2]
            else:
                tree = tree[3]
        return tree

''' Question 2.2:
Handcraft a small training set where both classes are present but the
algorithm refuses to split; instead it makes the root a leaf and stop; Importantly, if we were to manually
force a split, the algorithm will happily continue splitting the data set further and produce a deeper tree with
zero training error. You should (1) plot your training set, (2) explain why. Hint: you don’t need more than a
handful of items.
'''

refuse_split_data = np.array([[1,1],[1,-1],[-1,-1],[-1,1]])
refuse_split_labels = np.array([[1],[0],[1],[0]])
refuse_split_tree = DecisionTree(refuse_split_data, refuse_split_labels)

# Plot the Data and color the points based on their label
plt.scatter(refuse_split_data[:,0], refuse_split_data[:,1], c=refuse_split_labels[:,0])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Refuse Split Data')
plt.colorbar()
plt.savefig('figs/refuse_split_data.pdf')
plt.clf()

''' Question 2.3:
Use the training set Druns.txt.  For the root node, list all candidate cuts and their information gain ratio. If the entropy
of the candidate split is zero, please list its mutual information (i.e. information gain). Hint: to get $\log_2(x)$ when your 
programming language may be using a different base, use \verb|log(x)/log(2)|. Also, please follow the split rule in the first section. 
'''

# Load the data
druns_data = np.loadtxt('Homework 2 data/Druns.txt')
druns_labels = druns_data[:,2]
druns_data = druns_data[:,:2]

# Build the tree
druns_tree = DecisionTree(druns_data, druns_labels)

# Calculate the information gain ratio for each feature and threshold on first split
def get_gain_ratios(labels, left_labels, right_labels):
    # Compute the gain ratio
    if druns_tree.split_info(left_labels, right_labels) == 0:
        return druns_tree.entropy(labels) - druns_tree.split_entropy(left_labels, right_labels)
    else:
        return (druns_tree.entropy(labels) - druns_tree.split_entropy(left_labels, right_labels)) / druns_tree.split_info(left_labels, right_labels)

candidate_cuts = []
for feature in range(len(druns_data[0])):
    for threshold in druns_data[:, feature]:
        left_data, left_labels, right_data, right_labels = druns_tree.split_data(druns_data, druns_labels, feature, threshold)
        gain = get_gain_ratios(druns_labels, left_labels, right_labels)
        candidate_cuts.append([feature, threshold, gain])

print('Candidate Root Cuts:')
print('Feature, Threshold, Gain Ratio')
for cut in candidate_cuts:
    print(cut)

''' Question 2.4:
Decision tree is not the most accurate classifier in general. However,
it persists. This is largely due to its rumored interpretability: a data scientist can easily explain a tree to a
non-data scientist. Build a tree from D3leaves.txt. Then manually convert your tree to a set of logic rules.
Show the tree and the rules.
'''

# Load the data
d3leaves_data = np.loadtxt('Homework 2 data/D3leaves.txt')
d3leaves_labels = d3leaves_data[:,2]
d3leaves_data = d3leaves_data[:,:2]

# Build the tree
d3leaves_tree = DecisionTree(d3leaves_data, d3leaves_labels)

# Print the tree into the terminal and format it
def print_tree(tree, depth=0):
    if type(tree) != list:
        print(depth * 3 * '-' + str(tree)+ '\\\\')
    else:
        print(depth * 3 * '-' + 'Feature ' + str(tree[0]) + ' $\\geq$ ' + str(tree[1]) + '\\\\')
        print_tree(tree[2], depth + 1)
        print_tree(tree[3], depth + 1)

print('Decision Tree for D3leaves.txt:')
print_tree(d3leaves_tree.tree)

''' Question 2.5:
For this question only, make sure you DO NOT VISUALIZE the data sets or plot your
tree’s decision boundary in the 2D x space. If your code does that, turn it off before proceeding. This is
because you want to see your own reaction when trying to interpret a tree. You will get points no matter
what your interpretation is. And we will ask you to visualize them in the next question anyway.
• Build a decision tree on D1.txt. Show it to us in any format (e.g. could be a standard binary tree with
nodes and arrows, and denote the rule at each leaf node; or as simple as plaintext output where each
line represents a node with appropriate line number pointers to child nodes; whatever is convenient
for you). Again, do not visualize the data set or the tree in the x input space. In real tasks you will not
be able to visualize the whole high dimensional input space anyway, so we don’t want you to “cheat”
here.
• Look at your tree in the above format (remember, you should not visualize the 2D dataset or your
tree’s decision boundary) and try to interpret the decision boundary in human understandable English.
• Build a decision tree on D2.txt. Show it to us.
• Try to interpret your D2 decision tree. Is it easy or possible to do so without visualization?
'''

# Load the data
d1_data = np.loadtxt('Homework 2 data/D1.txt')
d1_labels = d1_data[:,2]
d1_data = d1_data[:,:2]

# Build the tree
d1_tree = DecisionTree(d1_data, d1_labels)

# Print the tree into the terminal and format it
print('Decision Tree for D1.txt:')
print_tree(d1_tree.tree)

# Load the data
d2_data = np.loadtxt('Homework 2 data/D2.txt')
d2_labels = d2_data[:,2]
d2_data = d2_data[:,:2]

# Build the tree
d2_tree = DecisionTree(d2_data, d2_labels)

# Print the tree into the terminal and format it
print('Decision Tree for D2.txt:')
print_tree(d2_tree.tree)
print(d2_tree.depth)

''' Question 2.6:
(Hypothesis space) [10 pts] For D1.txt and D2.txt, do the following separately:
• Produce a scatter plot of the data set.
• Visualize your decision tree’s decision boundary (or decision region, or some other ways to clearly
visualize how your decision tree will make decisions in the feature space).
Then discuss why the size of your decision trees on D1 and D2 differ. Relate this to the hypothesis space of
our decision tree algorithm.
'''

# Write a funciton that will plot the decision boundary of a tree
def plot_decision_boundary(tree, data, labels, savefile, title, lims):
    # Create a meshgrid to plot the decision boundary
    x_min, x_max = data[:,0].min() - 1, data[:,0].max() + 1
    y_min, y_max = data[:,1].min() - 1, data[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Predict the labels for the meshgrid
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(data[:,0], data[:,1], c=labels, cmap=plt.cm.Spectral, s=1)
    plt.xlim(lims[0], lims[1])
    plt.ylim(lims[0],lims[1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.savefig(savefile)
    plt.clf()

# Write a function that will plot the data
def plot_data(data, labels, savefile, title):
    # Plot the Data and color the points based on their label
    plt.scatter(data[:,0], data[:,1], c=labels, cmap=plt.cm.Spectral, s=2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.colorbar()
    plt.savefig(savefile)
    plt.clf()

# Plot the data for D1.txt
plot_data(d1_data, d1_labels, 'figs/d1_data.pdf', 'Data for D1.txt')

# Plot the data for D2.txt
plot_data(d2_data, d2_labels, 'figs/d2_data.pdf', 'Data for D2.txt')

# Plot the decision boundary for D1.txt
plot_decision_boundary(d1_tree, d1_data, d1_labels, 'figs/d1_decision_boundary.pdf', 'Decision Boundary for D1.txt', lims = (0,1))

# Plot the decision boundary for D2.txt
plot_decision_boundary(d2_tree, d2_data, d2_labels, 'figs/d2_decision_boundary.pdf', 'Decision Boundary for D2.txt', lims = (0,1))

''' Question 2.7:
We provide a data set Dbig.txt with 10000 labeled items. Caution: Dbig.txt is
sorted.
• You will randomly split Dbig.txt into a candidate training set of 8192 items and a test set (the rest).
Do this by generating a random permutation, and split at 8192.
• Generate a sequence of five nested training sets D32 ⊂ D128 ⊂ D512 ⊂ D2048 ⊂ D8192 from the
candidate training set. The subscript n in Dn denotes training set size. The easiest way is to take
the first n items from the (same) permutation above. This sequence simulates the real world situation
where you obtain more and more training data.
• For each Dn above, train a decision tree. Measure its test set error errn. Show three things in your
answer: (1) List n, number of nodes in that tree, errn. (2) Plot n vs. errn. This is known as a learning
curve (a single plot). (3) Visualize your decision trees’ decision boundary (five plots)
'''

# Load the data
dbig_data = np.loadtxt('Homework 2 data/Dbig.txt')
dbig_labels = dbig_data[:,2]
dbig_data = dbig_data[:,:2]

# Randomly split the data into a training and test set
np.random.seed(0)
permutation = np.random.permutation(len(dbig_data))
training_data = dbig_data[permutation[:8192]]
training_labels = dbig_labels[permutation[:8192]]
test_data = dbig_data[permutation[8192:]]
test_labels = dbig_labels[permutation[8192:]]

# Create a list of training subsets
training_subsets = []
training_subsets.append(training_data[:32])
training_subsets.append(training_data[:128])
training_subsets.append(training_data[:512])
training_subsets.append(training_data[:2048])
training_subsets.append(training_data)

# Create a Function that will count the number of nodes in the decision tree
def get_num_nodes(tree):
    if type(tree) != list:
        return 0
    return 1 + get_num_nodes(tree[2]) + get_num_nodes(tree[3])

# Loop through the training subsets and train a tree on each, print n, number of tree nodes, and error.
# Create a list of n versus error for plotting
# Create a list of trees for plotting
n = []
error = []
trees = []
for i in range(len(training_subsets)):
    tree = DecisionTree(training_subsets[i], training_labels[:len(training_subsets[i])])
    trees.append(tree)
    n.append(len(training_subsets[i]))
    error.append(np.sum(tree.predict(test_data) != test_labels) / len(test_labels))
    print('n =', len(training_subsets[i]), ', Number of Nodes =', get_num_nodes(tree.tree), ', Error =', error[i])

# Plot n versus error
plt.figure()
plt.plot(n, error, c='r')
plt.xlabel('n')
plt.ylabel('Error')
plt.title('Error versus n')
plt.savefig('figs/error_vs_n.pdf')
plt.clf()

# Plot the decision boundary for each tree
for i in range(len(trees)):
    plot_decision_boundary(trees[i], dbig_data, dbig_labels, 'figs/dbig_decision_boundary_' + str(i) + '.pdf', 'Decision Boundary for n = ' + str(n[i]), lims = (-1.5,1.5))

''' Question 3:
Learn to use sklearn (https://scikit-learn.org/stable/). Use sklearn.tree.DecisionTreeClassifier to produce trees for
datasets D32, D128, D512, D2048, D8192. Show two things in your answer: (1) List n, number of nodes in that
tree, errn. (2) Plot n vs. errn.
'''
import sklearn
from sklearn.tree import DecisionTreeClassifier

# Loop through the training subsets and train a tree on each, print n, number of tree nodes, and error.
# Create a list of n versus error for plotting
n = []
error = []
for i in range(len(training_subsets)):
    tree = DecisionTreeClassifier()
    tree.fit(training_subsets[i], training_labels[:len(training_subsets[i])])
    trees.append(tree)
    n.append(len(training_subsets[i]))
    error.append(np.sum(tree.predict(test_data) != test_labels) / len(test_labels))
    print('n =', len(training_subsets[i]), ', Number of Nodes =', tree.tree_.node_count, ', Error =', error[i])

# Plot n versus error
plt.figure()
plt.plot(n, error, c='r')
plt.xlabel('n')
plt.ylabel('Error')
plt.title('Error versus n')
plt.savefig('figs/error_vs_n_sklearn.pdf')
plt.clf()

''' Question 4:
Fix some interval [a, b] and sample n = 100 points x from this interval uniformly. Use these to build a training
set consisting of n pairs (x, y) by setting function y = sin(x).
Build a model f by using Lagrange interpolation, check more details in https://en.wikipedia.org/wiki/Lagrange
polynomial and https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.lagrange.html.
Generate a test set using the same distribution as your test set. Compute and report the resulting model’s train and
test error. What do you observe? Repeat the experiment with zero-mean Gaussian noise ε added to x. Vary the
standard deviation for ε and report your findings.
'''

# Generate the training data by randomly sampling through the interval [a, b]
a = 0
b = 1
n = 100
x = np.random.uniform(a, b, n)
y = np.sin(x)

# Build a Classifier using Lagrange Interpolation from scipy
import scipy
from scipy.interpolate import lagrange

classifier = lagrange(x, y)

# Generate the test data by randomly sampling through the interval [a, b]
x_test = np.random.uniform(a, b, 20)
y_test = np.sin(x_test)

# Compute the training and test error
# Use the log mean squared error
train_error = np.log(np.mean((y - classifier(x))**2))
test_error = np.log(np.mean((y_test - classifier(x_test))**2))

# Print the training and test error
print('Training Error =', train_error)
print('Test Error =', test_error)

# Add varying amounts of noise to the training data and compute the training and test error
# Use the log mean squared error
train_errors = []
test_errors = []
noise = np.linspace(0, 2, 20)

for i in range(len(noise)):
    x = np.random.uniform(a, b, n)
    x_eps = x + np.random.normal(0, noise[i], n)
    y_eps = np.sin(x_eps)
    classifier = lagrange(x_eps, y_eps)
    train_errors.append(np.log(np.mean((y_eps - classifier(x_eps))**2)))
    test_errors.append(np.log(np.mean((y_test - classifier(x_test))**2)))

# Print the training and test errors
print('Training Errors (increaing epsilon) =', train_errors)
print('Test Errors (increasing epsilon) =', test_errors)






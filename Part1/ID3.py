import numpy as np
import math


# In the decision tree, non-leaf nodes are going to be represented via TreeNode
class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}


# In the decision tree, leaf nodes are going to be represented via TreeLeafNode
class TreeLeafNode:
    def __init__(self, data, label):
        self.data = data
        self.labels = label


class DecisionTree:
    def __init__(self, dataset: list, labels, features, criterion="information gain"):
        """
        :param dataset: array of data instances, each data instance is represented via an Python array
        :param labels: array of the labels of the data instances
        :param features: the array that stores the name of each feature dimension
        :param criterion: depending on which criterion ("information gain" or "gain ratio") the splits are to be performed
        """
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None

        # further variables and functions can be added...

    def calculate_entropy__(self, dataset, labels):
        """
        :param dataset: array of the data instances
        :param labels: array of the labels of the data instances
        :return: calculated entropy value for the given dataset
        """
        entropy_value = 0.0

        """
        Entropy calculations
        """
        unique_labels = set(labels)
        number_of_instances = len(labels)
        for label in unique_labels:
            # calculate the probability of each label
            number_of_instances_with_label = labels.count(label)
            probability = number_of_instances_with_label / number_of_instances
            if probability != 0:  # to avoid log(0) case
                entropy_value += -probability * math.log(probability, 2)

        return entropy_value

    def calculate_average_entropy__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an average entropy value is calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute an average entropy value is going to be calculated...
        :return: the calculated average entropy value for the given attribute
        """
        average_entropy = 0.0
        """
            Average entropy calculations
        """
        dataset_size = len(dataset)  # number of data instances

        # split the dataset according to the given attribute
        # get the index of the attribute
        attribute_index = self.features.index(attribute)
        # get the values of the attribute
        attribute_values = set([instance[attribute_index] for instance in dataset])
        # split the dataset according to the attribute values
        split_dataset = {}
        split_labels = {}
        for attribute_value in attribute_values:
            split_dataset[attribute_value] = []
            split_labels[attribute_value] = []
            for index in range(dataset_size):
                if dataset[index][attribute_index] == attribute_value:
                    split_dataset[attribute_value].append(dataset[index])
                    split_labels[attribute_value].append(labels[index])

        # calculate the average entropy
        for attribute_value in attribute_values:
            # calculate the probability of the attribute value
            probability = len(split_dataset[attribute_value]) / dataset_size
            # calculate the entropy of the attribute value
            entropy = self.calculate_entropy__(
                split_dataset[attribute_value], split_labels[attribute_value]
            )
            if probability != 0:
                average_entropy += probability * entropy
        return average_entropy

    def calculate_information_gain__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an information gain score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the information gain score is going to be calculated...
        :return: the calculated information gain score
        """
        information_gain = 0.0
        """
            Information gain calculations
        """
        current_entropy = self.calculate_entropy__(dataset, labels)
        average_entropy = self.calculate_average_entropy__(dataset, labels, attribute)
        information_gain = current_entropy - average_entropy
        return information_gain

    def calculate_intrinsic_information__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances on which an intrinsic information score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the intrinsic information score is going to be calculated...
        :return: the calculated intrinsic information score
        """
        intrinsic_info = 0.0
        """
            Intrinsic information calculations for a given attribute
        """

        dataset_size = len(dataset)  # number of data instances

        # split the dataset according to the given attribute
        # get the index of the attribute
        attribute_index = self.features.index(attribute)
        # get the values of the attribute
        attribute_values = set([instance[attribute_index] for instance in dataset])
        # split the dataset according to the attribute values
        split_dataset = {}
        for attribute_value in attribute_values:
            num = 0
            for index in range(dataset_size):
                if dataset[index][attribute_index] == attribute_value:
                    num += 1
            split_dataset[attribute_value] = num

        # calculate the intrinsic information
        for attribute_value in attribute_values:
            # calculate the probability of the attribute value
            probability = split_dataset[attribute_value] / dataset_size
            if probability != 0:
                intrinsic_info += -probability * math.log(probability, 2)

        return intrinsic_info

    def calculate_gain_ratio__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances with which a gain ratio is going to be calculated
        :param labels: array of labels of those instances
        :param attribute: for which attribute the gain ratio score is going to be calculated...
        :return: the calculated gain ratio score
        """
        """
            Your implementation
        """
        gain_ratio = 0.0
        information_gain = self.calculate_information_gain__(dataset, labels, attribute)
        intrinsic_information = self.calculate_intrinsic_information__(
            dataset, labels, attribute
        )
        if intrinsic_information != 0:
            gain_ratio = information_gain / intrinsic_information
        return gain_ratio

    def _should_be_leaf_node__(self, dataset, labels, used_attributes):
        """
        It checks whether the node should be a leaf node or not.
        It takes the dataset and labels of the data instances falling under the current node as input.
        It also takes the array of the attributes that are already used while recursively constructing the tree.

        It returns True if the node should be a leaf node, otherwise it returns False.
        """
        if len(set(labels)) == 1:
            return True
        if len(used_attributes) == len(self.features):  # all attributes are used
            return True
        return False

    def _get_unused_attributes__(self, used_attributes):
        """
        It takes the array of the attributes that are already used while recursively constructing the tree.
        It returns the array of the attributes that are not used yet.
        """
        unused_attributes = []
        for attribute in self.features:
            if attribute not in used_attributes:
                unused_attributes.append(attribute)
        return unused_attributes

    def ID3__(self, dataset, labels, used_attributes):
        """
        Recursive function for ID3 algorithm
        :param dataset: data instances falling under the current  tree node
        :param labels: labels of those instances
        :param used_attributes: while recursively constructing the tree, already used labels should be stored in used_attributes
        :return: it returns a created non-leaf node or a created leaf node
        """
        """
            Your implementation
        """
        if self._should_be_leaf_node__(dataset, labels, used_attributes):
            # create a leaf node
            leafNode = TreeLeafNode(dataset, labels)
            return leafNode
        else:  # create a non-leaf node
            # find the best attribute to split
            best_attribute = None
            unused_attributes = self._get_unused_attributes__(used_attributes)
            best_attribute = unused_attributes[0]
            for attribute in unused_attributes:
                if self.criterion == "information gain":
                    if best_attribute == None or self.calculate_information_gain__(
                        dataset, labels, attribute
                    ) > self.calculate_information_gain__(
                        dataset, labels, best_attribute
                    ):
                        best_attribute = attribute
                elif self.criterion == "gain ratio":
                    if best_attribute == None or self.calculate_gain_ratio__(
                        dataset, labels, attribute
                    ) > self.calculate_gain_ratio__(dataset, labels, best_attribute):
                        best_attribute = attribute

            # create a non-leaf node
            nonLeafNode = TreeNode(best_attribute)
            # split the dataset according to the best attribute
            # get the index of the attribute
            attribute_index = self.features.index(best_attribute)
            # get the values of the attribute
            attribute_values = set([instance[attribute_index] for instance in dataset])
            # split the dataset according to the attribute values
            split_dataset = {}
            split_labels = {}
            for attribute_value in attribute_values:
                split_dataset[attribute_value] = []
                split_labels[attribute_value] = []
                for index in range(len(dataset)):
                    if dataset[index][attribute_index] == attribute_value:
                        split_dataset[attribute_value].append(dataset[index])
                        split_labels[attribute_value].append(labels[index])

            # recursively create the subtrees
            for attribute_value in attribute_values:
                nonLeafNode.subtrees[attribute_value] = self.ID3__(
                    split_dataset[attribute_value],
                    split_labels[attribute_value],
                    used_attributes + [best_attribute],
                )

            return nonLeafNode

    def _get_most_common_label(self, labels):
        """
        It takes array of labels as input and returns the most common label in the given array
        """
        unique_labels = set(labels)
        most_common_label = labels[0]
        for label in unique_labels:
            if labels.count(label) > labels.count(most_common_label):
                most_common_label = label
        return most_common_label

    def predict(self, x):
        """
        :param x: a data instance, 1 dimensional Python array
        :return: predicted label of x

        If a leaf node contains multiple labels in it, the majority label should be returned as the predicted label
        """
        predicted_label = None
        """
            Your implementation
        """
        node = self.root
        while type(node) == TreeNode:  # while node is not a leaf node traverse the tree
            # get the attribute of the node
            attribute = node.attribute
            # get the index of the attribute
            attribute_index = self.features.index(attribute)
            # get the instance's attribute value
            attribute_value = x[attribute_index]

            # get the next node (leaf or non-leaf)
            node = node.subtrees[attribute_value]

        # node is a leaf node
        # get the labels of the leaf node
        labels = node.labels  # type: ignore
        predicted_label = self._get_most_common_label(labels)
        return predicted_label

    def train(self):
        self.root = self.ID3__(self.dataset, self.labels, [])
        print("Training completed")

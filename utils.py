from tabulate import tabulate
import matplotlib.pyplot as py
import numpy as np
import itertools
import operator
import random
import math
import csv

#######################################
# READ/WRITE FILES
#######################################
def parse_csv(filename):
	'''
    Converts comma deliminated file into 2D list of instances with given attributes
    PARAMETERS: filename = str representation of txt filename
    RETURNS: attributes = list of attribute headers
			 table = 2D list of instances with given attributes
    '''
	attributes = []
	table = []
	with open(filename) as file:
		readCSV = csv.reader(file, delimiter=',')
		for row in readCSV:
			table.append(row)

		# Set attibute names to header of file
		attributes = table[0]
		table = table[1:]
	file.close()
	return attributes, table

def parse_file_data(filename):
    '''
    Converts comma deliminated text file into 2D list
    of instances with given attributes
    PARAMETERS: filename = str representation of txt filename
    RETURNS: table = 2D list of instances with given attributes
    '''
    table = []     # nested table to hold mpg values
    infile = open(filename, "r")   # read file mode
    lines = infile.readlines()     # parse lines in file
    for line in lines:
        line = line.strip()        # strip whitespace
        values = line.split(",")   # break up attributes in instance by ','
        convert_column_to_numeric(values)
        table.append(values)       # add instance to end of table
        infile.close()
    return table

def write_csv(filename, attributes, table):
	'''
    Write clean data table into comma deliminated .txt file
    where each instance in table is a row in file
    PARAMETERS: filename = filename to be created/cleaned
				attributes = list of attribute headings
                table = clean dataset
    RETURNS: csv file outputted with clean data
    '''
	# Insert attributes header at the top of CSV table filename
	table.insert(0, attributes)

	with open(filename, 'w', newline='') as file:
		writeCSV = csv.writer(file)
		writeCSV.writerows(table)
	file.close()

def write_to_file(filename, table):
    '''
    Write clean data table into comma deliminated .txt file
    where each instance in table is a row in file
    PARAMETERS: filename = filename to be created/cleaned
                table = clean dataset
    RETURNS: .txt file outputted with clean data
    '''
    outfile = open(filename, "w")
    for row in table:
        for i in range(len(row) - 1):
            outfile.write(str(row[i]) + ",")
        outfile.write(str(row[-1]) + "\n")
    outfile.close()

#######################################
# TABLE MANIPULATION
#######################################
def convert_column_to_numeric(values):
    '''
    Converts str representation of numeric values to such in a 2D list
    PARAMETERS: values = 2D list of dataset
    RETURNS: values = 2D list of dataset with respective str or numeric values
    '''
    for i in range(len(values)):
        try:
            numeric_value = int(values[i])
            values[i] = numeric_value       # attribute is a numeric value
        except ValueError:
            pass

def convert_data_to_numeric(data):
    '''
    Converts float representation of values in 2D dataset to ints
    PARAMETERS: data = 2D list of dataset
    RETURNS: data = 2D list of dataset with numeric values
    '''
    for row in data:
        for i in range(len(row)):
            try:
                int_value = int(row[i])
                row[i] = int_value       # attribute is a numeric value
            except ValueError:
                pass

def format_confusion_table(table, num_class, labels=None):
    '''
    Format 2D list of confusion matrix
    PARAMETERS: table = 2D list of confusion matrix
                num_class = number of classes that exist under class label
                labels = list of class labels
    RETURNS: table = confusion matrix with inserted labels needed
    '''
    for i in range(num_class):
        # Total instances of this class
        total = sum(table[i])
        if table[i][i] != 0:
            recognition = (table[i][i] / total) * 100
        else:
            recognition = 0
        # Insert data labels
        table[i].append(total)
        table[i].append(recognition)
        if labels is None:
            table[i].insert(0, i + 1)
        else:
            table[i].insert(0, labels[i])

    return table

#######################################
# DATA FORMATTING
#######################################
def get_column(table, column_index):
    '''
    Return list of values for a column in table
    PARAMETERS: table = 2D dataset
                column_index = int representing index of attribute
    RETURNS: column = list of values of an attribute in table
    '''
    column = []
    for row in table:
        if row[column_index] != "NA":
            column.append(row[column_index])
    return column

def convert_boolean_attr_to_numeric(table, column_index):
    '''
    Convert boolean (or "yes"/"no") attribute to 1/0 numeric values
    Used to help compute distance of classifiery in kNN
    PARAMETERS: table = 2D list of dataset
                column_index = int representing index of column to modify
    '''
    for row in table:
        if row[column_index] == True or row[column_index].lower() == "yes":
            row[column_index] = 1
        else:
            row[column_index] = 0

def remove_missing_vals(table):
    '''
    Remove instances with missing attribute values
    PARAMETERS: table = 2D dataset
    RETURNS: NA
    '''
    for row in table:
        if 'NA' or 'none' in row:
            table.remove(row)

def replace_missing_vals_mean(table, column_index):
    '''
    Replace missing values with their attribute avg
    PARAMETERS: table = 2D dataset
				column_index = int representing index of attribute
    RETURNS: NA
    '''
    column_avg = np.mean(get_column(table, column_index))
    for row in table:
        if row[column_index] == 'NA':
            row[column_index] = round(column_avg, 1)

def get_frequencies(table, column_index):
    '''
    Get frequency of values of an attribute
    PARAMETERS: table = 2D dataset
                column_index = int representing index of attribute
    RETURNS: freq = dictionary with keys indicating classification
					and values of keys indicating frequency count
    '''
    column = get_column(table, column_index)
    freqs = dict()

    for value in column:
        if value not in freqs:
            # first time we have seen this vlaue
            freqs[value] = 1
        else: # we've seen it before, the list is sorted...
            freqs[value] +=1

    return freqs

def compute_equal_widths_cutoffs(values, num_bins):
    '''
    Compute bin ranges
    PARAMETERS: values = values of the existing attribute
                num_bins = number of bins to divide data into
    RETURNS: cutoffs = ranges representing bins
    '''
    # Compute the width using the range
    values_range = max(values) - min(values)
    width = values_range / num_bins
    cutoffs = list(np.arange(min(values) + width, max(values) + width, width))
    # Round each cutoff to whole numbers before we return it
    cutoffs = [int(round(cutoff, 0)) for cutoff in cutoffs]
    return cutoffs

def compute_bin_frequencies(values, cutoffs):
    '''
    Compute frequencies of values that occur for each bin range
    PARAMETERS: values = values of the exisiting attribute
                cutoffs = ranges of bins
    RETURNS: freqs = list of frequencies corresponding to each bin
    '''
    freqs = [0] * len(cutoffs)
    for val in values:
        for i, cutoff in enumerate(cutoffs):
            if val <= cutoff:
                freqs[i] += 1
                break
    return freqs

def group_by(table, column_index, include_only_column_index=None):
    '''
    Categorize data by continuous value
    PARAMETERS: table = 2D dataset
                column_index = int representing index of attribute
    RETURNS: group_names = categories of dataset
             groups = values existing within given category
    '''
    # Identify unique values in the column
    group_names = sorted(list(set(get_column(table, column_index))))
    # List of subtables corresponding to a value in group_names
    groups = [[] for name in group_names]
    for row in table:
        group_by_value = row[column_index]
        index = group_names.index(group_by_value)
        if include_only_column_index is None:
            groups[index].append(row.copy())       # Shallow copy
        else:
            groups[index].append(row[include_only_column_index])

    return group_names, groups

def discretize_data(table, attr, attr_indexes, attr_domains):
    '''
    Convert categorical data into continuous data based on values
    attr_domains
    PARAMETERS: table = 2D list of data instances
                attr = list of attributes
                attr_indexes = indexes of attributes to be discretized
                attr_domains = dictionary of domain values for attributes
    RETURNS: table with new discretized attribute values
    '''
    for a in attr:
        attr_index = attr.index(a)
        attr_col = get_column(table, attr_index)
        if all(isinstance(n, int) for n in attr_col) == False:
            attr_domain = attr_domains[a]
            sorted(attr_domain)
            for row in table:
                if row[attr_index] != '':
                    row[attr_index] = attr_domain.index(row[attr_index])
                else:
                    del row
            attr_domains[a] = attr_domain

#######################################
# STATS CALCULATIONS
#######################################
def calculate_mean_std(data):
	'''
	Calculate the mean and standard deviation of a 1D dataset
	PARAMETERS: data = 1D list of data values
	RETURNS: mean = mean of dataset
			 std = standard deviation of dataset
	'''
	mean = sum(data) / len(data)
	mean_diff_squared = [(x - mean) ** 2 for x in data]
	std = np.sqrt(sum(mean_diff_squared) / len(mean_diff_squared))

	return mean, std

def calculate_linear_regression(x, y):
	'''
	Calculate the m and b values of a linear regression model based
	on a list of x and y values to be paired.
	PARAMETERS: x = x-values of the dataset to base linear regression on
				y = y-values of the dataset to base linear regression on
	RETURNS: m = slope of linear regression model
			 b = y-intercept of linear regression model
	'''
	mean_x, _ = calculate_mean_std(x)
	mean_y, _ = calculate_mean_std(y)

	m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
	b = mean_y - m * mean_x

	return m, b

def calculate_correlation_coeff(x, y):
	'''
	Calculate the correlation coefficient of a dataset of paired x and y values
	PARAMETERS: x = x-values of the dataset
				y = y-values of the dataset
				y_pred = predicted y-avlue of the data pairing for said x-value
	RETURNS: r = correlation coefficient for the dataset
	'''
	mean_x, _ = calculate_mean_std(x)
	mean_y, _ = calculate_mean_std(y)

	r = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / np.sqrt(sum([(x[i] - mean_x) ** 2 for i in range(len(x))]) * sum([(y[i] - mean_y) ** 2 for i in range(len(y))]))

	return r

def calculate_stderr(y, y_pred):
	'''
	Calculate the standard error of a dataset of y values
	PARAMETERS: y = y-values of the dataset
				y_pred = predicted y-avlue of the data pairing for said x-value
	RETURNS: stderr = standard error for the dataset
	'''
	stderr = np.sqrt(sum([(y[i] - y_pred[i]) ** 2 for i in range(len(y))]) / len(y))

	return stderr

def calculate_mean_abs_error(y_unseen, y_predict):
	'''
	Calculate the mean absolute error (MAE) value of a predictors predictions
	PARAMETERS: y_unseen = the known y-value of a data pairing (based on an x unseen)
				y_predict = the predicted y-value of a data pairing (based on an x unseen)
	RETURNS: mae = mean absolute error of predictors predictions
	'''
	residuals = [abs(y_unseen[i] - y_predict[i]) for i in range(len(y_unseen))]
	mae = sum(residuals) / len(residuals)

	return mae

#######################################
# SUPERVISED LEARNING
#######################################
def compute_distance(v1, v2, attr):
    '''
    Calculate Euclidean distance between two points in n-space
    PARAMETERS: v1 = point1 with n attributes
                v2 = point2 with n attributes
                attr = list of indicies of attributes to predict on
    RETURNS: dist = length of straight line joining points
    '''
    assert(len(v1) == len(v2))
    dist = math.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in attr]))
    return dist

def normalize(data, attr):
    '''
    Normalize attribute values by scaling values between 0 and 1
    PARAMETERS: data = 2D list of attributes
                attr = attribute indicies of data that need to be normalized on
    RETURNS: 2D list of attributes normalized on attributes with indicies
             listed in attr
    '''
    for i in attr:
        xs = get_column(data, i)
        for row in data:
            row[i] = (row[i] - min(xs)) / ((max(xs) - min(xs)) * 1.0)

def kNN_classifier(training_set, attr, n, instance, k, discretization=None):
    '''
    Classify an instance based on classification of k closest instances in training set
    PARAMETERS: training_set = 2D list of training instances
                attr = list of attribute indicies to predict on
                n = classification label index
                instance = unknown instance to classify
                discretization = possible dicretization function
                k = k-value for kNN
    RETURNS: class_label = most common ranking of k closest instances
    '''
    labels = dict()
    row_distances = []
    # Calculate and store distance of each instance in training set relative to instance
    for row in training_set:
        d = compute_distance(row, instance, attr)
        row_distances.append([d, row])

    row_distances.sort(key=operator.itemgetter(0)) # Sort by distance caluclated
    top_k_rows = row_distances[:k] # Sample top k instances

    if discretization is None:
        pass
    else:
        # Replace distance label with classifier of instance
        for row in top_k_rows:
            row[0] = discretization(row[1][n])

    # Get frequencies of each class rank in top k sampling
    labels = get_frequencies(top_k_rows, 0)

    # Find largest frequency value and return matching class rank key label
    k = list(labels.keys())
    v = list(labels.values())
    class_label = k[v.index(max(v))]

    return class_label

def randomize_data(table):
	'''
	Shuffle table
	PARAMETERS: table = 2D list of data set
	RETURNS: table with shuffled indicies
	'''
	randomized = table[:]			# Copy of the table
	n = len(randomized)
	for i in range(n):
		j = random.randrange(0, n)	# Pick random index in [0, n) to swap
		randomized[i], randomized[j] = randomized[j], randomized[i]
	return randomized

def compute_holdout_partitions(table, train, test):
	'''
	Randomly divide data set into train and test sets
	PARAMETERS: table = 2D list of data set
				train = int representing parition of data to be trained
				test = int representing partiion of data to be held for testing
	RETURNS: 2D list of training dataset, 2D list of testing dataset
	'''
	# Randomize the table
	randomized = randomize_data(table)
	# Split train and test sets
	split_index = int(train / (train + test) * len(table))	# Train : Total (Train and Test) partition
	return randomized[0:split_index], randomized[split_index:]

def stratify_data(table, column_index, k, discretization=None):
    '''
    Stratify data set into k partitions
    Sorting folds based on the classification of column_index by discretization
    PARAMETERS: table = 2D list of data set
                column_index = int index of table to be classified on
                k = number of folds to be separated into
                discretization: function that will classify column_index of table
    RETURNS: stratified_table = 3D table, where each instance represents an equal fold
    '''
    table = randomize_data(table)
    classified_table = []
    stratified_table = [[] for _ in range(k)]
    index = 0

    if discretization is None:
        classes, class_values = group_by(table, column_index)
        for i in range(len(classes)):
            for j in class_values[i]:
                stratified_table[index % len(stratified_table)].append(j)
                index += 1
    else:
        for row in table:
            classified_table.append([discretization(row[column_index]), row])
        classes, class_values = group_by(classified_table, 0)
        for i in range(len(classes)):
            for j in class_values[i]:
                stratified_table[index % len(stratified_table)].append(j[1])
                index += 1

    return stratified_table

#######################################
# NAIVE BAYES
#######################################
def prior_post_probabilities(data, attributes, class_index, given_conditions=None):
    '''
	Create a Naive Bayes Classifier for dataset to predict classifier for test instance
	PARAMETERS: data = 2D data set of training values
				attributes = 1D list of attibute/column names
				class_index = int representing index of attribute to classify test instance on
				given_conditions = indicies of attributes to train on in dataset
	RETURNS: classes = class labels
			 conditions = attributes trained on
			 prior_probabilities = prior probabilities of dataset
			 post_probabilities = posterior probabilities of dataset
    '''
    prior_probabilities = dict()
    post_probabilities = dict()

    if given_conditions is None:
        conditions = set(range(len(attributes)))
        conditions.remove(class_index)
    else:
        conditions = given_conditions

    classes, class_data = group_by(data, class_index)

	# Grouping data by class label
    for curr_class_data in class_data:
        class_total = len(curr_class_data)	# denominator of probability; number of inst with class label
        class_label = curr_class_data[0][class_index]
        prior_probabilities[class_label] = class_total

	# Sorting instances of training set by attribute labelings
    for inst in data:
        for attr in conditions:
            if (attributes[attr] + " = " + str(inst[attr]), attributes[class_index] + " = " + str(inst[class_index])) in post_probabilities:
                post_probabilities[attributes[attr] + " = " + str(inst[attr]), attributes[class_index] + " = " + str(inst[class_index])] += 1
            else:
                post_probabilities[attributes[attr] + " = " + str(inst[attr]), attributes[class_index] + " = " + str(inst[class_index])] = 1

	# Reduce post and posterior probabilities to probabilities
    for key in post_probabilities:
        class_label = key[1].split("= ", 1)
        try:
            class_label = float(class_label[1])     # attribute is a numeric value
        except ValueError:
            class_label = class_label[1]
        post_probabilities[key] = post_probabilities[key] / prior_probabilities[class_label]
    for key in prior_probabilities:
        prior_probabilities[key] = prior_probabilities[key] / len(data)

    return classes, conditions, prior_probabilities, post_probabilities

def naive_bayes(data, classes, conditions, attributes, prior_probabilities, post_probabilities, test, class_index, apply_gaussian=False, g_index=None):
    '''
    Classify test instance based on prior and posterior probabilities of dataset
    PARAMETERS: data = 2D list of dataset
                classes = class labels
                conditions = attributes data was trainined on
                prior_probabilities = prior probabilities calculated
                post_probabilities = post probabilities calculate
                test = test instance with list of attribute values
                class_index = int representing index of attribute to classify test instance on
                apply_gaussian = boolean determining if attribute should be treated as continous or categorical property
                g_index = index of Gaussian attribute if exists
    RETURNS: classified label of test instance
    '''
    class_probabilities = dict()

    for class_label in classes:
        class_probabilities[class_label] = prior_probabilities[class_label]
        if apply_gaussian == True:
            class_data = [row for row in data if row[class_index] == class_label]
            mean, stdev = calculate_mean_std(get_column(class_data, g_index))
            class_probabilities[class_label] *= gaussian(test[g_index], mean, stdev)
        for attr in conditions:
            attr_label = test[attr]
            if (attributes[attr] + " = " + str(attr_label), attributes[class_index] + " = " + str(class_label)) in post_probabilities:
                class_probabilities[class_label] *= post_probabilities[attributes[attr] + " = " + str(attr_label), attributes[class_index] + " = " + str(class_label)]
            else:
                class_probabilities[class_label] *= 0

    k = list(class_probabilities.keys())
    v = list(class_probabilities.values())

    return k[v.index(max(v))]

def gaussian(x, mean, stdev):
	'''
	Using a continuous attribute sampled from a Gaussian distribution, apply Naive Bayes
	PARAMETERS: x = instance being sampled
				mean = mean of the attribute for all instances labeled a specific class_index
				stdev = standard deviation of the attribute for all instances labeled a specific class_index
	'''
	first, second = 0, 0
	if stdev > 0:
		first = 1 / (math.sqrt(math.pi) * stdev)
		second = math.e ** (-((x - mean) ** 2) / (2 * (stdev ** 2)))
	return first * second

#######################################
# DECISION TREES
#######################################
def get_attr_domains(table, attr, attr_indexes):
    '''
    Gather all possible values (domains) for given attributes
    PARAMETERS: table = 2D list of data set
                attr_indexes = list of attribute column indexes to gather
    RETURNS: attr_domains = dictionary containing attribute domains per index
    '''
    attr_domains = dict()
    for i in attr_indexes:
        attr_col = get_column(table, i)
        attr_index = attr[i]
        attr_domains[attr_index] = list(set(attr_col))

    return attr_domains

def calculate_entropy(instances, class_index):
    '''
    Calculate domain entropy of a given attribute
    PARAMETERS: instances = list of instances containing specified domain for attribute
                class_index = column index of classification attrubute
    RETURNS: entropy = entropy value of attribute values
    '''
    entropy = 0

    class_domains, class_groups = group_by(instances, class_index)

    for domain_index in range(len(class_domains)):
        domain_count = len(class_groups[domain_index])
        entropy -= (domain_count / len(instances)) * math.log(domain_count / len(instances), 2)

    return entropy

def attribute_entropy(instances, attr_index, class_index):
    '''
    Calculate entropy of all domains of a given attribute
    PARAMETERS: instances = list of instances to classify
                attr_index =  column index of attribute to calculate entropy of
                class_index = column index of classifier
    RETURN: e_new = entropy of given attribute
    '''
    e_new = 0

    attr_domains, attr_groups = group_by(instances, attr_index)

    for domain_index in range(len(attr_domains)):
        domain_entropy = calculate_entropy(attr_groups[domain_index], class_index)
        e_new += (len(attr_groups[domain_index]) / len(instances)) * domain_entropy

    return e_new

def tdidt(instances, attr_indexes, attr_domains, class_index, header=None, chosen_attr=False):
    '''
    Create a TDIDT classifier using entropy to select splitting attributes
    PARAMETERS: instances = 2D list of data set instances
                attr_indexes = list of attribute indexes to classify instances on
                attr_domains = list of dictionaries containing domains of attributes
                class_index = column index of classifier
                header = list of names for columns/attributes of instances
    RETURN: sub_tree = tdidt
    '''
    # Pick attribute ("attribute selection")
    if not chosen_attr:
        attr_index = select_attribute(instances, attr_indexes, class_index)
        attr_indexes.remove(attr_index)
    else:
        attr_index = attr_indexes.pop(0)

    # Parition data by attribute values
    partition = partition_instances(instances, attr_index, attr_domains[attr_index])

    case3 = False
    sub_tree = ["Attribute", header[attr_index]]
    for partition_label in partition:
        partition_label_instances = partition[partition_label]
        if len(partition_label_instances) == 0:
            # Case 3: No more instances to partition
            case3 = True
            break
        else:
            if check_all_same_class(partition_label_instances, class_index):
                # Case 1: Partition has only class labels that are the same
                node = ["Leaves", [partition_label_instances[0][class_index], len(partition_label_instances), count_partition(partition), round(len(partition_label_instances) / count_partition(partition), 2)]]
            elif len(attr_indexes) == 0:
                # Case 2: No more attributes to partiton
                label, _ = compute_partition_stats(partition_label_instances, class_index)
                node = ["Leaves", [label, len(partition_label_instances), count_partition(partition), round(len(partition_label_instances) / count_partition(partition), 2)]]
            else:
                new_attr_indexes = attr_indexes[:]
                node = tdidt(partition_label_instances, new_attr_indexes, attr_domains, class_index, header)

        value_list = ["Value", partition_label, node]
        sub_tree.append(value_list)

    # If case 3, attribute => leaf node
    if case3:
        label = compute_partition_voting(partition, class_index)
        sub_tree = ["Leaves", [label, count_partition(partition), count_partition(partition), round(count_partition(partition) / count_partition(partition), 2)]]

    return sub_tree

def select_attribute(instances, attr_indexes, class_index):
    '''
    Select attribute based on maximum information gain (smallest e_new)
    PARAMETERS: instances = 2D list of dataset instances
                attr_indexes = list of attribute indexes to classify instances on
                class_index = column index of classifier
    RETURN: attr_index giving maximum information gain
    '''
    e_start = calculate_entropy(instances, class_index)
    gains = dict()

    for attr in attr_indexes:
        e_new = attribute_entropy(instances, attr, class_index)
        gains[attr] = e_start - e_new

    # Select attribute with maximum information gain
    k = list(gains.keys())
    v = list(gains.values())
    return k[v.index(max(v))]

def partition_instances(instances, attr_index, attr_domain):
    '''
    Seperate instances by attribute domain
    PARAMETERS: instances = 2D list of data set instances
                attr_indexes = list of attribute indexes to classify instances on
                attr_domains = list of dictionaries containing domains of attributes
    RETURNS: partition = dictionary containing attribute domain as key and list
                list of instances under that domain that match
    '''
    partition = {}
    for attr_value in attr_domain:
        subinstances = []
        for instance in instances:
            # Check if this instance has attr_value at attr_index
            if instance[attr_index] == attr_value:
                subinstances.append(instance)
        partition[attr_value] = subinstances
    return partition

def check_all_same_class(instances, class_index):
    '''
    Check if instances hold same classifying label
    PARAMETERS: instances = 2D list of data set instances
                class_index = column index of classifier
    RETURNS: True if all instances have same label; False otherwise
    '''
    same_class = True
    class_label = instances[0][class_index]

    for instance in instances:
        if instance[class_index] == class_label:
            pass
        else:
            same_class = False
            break
    return same_class

def count_partition(partition_instances):
    '''
    Count instances that occur in a partition
    PARAMETERS: partition_instances = sictionary of instances in a partition label
    RETURNS: sum of instances
    '''
    return sum(len(partition_instances[partition]) for partition in partition_instances)

def compute_partition_stats(instances, class_index):
    '''
    Calculate majority voting within a partition label
    PARAMETERS: instances = list of data instances under partition label
                class_index = column index of classifying attribute
    RETURNS: class_label with majority instances, majority percentage
    '''
    partition_stats = dict()
    for inst in instances:
        label = inst[class_index]
        if label in partition_stats.keys():
            partition_stats[label] += 1
        else:
            partition_stats[label] = 1

    k = list(partition_stats.keys())
    v = [val / len(instances) for val in partition_stats.values()]
    return k[v.index(max(v))], max(v)

def compute_partition_voting(partitions, class_index):
    '''
    Calculate majority voting within a partition (due to domain labels missing -- CASE3)
    PARAMETERS: partition = list of data instances under partition label
                class_index = column index of classifying attribute
    RETURNS: class_label with majority instances
    '''
    votes = dict()
    for part in partitions:
        if len(partitions[part]) != 0:
            label, vote = compute_partition_stats(partitions[part], class_index)
            if label in votes.keys():
                votes[label] += vote
            else:
                votes[label] = vote
    k = list(votes.keys())
    v = list(votes.values())
    return k[v.index(max(v))]

def classify_tdidt(decision_tree, instance, header):
    '''
    Trace TDIDT to find classification of a given instance
    PARAMETERS: dection_tree = TDIDT
                instance = list of attributes to classify instances
                header = list of attribute labels
    RETURNS: classification label
    '''
    if 'Leaves' in decision_tree[0]:
        return decision_tree[1][0]   # label of leaf instance
    else:
        attr = decision_tree[1]
        attr_index = header.index(attr)
        val = instance[attr_index]

        for v in range(2, len(decision_tree)):
            if decision_tree[v][1] == val:
                return classify_tdidt(decision_tree[v][2], instance, header)

def tdidt_stratify_and_confusion_matrix(data, attributes, attr_indexes, attr_domains, class_index, k, discretization=None):
    '''
    Using stratified k-fold cross-validation, created a TDIDT classifier and generate confusion matrix of results
    PARAMETERS: data = 2D list of data set instances
                attributes = list of attribute labels
                attr_indexes = list of attribute indexes to classify instances on
                attr_domains = list of dictionaries containing domains of attributes
                class_index = column index of classifying attribute
                k = k-value
                discretization = function to discretize categorical data
    RETURNS: accuracy and printed confusion matrix of classifying results
    '''
    if discretization is None:
        class_labels = list(get_attr_domains(data, [class_index]).values())
        class_labels = class_labels[0]
    else:
        class_labels = list(range(1, 10))

    confusion_matrix = {(i , j): 0 for i in class_labels for j in class_labels}

    sample_accuracies = []
    stratified_data = stratify_data(data, class_index, k, discretization)

    # Perform k-folds of Cross Validation
    for i in range(k):
        # Establish train and test sets of partitions
        train_set = []
        test_set = []
        test_attr_indexes = attr_indexes[:]
        test_attr_domains = attr_domains.copy()
        tp_tn = 0

        test_set = stratified_data.pop(i)
        for j in stratified_data:
            train_set.extend(j)

        # Calculate TDIDT classifier
        decision_tree = tdidt(train_set, test_attr_indexes, test_attr_domains, class_index, attributes)

        # Classify partition test set
        for inst in test_set:
            pred_class = classify_tdidt(decision_tree, inst, attributes)
            actual_class = inst[class_index]

            if discretization is not None:
                pred_class = discretization(pred_class)
                actual_class = discretization(actual_class)

            if pred_class == actual_class:
                tp_tn += 1

            confusion_matrix[(actual_class, pred_class)] += 1

        # Store accuracy of classifier for subsample
        sample_accuracies.append(tp_tn / len(test_set))
        # Reset train and test sets
        stratified_data.insert(i, test_set)

    accuracy, _ = calculate_mean_std(sample_accuracies)
    print("\nACCURACY:", accuracy, end='\n\n')

    # Format Confusion Matrix
    data_table = [[0 for _ in class_labels] for _ in class_labels]
    for key in confusion_matrix.keys():
        row = class_labels.index(key[0])
        col = class_labels.index(key[1])
        data_table[row][col] = confusion_matrix[key]

    data_table = format_confusion_table(data_table, len(class_labels), class_labels)
    headers = class_labels
    headers.append("Total")
    headers.append("Recognition (%)")

    print(tabulate(data_table, headers, tablefmt='rst'))
    print()

def tdidt_rules(decision_tree, attributes, class_index, path=[]):
    '''
    '''
    if 'Leaves' in decision_tree[0]:
        rule_str = ''
        for index in range(0, len(path) - 1, 2):
            if rule_str != '':
                rule_str += ' AND '
            rule_str += 'IF ' + str(path[index]) + ' == ' + str(path[index + 1])
        rule_str += ' THEN ' + str(attributes[class_index]) + ' = ' + str(decision_tree[1][0])
        print(rule_str)
    else:
        attr = decision_tree[1]
        path.extend([attr])
        for v in range(2, len(decision_tree)):
            rule = path[:]
            rule.extend([decision_tree[v][1]])
            sub_tree = decision_tree[v][2]
            tdidt_rules(sub_tree, attributes, class_index, rule)

def pretty_print_tree(decision_tree):
    '''
    Nice print of tree
    '''
    tab_count = -1
    tree_str = str(decision_tree)
    for ch in tree_str:
        if ch == '[':
            tab_count += 1
            print()
            print('\t' * tab_count, ch, end='')
        elif ch == ']':
            tab_count -= 1
            print()
            print('\t' * tab_count, ch, end='')
        else:
            print(ch, end='')
    print('\n')

#######################################
# ENSEMBLE LEARNING
#######################################
def get_random_attr_subset(attributes, F):
    '''
    Randomly select F attributes to train on given list of available attributes
    PARAMETERS: attributes = list of attribute indexes to be used to train tree
                F = number of attributes used to generate tree
    RETURNS: slice of list with F attribute indexes
    '''
    shuffled = attributes[:]    # make a copy
    random.shuffle(shuffled)
    return shuffled[:F]

def compute_bootstrapped_sample(table):
    '''
    Generate bootstrap sample of instances
    PARAMETERS: table = 2D list of instances
    RETURNS: sample = 2D list of instances randomly selected from table with same
                        number of instances
    '''
    n = len(table)
    sample = []
    for _ in range(n):
        rand_index = random.randrange(0, n)
        sample.append(table[rand_index])

    return sample

def random_forest(table, attr_indexes, attr_domains, class_index, col_names, N, M, F):
    '''
    Generate random forest
    PARAMETERS: table = 2D list of data set instances
                attr_indexes = list of attribute indexes to classify instances on
                attr_domains = list of dictionaries containing domains of attributes
                class_index = column index of classifier
                col_names = list of names for columns/attributes of instances
                N = number of bootstrap samples to be generated
                M = number of top decision trees to be selected
                F = number of attributes to train tree on
    RETURNS: top_m_classifiers = top M classifiers (forest ensemble)
    '''
    # M best classifiers
    top_m_classifiers = [0 for _ in range(M)]
    top_m_accuracies = [-1 for _ in range(M)]

    # Creating N bootstrap samples (from remainder set)
    for _ in range(N):
        # Bootstrap remainder set into training and validation set
        train_set = compute_bootstrapped_sample(table)
        validation_set = [x for x in table if x not in train_set]

        # Build classifier for sample
        attr_index_subset = get_random_attr_subset(attr_indexes, F)
        tree = tdidt(train_set, attr_index_subset, attr_domains, class_index, col_names)

        # Determine accuracy of classifier
        tp_tn = 0
        for v in validation_set:
            pred = classify_tdidt(tree, v, col_names)
            actual = v[class_index]
            if pred == actual:
                tp_tn += 1
        classifier_accuracy = tp_tn / len(validation_set)

        # Check if classifier qualifiers for top M classifiers
        if classifier_accuracy > min(top_m_accuracies):
            acc_index = top_m_accuracies.index(min(top_m_accuracies))
            top_m_accuracies[acc_index] = classifier_accuracy
            top_m_classifiers[acc_index] = tree

    return top_m_classifiers

#######################################
# APRIORI ALGORITHM
#######################################
def prepend_attribute_labels(table, col_names):
    '''
    Preappend attribute labels before each attribute value in each row of the table
    PARAMETERS: table = 2D list of instances with attribute values
                col_names = list of attribute labels
    RETURNS: altered table state with attribute labels added before each values
    '''
    for row in table:
        for i in range(len(col_names)):
            row[i] = str(col_names[i]) + "=" + str(row[i])

def convert_nominal_attr(table, attr_keys):
    '''
    Convert table dataset from keys to nominal attribute values
    PARAMETERS: table = 2D list of table instances
                attr_keys = 2D list of key value pairs for each attribute instance
    RETURNS: table with update attribute values (nominal)
    '''
    # Convert list of nominal attribute domains into dictionary per attribute
    for attr in attr_keys:
        nominal_attr = dict()
        for key in attr:
            pair = key.split("=")
            nominal_attr[pair[1]] = pair[0]

        attr_index = attr_keys.index(attr)
        attr_keys[attr_index] = nominal_attr

    # Convert all table values to their corresponding nominal attribute values:
    for row in table:
        for i in range(len(row)):
            attr_domain = attr_keys[i]
            attr_val = attr_domain[row[i]]
            row[i] = attr_val

def check_row_match(terms, row):
    '''
    Check that rule contains all attribute values of given rule set (LHS or RHS)
    exists in table instance
    PARAMETERS: terms = rule set to be looking for in given attribute values existence
                row = table instance to be looking through
    RETURN: True if terms is a subset of row, False otherwise
    '''
    for term in terms:
        if term not in row:
            return 0
    return 1

def compute_unique_values(table):
    '''
    Get I representing possible values of itemset
    PARAMETERS: table = 2D list of attribute instances
    RETURNS: unique_values = sorted list of pssible attribute values
    '''
    unique_values = set()
    for row in table:
        for value in row:
            unique_values.add(value)
    return sorted(list(unique_values))

def compute_itemset_support(itemset, table, minsup):
    '''
    Given a canadite set of itemsets, evaluate the relationship of each
    itemset to the table to see if it holds as a valid itemset meeting minsup
    PARAMETERS: itemset = Ck, list of itemsets generated as canidate set
                table = 2D list of instances
                minsup = minimum support value accepted
    RETURNS: supported_itemsets = Lk, list of supported itemsets
    '''
    supported_itemsets = []
    for item in itemset:
        count = 0
        for row in table:
            row = set(row)
            if item.issubset(row):
                count += 1
        # Check that support of itemset meets minsup
        if count/len(table) >= minsup:
            supported_itemsets.append(item)

    return supported_itemsets

def compute_rule_counts(rule, table):
    '''
    Given a rule, count the number of times it appears in a table
    PARAMETERS: rule = LHS and RHS itemsets of a rule
                table = 2D list of instances
    RETURNS: rule = LHS and RHS itemsets, as well as table count for
                    each side, both, and total
    '''
    N_left = N_right = N_both = 0
    N_total = len(table)

    for row in table:
        N_left += check_row_match(rule["lhs"], row)
        N_right += check_row_match(rule["rhs"], row)
        N_both += check_row_match(rule["lhs"] + rule["rhs"], row)

    rule["N_left"] = N_left
    rule["N_right"] = N_right
    rule["N_both"] = N_both
    rule["N_total"] = N_total

def compute_rule_interestingness(rule, table):
    '''
    Calculate the confidence, support, completeness, and lift of a rule in a table
    PARAMETERS: rule = LHS and RHS itemsets of a rule
                table = 2D list of instances
    RETURNS: rule = LHS, RHS, and all interestingness values
    '''
    compute_rule_counts(rule, table)
    rule["confidence"] = rule["N_both"] / rule["N_left"]
    rule["support"] = rule["N_both"] / rule["N_total"]
    rule["completeness"] = rule["N_both"] / rule["N_right"]
    rule["lift"] = rule["N_both"] / (rule["N_left"] * rule["N_right"])

def generate_canidate_set(Lk, I, k):
    '''
    Generate canidate set given a Lk set
    PARAMETERS: Lk = Lk itemset
                I = itemset of possible attribute value pairings
                k = k value representing size of set to be generated
    RETURNS: Ck = valid canidate set with itemsets of size k
    '''
    Ck = []
    for A in Lk:
        for B in Lk:
            list_A = sorted(list(A))
            list_B = sorted(list(B))
            if list_A[0:-1] == list_B[0:-1]:
                subset = A.union(B)
                if len(subset) == k and subset not in Ck:
                    Ck.append(set(subset))

    return Ck

def generate_apriori_rules(table, supported_itemsets, minconf):
    '''
    Generate rules that hold at least minconf, using Apriori Algorithm
    PARAMETERS: table = 2D list of instances
                supported_itemset = list of supported itemsets
                minconf = minimum confidence value accepted
    RETURNS: rules = list of rules accepted by algorithm for reaching
                     minconf value
    '''
    rules = []
    for itemset in supported_itemsets:
        k = len(itemset)
        # All possible sizes of items in RHS
        for i in range(1, k):
            rhs = list(itertools.combinations(itemset, i))      # Generate all possible RHS of a given size
            # Test all rule sets for possible RHSs
            for r in rhs:
                lhs = [i for i in itemset if i not in r]        # Generate LHS as all items in itemset not in RHS
                rule = {"lhs": lhs, "rhs": list(r)}
                compute_rule_interestingness(rule, table)
                if rule["confidence"] > minconf:
                    rules.append(rule)
    return rules

def apriori(table, minsup, minconf):
    '''
    Implement Apriori Algorithm
    PARAMETERS: table = 2D list of instances
                minsup = minimum support value
                minconf = minimum confidence value
    RETURNS: rules = list of rules for dataset generated by algorithm
    '''
    supported_itemsets = []
    I = compute_unique_values(table)
    Lk = [set([item]) for item in I]
    k = 2

    while len(Lk) != 0:
        sorted(Lk)
        Ck = generate_canidate_set(Lk, I, k)
        Lk = compute_itemset_support(Ck, table, minsup)
        k += 1
        supported_itemsets.extend(Lk)

    rules = generate_apriori_rules(table, supported_itemsets, minconf)
    return rules

def pretty_print_apriori(rules):
    '''
    Pretty print Association Rules along with their support, confidence, and lift
    PARAMETERS: rules = list of rules
    '''
    header = ["Association Rule", "Support", "Confidence", "Lift"]
    table = []

    for rule in rules:
        row = []
        # Format rule appearence
        association_rule = " AND ".join(rule["lhs"])
        association_rule += " => "
        association_rule += " AND ".join(rule["rhs"])
        row.append(association_rule)
        row.append(rule["support"])
        row.append(rule["confidence"])
        row.append(rule["lift"])

        table.append(row)

    print(tabulate(table, header, tablefmt='rst'))

#######################################
# CLUSTERING
#######################################
def get_inital_centroids(data, k):
    '''
    Return initial k inital centroids
    PARAMETERS: data = 2D list of instances 
                k = k-value
    RETURNS: centroids = list of centorid instances 
    '''
    centroids = []
    for _ in range(k):
        centroids.append(data[random.randrange(len(data))])

    return centroids

def compute_centroids(clusters):
    '''
    Calculate centroid for each cluster given instances in each cluster
    PARAMETERS: clusters = 2D list of instances in each cluster
    RETURNS: centroids = new calculated centroid value for each cluster
    '''
    centroids = []
    for cluster in clusters:
        centroid = []
        for attr in range(len(cluster[0])):
            attr_col = get_column(cluster, attr)
            centroid.append(sum(attr_col)/len(attr_col))

        centroids.append(centroid)

    return centroids

def check_clusters_moved(old_centroids, centroids):
    '''
    Compare old_centorids and centroids at i to see if they moved location
    PARAMETERS: old_centroids = list of old centroid values
                centroids = list of new calculated centroid values
    RETURNS: True if centroid position moves, otherwise Flase
    '''
    for i in range(len(centroids)):
        if not old_centroids[i] == centroids[i]:
            return True
    return False

def cluster_quality(centroids, clusters):
    '''
    Utlize Total Sum of Squares (TSS) as objective function
    to measure cluster quality
    PARAMETERS: centroids = centroid value of all clusters
                clusters = 2D list of instances in each cluster
    RETURNS: TSS value for clusters
    '''
    TSS = 0
    for i in range(len(centroids)):
        cluster_qual = 0
        for inst in clusters[i]:
            for attr in range(len(centroids[i])):
                cluster_qual += (centroids[i][attr] - inst[attr]) ** 2
        TSS += cluster_qual

    return TSS

def k_means_clustering(data, attr_indexes, k):
    '''
    Perrform k-Means Clustering Classifier
    PARAMETERS: data = 2D list of instances
                attr_indexs = list of indexes to create clusters 
                k = k-value 
    RETURNS: cluster_quality = TSS value of cluster quality
             clusters = list of clustered instances
             centroids = list of centroid values for each cluster
    '''
    # Select k instances for initial centroids
    centroids = get_inital_centroids(data, k)
    clusters = [[c] for c in centroids]

    moved = True
    i = 0
    while moved:
        # Assign each instance to it's nearest cluster
        inst = data[i]
        print(inst)
        distances = []
        for c in centroids:
            dist = compute_distance(inst, c, attr_indexes)
            distances.append(dist)

        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(inst)

        # Recalculate centroid
        old_centroids = centroids
        centroids = compute_centroids(clusters)

        # Check if centroids moved to continue
        moved = check_clusters_moved(old_centroids, centroids)
        if moved:
            i += 1
            if i >= len(data):
                moved = False

    # Output Cluster Quality of Classifier, Clusters and Centroids
    return cluster_quality(centroids, clusters), clusters, centroids

'''
Functions to get a decision tree classification
'''
import random
import utils
import math
import copy

def classify_instance(header, instance_to_classify, tree):
    '''
    takes in a given instance and classifies it with a decision tree
    Returns the value of the classification (usually 'True' or 'False)
    If there is an unseen instance and an error in the tree that does
    not account for this instance, it returns 'Unable to classify'
    '''
    for branch in tree:
        #print(branch)
        if branch == "Leaves":
            return tree[1][0]

        attribute = tree[1]
        instance_index = header.index(attribute)
        if instance_to_classify[instance_index] == branch[1]:
            classification = classify_instance(header, instance_to_classify, branch[2])
        else:
            pass  
    if classification == None:
        return "Unable to classify"
    else:
        return classification
            # return what is in the leaf as the classification

def tdidt(instances, att_indexes, all_att_indexes, att_domains, class_index, header, tree):
    '''
    Uses the tdidt algorithm to build a decision tree based on a given set of data
    '''
    print("CURRENT TREE: ", tree)
    if att_indexes == []:
        return
    att_index = entropy(instances, header, att_domains, att_indexes)
    print("att_index = ", att_index, "\n")
    att_indexes.remove(att_index)
    print("att_domains = ", att_domains, "\n")
    #print(att_domains)
    #print(att_indexes)
    partition = partition_instances(instances, att_index, att_domains[att_index])
    partition_keys = partition.keys()
    
    tree.append("Attribute")
    tree.append(header[att_index])
    count = 0
    for i in range(len(att_domains[att_index])):
        tree.append(["Value", att_domains[att_index][count]])
        col = utils.get_column(partition.get(att_domains[att_index][i]), len(header)-1)
        items_in_col = []
        for item in col:
            #print("checking item: ", item)
            #print(items_in_col)
            if item not in items_in_col:
                items_in_col.append(item)
        #print("Looking at col: ", items_in_col)
        if len(items_in_col) == 1:
            tree[2+count].append(["Leaves", has_same_class_label(instances, header, att_index, class_index, col, items_in_col[0])])
        elif len(att_indexes) == 0 and len(col) > 0:
            majority_class = compute_partition_stats(col)
            tree[2+count].append(["Leaves", has_same_class_label(instances, header, att_index, class_index, col, majority_class)])
        elif col == []:
            del tree[2+count]
            return []
        else:
            tree[2+count].append([])
            new_branch = [tdidt(partition.get(att_domains[att_index][i]), att_indexes, all_att_indexes, att_domains, class_index, header, tree[2+count][2])]
            if new_branch == [[]]:                
                majority_class = compute_partition_stats(col)
                tree[2][2] = ["Leaves", has_same_class_label(instances, header, att_index, class_index, col, majority_class)]
            else:
                tree[2][2] = new_branch
        count += 1
    return tree

def find_entropy_value(table, header, attribute_index, att_domains):
    '''
    calculate the e_new value of a given attribute and returns e_new
    '''
    e_yes_values = []
    e_no_values = []
    for att in att_domains:
        num_yes = 0
        total = 0
        for instance in table:
            if instance[attribute_index] == att:
                num_yes += 1
            total += 1
        e_yes_values.append(num_yes / total)
        e_no_values.append((total - num_yes)/ total)
        e_new = 0
        for i in range(len(e_yes_values)):
            if e_yes_values[i] == 0:
                e_new += - (e_no_values[i] * math.log(e_no_values[i], 2))
            elif e_no_values[i] == 0:
                e_new += - (e_yes_values[i] * math.log(e_yes_values[i], 2))
            else:
                e_new += - (e_yes_values[i] * math.log(e_yes_values[i], 2)) - (e_no_values[i] * math.log(e_no_values[i], 2))
    return e_new

def entropy(table, header, att_domains, att_indexes):
    '''
    Using e_new values, finds the next best attribute to split 
    for in the decision tree adn returns the index of that attribute
    '''
    entropy_values = []
    max_val = 0
    max_index = 0
    if len(att_indexes) == 1:
        return att_indexes[0]
    for i in range(len(header) - 1):
        value = find_entropy_value(table, header, i, att_domains[i])
        if value != 0:
            entropy_values.append(value)
    for i in range(len(entropy_values)):
        if entropy_values[i] > max_val:
            max_val = entropy_values[i]
            max_index = i
    return att_indexes[max_index]

def partition_instances(instances, att_index, att_domain):
    '''
    Takes a set of instances and splits them into groups based on the 
    value of a specific attribute
    '''
    partition = {}
    for att_value in att_domain:
        subinstances = []
        for instance in instances:
            if instance[att_index] == att_value:
                subinstances.append(instance)
        partition[att_value] = subinstances
    return partition

def has_same_class_label(instances, header, att_index, class_index, col, classification):
    '''
    Returns the leaf node to a branch in the tree given that there is only
    classifications of the same type. 
    '''
    if "False" not in col: 
        x = len(col)
        y = len(instances)
        return [classification, x, y, x / y]
    else: 
        x = len(col)
        y = len(instances)
        return [classification, x, y, x / y]

def compute_partition_stats(partition_classes):
    '''
    Computes the maximum classification in a partition to deal with cases
    when we run out of attributes to split on and need to make a leaf node
    or when we run out of instances to classify in the next attribute and 
    need a leaf node
    '''
    possibilities = []
    possibility_counts = []
    for i in range(len(partition_classes)):
        if partition_classes[i] not in possibilities:
            possibilities.append(partition_classes[i])
            possibility_counts.append(1)
        else:
            possibility_counts[possibilities.index(partition_classes[i])] += 1
    maximum = 0 
    max_index = 0
    for i in range(len(possibilities)):
        if(possibility_counts[i] / len(partition_classes)) > maximum:
            maximum = possibility_counts[i] / len(partition_classes)
            max_index = i
    return possibilities[max_index]
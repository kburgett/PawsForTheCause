from tabulate import tabulate
import matplotlib.pyplot as plt
import utils
import decision_tree
import random
import copy

# Preprocessing Data
def preprocess():
    '''
    KEEP ATTRIBUTES:
    'date_time_intake'
    'intake_type'
    'breed_intake'
    'color_intake'
    'date_time_outcome'
    'outcome_type'
    'outcome_age'
    'gender_intake'
    'fixed_outcome'
    'age_bucket'
    'retriever'
    'shepherd'
    'beagle'
    'terrier'
    'boxer'
    'poodle'
    'rottweiler'
    'dachshund'
    'chihuahua'
    'pitbull'
    'time_bucket'

    DELETE ATTRIBUTES:
    'animal_id'
    'name_intake'
    'date_time_intake'
    'found_location'
    'animal_type_intake'
    'intake_condition'
    'month_year_intake'
    'intake_sex'
    'age'
    'breed_intake'
    'color_intake'
    'name_outcome'
    'date_time_outcome'
    'month_year_outcome'
    'outcome_subtype'
    'outcome_sex'
    'outcome_age'
    'gender_outcome'
    'fixed_intake'
    'fixed_changed'
    'date_time_length'
    '''
    attr, table = utils.parse_csv("adoption_data.csv")

    # Preserve animal entries for dogs and classifying attribute entry
    animal_index = attr.index('animal_type_intake')
    class_index = attr.index('time_bucket')
    table = [row for row in table if row[animal_index] == 'Dog' and row[class_index] != '']

    # Remove all duplicate entries
    animal_ids = set()
    animal_id_index = attr.index('animal_id')
    gender_index = attr.index('gender_intake')
    for row in table:
        # Check for duplicates
        if row[animal_id_index] in animal_ids:
            table.remove(row)
        else:
            print(row[animal_id_index])
            animal_ids.add(row[animal_id_index])
        # Check that entry has gender
        if row[gender_index] == '':
            table.remove(row)

    dogs_data = copy.deepcopy(table)
    utils.write_csv('dogs_data.csv', attr, dogs_data)

    # Remove attributes not to be trained on from instances in the dataset
    remove_attr = ['animal_id', 'name_intake', 'date_time_intake', 'found_location',
                    'animal_type_intake', 'month_year_intake', 'intake_sex', 'age', 'breed_intake', 'color_intake',
                    'name_outcome', 'date_time_outcome', 'month_year_outcome','outcome_subtype', 'outcome_sex',
                    'outcome_age', 'gender_outcome', 'fixed_intake', 'fixed_changed', 'date_time_length']

    # Remove each attribute from all rows
    for col in remove_attr:
        index = attr.index(col)
        attr.pop(index)
        for row in table:
            row.pop(index)

    utils.write_csv('clean_data.csv', attr, table)

# Naive Bayes: Kristen
def naive_bayes(table, attr, attr_indexes, class_index):
    '''
    Utilize Naive Bayes Classifier from utils.py file
    Train on cleaned data, to find accuracy of classifier.
    '''
    # Stratify data across 10 folds
    stratified_data = utils.stratify_data(table, class_index, 10)

    # Initialize data set up
    tp_tn = 0
    total = 0
    class_domains = utils.get_attr_domains(table, attr, [class_index])
    class_domains = class_domains[attr[class_index]]
    class_domains.sort(key = lambda x: x.split()[1])
    confusion_table= [[0 for _ in class_domains] for _ in class_domains]

    # Stratified folds
    for i in range(len(stratified_data)):
        train_set = []
        test_set = stratified_data.pop(i)
        total += len(test_set)
        for j in stratified_data:
            train_set.extend(j)

        # Calculate probabilities of training set
        classes, conditions, priors, posts = utils.prior_post_probabilities(train_set, attr, class_index, attr_indexes)

        # Iterate through test set
        for inst in test_set:
            # Classify predicted and actual classes
            pred_class = utils.naive_bayes(train_set, classes, conditions, attr, priors, posts, inst, class_index)
            actual_class = inst[class_index]
            if pred_class == actual_class:
                tp_tn += 1

            pred_label = class_domains.index(pred_class)
            actual_label = class_domains.index(actual_class)
            confusion_table[actual_label][pred_label] += 1

        # Return test set to stratified folds
        stratified_data.insert(i, test_set)

    # Calculate accuracy and Confusion Matrix
    acc = tp_tn / total
    confusion_matrix = utils.format_confusion_table(confusion_table, len(class_domains), class_domains)
    headers = class_domains
    headers.append("Total")
    headers.append("Recognition (%)")

    # OUTPUT
    print(class_domains)
    print("\n\nNAIVE BAYES")
    print("-" * 50)
    print("Accuracy = %f" % acc)
    print("Error Rate = %f" % (1 - acc))
    print()
    print(tabulate(confusion_matrix, headers, tablefmt='rst'))

# Decision Trees: Alana
def decision_tree_classifier(table, original_table, attr_indexes, attr_domains, class_index, header, instance_to_classify):
    '''
    Calls the functions to get a decision tree for the data and uses that decision
    tree and classifies a given instance. Returns the classification to main()
    '''
    rand_index = random.randint(0, len(table) - 1)
    instance = table[rand_index]
    tree = decision_tree.tdidt(table, attr_indexes, attr_indexes, attr_domains, class_index, header, [])
    utils.pretty_print(tree)
    classification = decision_tree.classify_instance(header, instance, tree)

    print("\n\nDECISION TREE ")
    print("-" * 50)
    print("Classifying instance: ", instance)
    print(original_table[rand_index])
    print("Classification: ", classification)

# k-Means Clustering: Kristen
def clustering(table, attr, attr_indexes, attr_domains):
    '''
    '''
    # Change categorical data into continuous data
    utils.discretize_data(table, attr, attr_indexes, attr_domains)

    # Find best k-value
    k_clusters = []
    cluster_scores = []
    for i in range(2, 10, 2):
        print("Cluster ", i, ":", end="")
        cluster_quality = utils.k_means_clustering(table, attr_indexes, i)
        k_clusters.append(i)
        cluster_scores.append(cluster_quality)
        print(cluster_quality)

    # Show k-value Cluster Qualities to determine best k
    plt.figure()
    plt.title("Best k-value")
    plt.plot(k_clusters, cluster_scores)
    plt.show()



# Ensemble Learning (kNN or Decision Trees): Alana

def main():
    '''
    Driver program
    '''
    # Preprocess and prep data to be manipulated
    #preprocess()
    attr, table = utils.parse_csv("clean_data.csv")
    utils.convert_data_to_numeric(table)

    # Gather attribute indexes, attribute domains, and classifying attribute index
    attr_indexes = list(range(len(attr)))
    class_index = attr_indexes.pop(len(attr) - 1)
    attr_domains = utils.get_attr_domains(table, attr, attr_indexes)

    # Naive Bayes
    #naive_bayes(table, attr, attr_indexes, class_index)

    # Decision Trees

    # k-Means Clustering
    # Attribute Selection
    remove_attr = ['retriever', 'shepherd', 'beagle', 'terrier', 'boxer',
                    'poodle', 'rottweiler', 'dachshund', 'chihuahua', 'pitbull']
    # Remove each attribute from all rows
    for col in remove_attr:
        index = attr.index(col)
        attr.pop(index)
        for row in table:
            row.pop(index)

    attr_indexes = list(range(len(attr)))
    attr_domains = utils.get_attr_domains(table, attr, attr_indexes)
    utils.randomize_data(table)
    table = table[:10]
    print(table)
    clustering(table, attr, attr_indexes, attr_domains)

    # Ensemble Learning (Random Forest)

if __name__ == "__main__":
    main()

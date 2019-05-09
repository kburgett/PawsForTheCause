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
    'age'
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
    'date_time_length'
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
    for row in table:
        # Check for duplicates
        if row[animal_id_index] in animal_ids:
            table.remove(row)
        else: 
            print(row[animal_id_index])
            animal_ids.add(row[animal_id_index]) 
    dogs_data = copy.deepcopy(table)
    utils.write_csv('dogs_data.csv', attr, dogs_data)

    # Remove attributes not to be trained on from instances in the dataset 
    remove_attr = ['animal_id', 'name_intake', 'date_time_intake', 'found_location', 'intake_condition', 
                    'animal_type_intake', 'month_year_intake', 'intake_sex', 'breed_intake', 'color_intake', 
                    'name_outcome', 'date_time_outcome', 'month_year_outcome','outcome_subtype', 'outcome_sex', 
                    'outcome_age', 'gender_outcome', 'fixed_intake', 'fixed_changed', 'date_time_length']

    # Remove each attribute from all rows 
    for col in remove_attr: 
        index = attr.index(col)
        attr.pop(index)
        for row in table: 
            row.pop(index)    

    utils.write_csv('clean_data.csv', attr, table)

def discretize_age(table, attr):
    '''
    '''
    age_index = attr.index('age')
    age_bucket_index = attr.index('age_bucket')
    age_bucket_domain = utils.get_attr_domains(table, attr, [age_bucket_index])
    age_bucket_domain = age_bucket_domain['age_bucket']
    
    # Bucket Keys
    years = sorted([y for y in age_bucket_domain if 'year' in y], reverse=True)
    months = sorted([m for m in age_bucket_domain if 'month' in m], reverse=True)
    weeks = sorted([w for w in age_bucket_domain if 'week' in w], reverse=True)
    days = weeks.pop(0)

    for row in table: 
        print(row[age_index], end='\t\t')
        # Days
        if 'day' in row[age_index]:
            row[age_index] = days 
        # Weeks
        if 'week' in row[age_index]:
            row[age_index] = weeks[0]
        # Months
        if 'month' in row[age_index]:
            val = row[age_index].split(' ')[0]
            for m in months:
                if val > m[0]:
                    row[age_index] = m
                    break
            row[age_index] = months[len(months) - 1]
        # Years
        if 'year' in row[age_index]:
            val = row[age_index].split(' ')[0]
            for y in years:
                if val > y[0]:
                    row[age_index] = y
                    break
            row[age_index] = years[len(years) - 1]  
         
        print(row[age_index])

    print("SOS:", age_bucket_domain)
    
    return attr, table


# Naive Bayes: Kristen
def naive_bayes(table, attr, attr_indexes, class_index): 
    '''
    '''  
    # Stratify data across 10 folds
    stratified_data = utils.stratify_data(table, class_index, 10)

    tp_tn = 0
    for fold in stratified_data:
        train_set = []
        test_set = stratified_data.pop(fold)
        for i in stratified_data:
            train_set.extend(i)
        
        # Calculate probabilities of training set 
        classes, conditions, priors, posts = utils.prior_post_probabilities(train_set, attr, class_index, attr_indexes)

        # Iterate through test set 
        for inst in test_set:
            # Classify predicted and actual classes
            pred_class = utils.naive_bayes(train_set, classes, conditions, attr, priors, posts, inst, class_index)
            actual_class = inst[class_index]

# Decision Trees: Alana
def decision_tree_classifier(table, original_table, attr_indexes, attr_domains, class_index, header, instance_to_classify):
    '''
    Calls the functions to get a decision tree for the data and uses that decision
    tree and classifies a given instance. Returns the classification to main()
    '''
    rand_index = random.randint(0, len(table) - 1)
    instance = table[rand_index]
    print("Classifying instance: ", instance)
    tree = utils.tdidt(table, attr_indexes, attr_domains, class_index, header, False)
    utils.pretty_print(tree)
    classification = decision_tree.classify_instance(header, instance, tree)
    print(original_table[rand_index])
    print("Classification: ", classification)

# k-Means Clustering: Kristen
# Ensemble Learning: kNN or Decision Trees Alana ???

def bootstrap(remainder_set):
    training_set = []
    validation_set = []
    for i in range(2 * len(remainder_set) // 3):
        random_index = random.randint(0, len(remainder_set)-1)
        training_set.append(remainder_set[random_index])
    for i in range(len(remainder_set) // 3):
        random_index = random.randint(0, len(remainder_set)-1)
        validation_set.append(remainder_set[random_index])
    return training_set, validation_set

def forest_classifier(table, att_indexes, att_domains, class_index, header, class_values, n, m):
    '''
    Calls the functions to get a decision tree for the data and uses that decision
    tree and classifies a given instance. Returns the classification to main()
    '''
    test_set, remainder_set = utils.random_test_set(table, header, 3, att_domains, class_values)
    forest = generate_forest(remainder_set, att_indexes, att_domains, class_index, header, [], n, m)
    #print(forest)
    correct_classifications = 0
    for instance in test_set:
        classifications = []
        for tree in forest:
            classifications.append(utils.classify_tdidt(tree, instance, header))
        classification = get_majority_vote(classifications)
        print(instance," classified as ", classification)
        if instance[len(instance) - 1] == classification:
            correct_classifications += 1
        
    print("Forest Accuracy: ", correct_classifications / len(test_set))

    #classification = classify_instance(header, instance_to_classify, tree)
    #return classification
def get_majority_vote(classifications):
    max_count = 0
    majority_classification = None
    classifications_set = set(classifications)
    for item in classifications_set:
        count = 0
        for classification in classifications:
            if classification == item:
                count += 1
        if count > max_count:
            majority_classification = item
    return majority_classification

def random_test_set(table, header, k, att_domains, class_values):
    '''
    Build random test and training sets. 
    The training set is 2/3 of the data
    and the test set is 1/3 of the data
    '''
    random_table = table
    random.shuffle(random_table)
    training_set = []
    test_set = []
    for i in range(2 * len(table) // 3):
        training_set.append(table[i])
    for i in range(2 * len(table) // 3, len(table)- 1):
        test_set.append(table[i])

    return test_set, training_set

def generate_forest(remainder_set, attr_indexes, attr_domains, class_index, header, tree, n, m):
    # get training and validation set
    # bootstrap method
    forest = []
    best_trees = []
    for index in range(n):
        training_set, validation_set = bootstrap(remainder_set)
        #print("TRAIN ON : ", training_set)
        #print("VALIDATE ON : ", validation_set)
        tree = utils.tdidt(training_set, attr_indexes, attr_domains, class_index, header, False)
        print(type(tree[0]))
        forest.append(tree)
    #print("FOREST: ", forest)

    for i in range(m):
        best_trees.append(forest)

    for i in range(m, len(forest)):
        for j in range(len(best_trees)):
            if find_accuracy(best_trees[j], validation_set, header) < find_accuracy(forest[i], validation_set, header):
                best_trees[j] = forest[i]
                break
    print(forest)
    #print("len(forest: ", len(forest))
    return forest

def find_accuracy(tree, validation_set, header):
    correct_classifications = 0
    for instance in validation_set:
        classification = utils.classify_tdidt(tree, instance, header)
        if classification == instance[len(instance)-1]:
            correct_classifications += 1
    return correct_classifications / len(validation_set)

def main(): 
    '''
    '''
    # Preprocess and prep data to be manipulated 
    #preprocess()
    attr, table = utils.parse_csv("clean_data.csv")
    original_attr, original_table = utils.parse_csv('dogs_data.csv')
    #attr, table = discretize_age(table, attr)
    utils.convert_data_to_numeric(table)

    # Gather attribute indexes, attribute domains, and classifying attribute index 
    attr_indexes = list(range(len(attr)))
    class_index = attr_indexes.pop(len(attr) - 1)
    attr_domains = utils.get_attr_domains(table, attr, attr_indexes)
    
    for key in attr_domains:
        print(key, attr_domains[key])
    
    #naive_bayes(table, attr, attr_indexes, class_index)

    #instance_to_classify = table[0]
    #decision_tree_classifier(table, original_table, attr_indexes, attr_domains, class_index, attr, instance_to_classify)
    class_values = utils.get_attr_domains(table, attr, [len(attr)-1])
    print(class_values)
    forest_classifier(table, attr_indexes, attr_domains, class_index, attr, class_values.get("time_bucket"), 10, 5)



if __name__ == "__main__":
    main()
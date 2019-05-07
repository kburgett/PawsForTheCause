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
    total = 0
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
            print(pred_class, "\t\t", actual_class)
            if pred_class == actual_class:
                tp_tn += 1
        
        # Return test set to stratified folds 
        stratified_data.insert(i, test_set)
    
    acc = tp_tn / total
    print("\n\nNAIVE BAYES")
    print("-" * 50)
    print("Accuracy = %f" % acc)
    print("Error Rate = %f" % (1 - acc))
            

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
def clustering(table, attr, attr_indexex, class_index):
    '''
    '''
    
# Ensemble Learning: kNN or Decision Trees Alana ??? 

def main(): 
    '''
    '''
    # Preprocess and prep data to be manipulated 
    preprocess()
    attr, table = utils.parse_csv("clean_data.csv")
    original_attr, original_table = utils.parse_csv('dogs_data.csv')
    #attr, table = discretize_age(table, attr)
    utils.convert_data_to_numeric(table)
    
    # Gather attribute indexes, attribute domains, and classifying attribute index 
    attr_indexes = list(range(len(attr)))
    class_index = attr_indexes.pop(len(attr) - 1)
    attr_domains = utils.get_attr_domains(table, attr, attr_indexes)
    
    naive_bayes(table, attr, attr_indexes, class_index)
    
    #instance_to_classify = table[0]
    #decision_tree_classifier(table, original_table, attr_indexes, attr_domains, class_index, attr, instance_to_classify)

if __name__ == "__main__":
    main()
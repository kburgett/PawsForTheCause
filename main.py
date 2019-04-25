import utils 
import decision_tree

# Preprocessing Data 
def preprocess():
    '''
    KEEP ATTRIBUTES:
    'animal_id'
    'intake_type'
    'animal_type_intake'
    'age'
    'outcome_type'
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
    'name_intake'
    'date_time_intake'
    'found_location'
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

    # Remove animals that are not dogs 
    animal_index = attr.index('animal_type_intake')
    for row in table:
        if row[animal_index].strip().lower() != 'dog':
            table.remove(row)  

    # Remove all duplicate entries of animals
    animal_ids = set()
    animal_id_index = attr.index('animal_id')
    for row in table: 
        if row[animal_id_index] in animal_ids:
            table.remove(row)
        else: 
            animal_ids.add(row[animal_id_index])      

    # Remove attributes not to be trained on from instances in the dataset 
    remove_attr = ['name_intake', 'date_time_intake', 'found_location', 'intake_condition', 
                    'month_year_intake', 'intake_sex', 'breed_intake', 'color_intake', 'name_outcome', 
                    'date_time_outcome', 'month_year_outcome','outcome_subtype', 'outcome_sex', 
                    'outcome_age', 'gender_outcome', 'fixed_intake', 'fixed_changed', 'date_time_length']

    # Remove each attribute from all rows 
    for col in remove_attr: 
        index = attr.index(col)
        attr.pop(index)
        for row in table: 
            row.pop(index)    

    utils.write_csv('clean_data.csv', attr, table)

# Naive Bayes Kristen
def naive_bayes(attr, data): 
    '''
    '''
    class_index = attr.index('time_bucket')
    
    # Stratify data across 10 folds
    stratified_data = utils.stratify_data(data, class_index, 10)

# Decision Trees Alana
# k-Means Clustering Kristen
# Ensemble Learning: kNN or Decision Trees Alana ??? 

def main(): 
    '''
    '''
    preprocess()
    '''
    header, table = utils.parse_csv('clean_data.csv')

    att_domains = utils.get_attr_domains(table, list(range(len(header))))
    domain_header = ["intake_type", "gender_intake", "outcome_type", "fixed_outcome", "age_bucket", "retriver", "shepard",
                     "beagle", "terrier", "boxer", "poodle", "rottweiler", "dachshund", "chihuahua", "pibull"]

    att_indexes = list(range(14))
    class_index = len(header) - 1

    instance_to_classify = table[0]
    decision_tree_classifier(table, att_indexes, att_domains, class_index, domain_header, instance_to_classify)
    '''

def decision_tree_classifier(table, att_indexes, att_domains, class_index, header, instance_to_classify):
    '''
    Calls the functions to get a decision tree for the data and uses that decision
    tree and classifies a given instance. Returns the classification to main()
    '''
    tree = decision_tree.tdidt(table, att_indexes, att_indexes, att_domains, class_index, header, [])
    #print("Tree: ", tree)
    #classification = decision_tree.classify_instance(header, instance_to_classify, tree)
    #return classification

if __name__ == "__main__":
    main()
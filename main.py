import utils 
import decision_tree

# Preprocessing Data 
def preprocess():
    '''
    KEEP ATTRIBUTES:
    'animal_id'
    'date_time_intake'
    'intake_type'
    'animal_type_intake'
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
    'name_intake'
    'found_location'
    'intake_condition'
    'month_year_intake'
    'intake_sex'
    'name_outcome'
    'month_year_outcome'
    'outcome_subtype'
    'outcome_sex'
    'gender_outcome'
    'fixed_intake'
    'fixed_changed'
    '''
    attr, table = utils.parse_csv("adoption_data.csv")

    # Remove attributes not to be trained on from instances in the dataset 
    remove_attr = ['name_intake', 'found_location', 'intake_condition', 'month_year_intake', 'intake_sex', 'name_outcome', 
        'month_year_outcome', 'outcome_subtype', 'outcome_sex', 'gender_outcome', 'fixed_intake', 'fixed_changed',
        'breed_intake', 'color_intake', 'animal_id', 'date_time_intake', 'date_time_outcome', 'outcome_age', 'date_time_length']

    for col in remove_attr: 
        index = attr.index(col)
        attr.pop(index)
        for row in table: 
            row.pop(index)
    
    
    # Remove all animals that are not dogs 
    animal_index = attr.index('animal_type_intake')
    print(animal_index)
    print(table[:2])
    new_table = []
    for row in table:
        print(row[animal_index])
        if row[animal_index].strip().lower() == 'dog':
            new_table.append(row)

    utils.write_csv('clean_data.csv', attr, new_table)        
    '''row_index = 0
    for row in table:
        if table[row_index][3] != "Dog":
            del table[row_index]
            row_index -= 1
        row_index += 1'''
    
    return attr, table

# Naive Bayes Kristen
def naive_bayes(attr, data): 
    '''
    '''
    class_index = attr.index('time_bucket')
# Decision Trees Alana
# k-Means Clustering Kristen
# Ensemble Learning: kNN or Decision Trees Alana ??? 

def main(): 
    '''
    '''
    header, table = preprocess()
    utils.write_csv("clean_data.csv", header, table)

    att_domains = {0: ["Stray", "Owner Surrender", "Wildlife", "Public Assist"], 
        1: ["Male", "Female"],
        2: ["Adoption", "Transfar", "Return to Owner", "Euthanasia", "Died", "Disposal", "Missing", "Rto-Adopt", "Relocate"],
        3: ["Spayed", "Neutered", "Intact"],
        4: ["1-3 years", "1-6 months", "4-6 years", "1-6 weeks", "7+ years", "7-12 months", "Less than 1 week"], #age
        5: [0, 1], # retriver
        6: [0, 1], # shepard
        7: [0, 1], # beagle
        8: [0, 1], # terrier
        9: [0, 1], # boxer
        10: [0, 1], # poodle
        11: [0, 1], # rottweiler
        12: [0, 1], # dachshund
        13: [0, 1], # chihuahua
        14: [0, 1]} # pitbull
    domain_header = ["intake_type", "gender_intake", "outcome_type", "fixed_outcome", "age_bucket", "retriver", "shepard",
                     "beagle", "terrier", "boxer", "poodle", "rottweiler", "dachshund", "chihuahua", "pibull"]

    att_indexes = list(range(14))
    class_index = len(header) - 1

    instance_to_classify = table[0]
    decision_tree_classifier(table, att_indexes, att_domains, class_index, domain_header, instance_to_classify)

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
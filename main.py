import utils 

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
    remove_attr = ['name_intake', 'found_location', 'intake_condition', 'month_year_intake', 'intake_sex', 'name_outcome', 
        'month_year_outcome', 'outcome_subtype', 'outcome_sex', 'gender_outcome', 'fixed_intake', 'fixed_changed']

    for col in remove_attr: 
        index = attr.index(col)
        attr.pop(index)
        for row in table: 
            row.pop(index)

    row_index = 0
    for row in table:
        if table[row_index][3] != "Dog":
            del table[row_index]
            row_index -= 1
        row_index += 1
    
    return attr, table

# kNN Kristen
# Decision Trees Alana
# k-Means Clustering Kristen
# Ensemble Learning: kNN or Decision Trees Alana ??? 

def main(): 
    '''
    '''
    header, table = preprocess()
    utils.write_csv("clean_data.csv", header, table)

    att_domains = {0: ["Stray", "Owner Surrender", "Wildlife", "Public Assist"], 
        1: ["R", "Python", "Java"],
        2: ["yes", "no"],
        3: ["yes", "no"]}

    att_indexes = list(range(len(header) - 1))

def decision_tree_classifier(table, att_indexes, att_domains, class_index, header, instance_to_classify):
    '''
    Calls the functions to get a decision tree for the data and uses that decision
    tree and classifies a given instance. Returns the classification to main()
    '''
    tree = tdidt(table, att_indexes, att_indexes, att_domains, class_index, header, [])
    classification = classify_instance(header, instance_to_classify, tree)
    return classification

if __name__ == "__main__":
    main()
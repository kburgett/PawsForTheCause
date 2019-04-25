import utils 
import decision_tree
import random
import copy

# Preprocessing Data 
def preprocess(attr, table):

    remove_attr = ['name_intake', 'found_location', 'intake_condition', 'month_year_intake', 'intake_sex', 'name_outcome', 
        'month_year_outcome', 'outcome_subtype', 'outcome_sex', 'gender_outcome', 'fixed_intake', 'fixed_changed',
        'breed_intake', 'color_intake', 'animal_id', 'date_time_intake', 'date_time_outcome', 'outcome_age', 'date_time_length', 'age', 'animal_type_intake']

    for col in remove_attr: 
        index = attr.index(col)
        attr.pop(index)
        for row in table: 
            row.pop(index)
    
    return attr, table
            
    utils.write_csv('clean_data.csv', attr, table)

def remove_other_animals(attr, table):
    # Remove all duplicate entries of animals and animals that are not dogs 
    animal_ids = set()
    animal_id_index = attr.index('animal_id')
    animal_index = attr.index('animal_type_intake')

    for row in table:
        # Remove animals that are not dogs 
        if row[animal_index].strip().lower() != 'dog':
            table.remove(row)
        else: 
            # Remove all duplicate entries of animals
            if row[animal_id_index] in animal_ids:
                table.remove(row)
            else: 
                animal_ids.add(row[animal_id_index])
    return table

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
    attr, original_table = utils.parse_csv("adoption_data.csv")
    original_table = remove_other_animals(attr, original_table)

    header, table = preprocess(attr, copy.deepcopy(original_table))
    utils.write_csv("clean_data.csv", header, table)

    att_domains = {0: ["Stray", "Owner Surrender", "Wildlife", "Public Assist"], 
        1: ["Adoption", "Transfar", "Return to Owner", "Euthanasia", "Died", "Disposal", "Missing", "Rto-Adopt", "Relocate"],
        2: ["Male", "Female"],
        3: ["Spayed", "Neutered", "Intact"],
        4: ["1-3 years", "1-6 months", "4-6 years", "1-6 weeks", "7+ years", "7-12 months", "Less than 1 week"], #age
        5: ['0', '1'], # retriver
        6: ['0', '1'], # shepard
        7: ['0', '1'], # beagle
        8: ['0', '1'], # terrier
        9: ['0', '1'], # boxer
        10: ['0', '1'], # poodle
        11: ['0', '1'], # rottweiler
        12: ['0', '1'], # dachshund
        13: ['0', '1'], # chihuahua
        14: ['0', '1']} # pitbull

    att_indexes = list(range(14))
    class_index = len(header) - 1

    instance_to_classify = table[0]
    decision_tree_classifier(table, original_table, att_indexes, att_domains, class_index, header, instance_to_classify)

def decision_tree_classifier(table, original_table, att_indexes, att_domains, class_index, header, instance_to_classify):
    '''
    Calls the functions to get a decision tree for the data and uses that decision
    tree and classifies a given instance. Returns the classification to main()
    '''
    rand_index = random.randint(0, len(table) - 1)
    instance = table[rand_index]
    tree = decision_tree.tdidt(table, att_indexes, att_indexes, att_domains, class_index, header, [])
    utils.pretty_print(tree)
    classification = decision_tree.classify_instance(header, instance, tree)
    print(original_table[rand_index])
    print("Classification: ", classification)


if __name__ == "__main__":
    main()
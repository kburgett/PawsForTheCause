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

    # Remove attributes not to be trained on from instances in the dataset 
    remove_attr = ['name_intake', 'found_location', 'intake_condition', 'month_year_intake', 'intake_sex', 'name_outcome', 
        'month_year_outcome', 'outcome_subtype', 'outcome_sex', 'gender_outcome', 'fixed_intake', 'fixed_changed']

    for col in remove_attr: 
        index = attr.index(col)
        attr.pop(index)
        for row in table: 
            row.pop(index)

    # Remove duplicate entries of an animal 
    
    
    return attr, table

# kNN 

# Decision Trees 
# k-Means Clustering 
# Ensemble Learning: kNN or Decision Trees ??? 

def main(): 
    '''
    '''
    attr, data = preprocess()
    print(attr)
    print(data)

if __name__ == "__main__":
    main()
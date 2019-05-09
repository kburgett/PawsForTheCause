# PawsForTheCause
## Alana Dillinger and Kristen Burgett
## CPSC310 Final Project
## PawsForTheCause

### Purpose

The purpose of our project is to predict the amount of time a rescued dog will spend at the Austin Animal Shelter in Austin, TX.
If they are able to predict how long the dog will spend there, the shelter would be able to make plans to make the dogs stay better
and work to getting them adopted by a loving family sooner. With our classifiers we were also able to look for patterns in the breeds
and ages of the dogs to see if some were more popular than others to adopt.

### Cleaning the Data
To make our project possible, we had to clean a lot of the data. We removed all instances that were not dogs (cats, birds, etc.), removed
some of the attributes that had thousands of values such as breed, color, and location found, and discretized the data we had left to
make it easier to use for classification.

### Classification Methods
1. Naive Bayes
    * We used the Naive Bayes algorithm to classify how long the dogs would stay there and created a confusion matrix to see the
    distribution of the classifications.
2. Decision Trees
    * We used the TDIDT algorithm to create a decision tree with attribute selection based on entropy. The deicision tree was used to 
    develop a list of rules that could be used to classify each dog and to look for patterns in the data.
3. Random Forest Ensemble Method
    * We created a random forest using a random sub-sampling methods to create the training, validation, and test sets. The forest contains
    only the 7 best of 10 trees because the algorithm computational cost was high. The attributes for each decision tree were also selected
    using entropy and we were able to develop a similar list of rules to classify new rescued dogs with and this was the most accurate of 
    the classifiers.
4. K-Means Clustering
    * We used K-Means Clustering in a different way than the rest of the classifiers. It is an unsupervised learning technique and we wanted
    to be able to see if we were missing any patterns that may help us to improve our other classifiers. Since our data set is so large,
    we have split the data into sets of 1000 instances for the clustering algorithm to save on compuational costs. The result of the
    algorithm is a set of 7-17 clusters that can be used to see patterns in the data that may not be related to the amount of time the 
    dogs spend in the shelter. 
  

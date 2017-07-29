# I_vote_for_cancer


I spilt the (training_variants and training_text) into two fixed parts(the indices are fixed):

1) Use 80% data from (training_variants and training_text) for 1st layer training, to get the data use:
text, variants = helpers.get_training(train_variation_file, train_text_file)

Train your 1st layer models(tdidf, w2v, d2v, lda + whatever classification algorithms) using this data set. 
These models should be unaware of the remaining 20% data.


2) Use the same 80% data from (training_variants and training_text) for 2nd layer training, to get the data use:
text, variants = helpers.get_training(train_variation_file, train_text_file)

Report the probabilities for the 9 classes using our first layer models, and then we can combine 
these probabilities as new features to train our 2nd layer model. 

In order to combine our results easily, please report these probabilities using the same format as submissionFile for all 3321 samples. Put these results in a folder named "2nd_layer_data", and in the filename, indicate the logloss of the 20% test set using your 1st layer model, such as "tfidf_svm_1.2"

This 2nd layer model should be unaware of the remaining 20% data.

3) Use 20% data from (training_variants and training_text) for reporting final score, to get the data use:
text, variants = helpers.get_test(train_variation_file, train_text_file)








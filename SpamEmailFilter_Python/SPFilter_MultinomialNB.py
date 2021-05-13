# Justin Gallagher
# Craig Mcghee
# CSC 425
# 11/08/20

import os
import math
import numpy as np

most_common_word = 3000
# avoid 0 terms in features
smooth_alpha = 1.0
class_num = 2  # we have only two classes: ham and spam
class_log_prior = [0.0, 0.0]  # probability for two classes
feature_log_prob = np.zeros((class_num, most_common_word))  # feature parameterized probability
SPAM = 1  # spam class label
HAM = 0  # ham class label

class MultinomialNB_class:

    def MultinomialNB(self, features, labels):

        total_ham = 0                           # Sum of occurrences of every word in the ham class
        total_spam = 0                          # Sum of occurrences of every word in spam class
        sp_ham = np.zeros(most_common_word)     # Sum of occurrences for a given word in ham
        sp_spam = np.zeros(most_common_word)    # Sum of occurrences for a given word in spam

        # Calculate class_log_prior using total number of emails / number of classes / total number of emails
        class_log_prior[HAM] = len(features)//class_num / len(features)
        class_log_prior[SPAM] = len(features)//class_num / len(features)

        # Calculate total_ham and sp_ham
        for i in range(len(features)//class_num):
            for j in range(most_common_word):
                total_ham += features[i][j]
                sp_ham[j] += features[i][j]

        # Calculate total_spam and sp_spam
        for i in range(len(features)//class_num, len(features)):
            for j in range(most_common_word):
                total_spam += features[i][j]
                sp_spam[j] += features[i][j]

        # Take the logarithm of the probability that a given word is in a given class. This is calculated by taking the
        # number of occurrences of the given word plus a smoothing constant divided by the total number of words the given
        # class plus the total number of unique words in the given class times the smoothing constant.
        for j in range(most_common_word):
            feature_log_prob[HAM][j] = math.log((sp_ham[j] + smooth_alpha) / (total_ham + most_common_word * smooth_alpha))
            feature_log_prob[SPAM][j] = math.log((sp_spam[j] + smooth_alpha) / (total_spam + most_common_word * smooth_alpha))

    def MultinomialNB_predict(self, features):
        classes = np.zeros(len(features))

        # For a given email, sum up the product of a probability that a word is in a given class and the number of
        # occurrences of that word in the email and add the logarithm of the probability that an email is in a given class
        # (from the training set) and pick the maximum to decide what class the given email is.
        for i in range(len(features)):
            ham_prob = 0.0
            spam_prob = 0.0
            for j in range(most_common_word):
                ham_prob += feature_log_prob[HAM][j] * features[i][j] + math.log(class_log_prior[HAM])
                spam_prob += feature_log_prob[SPAM][j] * features[i][j] + math.log(class_log_prior[SPAM])

            if spam_prob < ham_prob:              # If ham_prob is greater than spam_prob:
                classes[i] = HAM                  # Label email as ham.
            else:                                 # Otherwise:
                classes[i] = SPAM                 # Label email as spam.

        return classes                            # Return list of labels

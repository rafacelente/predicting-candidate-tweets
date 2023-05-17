import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

def plot_top_words(classifier, vectorizer, title='untitled.png'):
    """
        Plot top features in a binary text classifier

        Based on https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
    """

    sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)

    try:
        if type(classifier) == MultinomialNB:
            prob_logic = classifier.feature_log_prob_
            results_type = 0
        elif type(classifier) == LinearSVC:
            prob_logic = classifier.coef_[0]
            results_type = 1
        elif (type(classifier) == RandomForestClassifier):
            prob_logic = classifier.feature_importances_
            results_type = 2
    except Exception as err:
        print(f"Model type not found {type(classifier)}: {err}")
        raise

        
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names_out()

    width = 0.5
    fig, ax = plt.subplots()
    bottom = np.arange(len(feature_names)*1.5, step=1.5)

    if results_type == 0:
        top_class = sorted(zip(prob_logic[0], prob_logic[1], feature_names))
        all_probs = {
            class_labels[0]: [np.exp(c[0]) for c in top_class],
            class_labels[1]: [np.exp(c[1]) for c in top_class]
            }
        feature_names = [c[2] for c in top_class]

        p = ax.bar(bottom, list(all_probs.items())[0][1], width=width, align='edge', label='Bolsonaro')
        p2 = ax.bar(bottom+width, list(all_probs.items())[1][1], width=width, align='edge', label='Lula')
        ax.legend(loc='upper left')
        plt.title('Word probabilities for the Naive-Bayes model (sorted by influence)')

    elif results_type == 1:
        top_class = sorted(zip(prob_logic, feature_names))
        plt.bar(bottom, [c[0] for c in top_class])
        feature_names = [c[1] for c in top_class]
        plt.title('Hyperplane coefficients for the LinearSVC model features')
    
    elif results_type == 2:
        std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
        top_class = sorted(zip(prob_logic, feature_names, std))
        values_sorted = [c[0] for c in top_class]
        feature_names = [c[1] for c in top_class]
        std_sorted = [c[2] for c in top_class]
        plt.bar(bottom, values_sorted, yerr=std_sorted, ecolor='black')
        plt.title('Feature importance in the Random Forest Classifier model')


    plt.xticks(bottom+width, list(feature_names), rotation=60, ha='right')
    plt.tick_params(axis='x', which='major', labelsize=8)
    plt.tight_layout()
    plt.savefig(title, format='png')
    plt.show()
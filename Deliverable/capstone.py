import numpy as np
from matplotlib import pyplot as plt
import os, time


def load_data(csv_filepath):
    import pandas as pd
    df = pd.read_csv(csv_filepath)  # Load provided source data CSV file of profiles
    return df


def inspect_data(df):
    print(df.head(5))
    print(df.describe())
    print(df.columns)

    print(df['age'].value_counts() / len(df))
    print(df['diet'].value_counts() / len(df))
    print(df['education'].value_counts() / len(df))
    print(df['income'].value_counts() / len(df))
    # print(df['job'].value_counts()/len(df))
    # print(df['location'].value_counts()/len(df))
    print(df['income'].value_counts() / len(df))
    print(df['sex'].value_counts() / len(df))
    # print(df['orientation'].value_counts()/len(df))
    # print(df['ethnicity'].value_counts()/len(df))
    print(df.isna().any())
    return


def map_data(df):  # Augment Data with new columns, mapping multiple choice strings to integers
    diet_mapping = {"mostly anything": 0, "anything": 0, "strictly anything": 0, "mostly vegetarian": 1, "mostly other": 3, "strictly vegetarian": 1, "vegetarian": 1, "strictly other": 3, "mostly vegan": 1, "other": 3, "strictly vegan": 1, "vegan": 1, "mostly kosher": 2, "mostly halal": 2, "strictly kosher": 2, "strictly halal": 2, "halal": 2, "kosher": 2}
    education_mapping = {"dropped out of space camp ": 0, "working on space camp": 0, "space camp": 0, "graduated from space camp": 0, "dropped out of high school": 1, "working on high school": 2, "high school": 2, "graduated from high school": 3, "dropped out of two-year college": 3, "dropped out of college/university": 3, "working on two-year college": 4, "two-year college": 4, "working on college/university": 5, "college/university": 5, "graduated from two-year college": 6, "graduated from college/university": 7,"dropped out of masters program": 7, "dropped out of law school": 7, "dropped out of med school": 7, "working on masters program": 8, "working on med school": 8, "med school": 8, "working on law school": 8, "law school": 8, "masters program": 8, "graduated from masters program": 9, "dropped out of ph.d program": 9, "working on ph.d program": 10, "ph.d program": 10, "graduated from law school": 11, "graduated from ph.d program": 11, "graduated from med school": 11}
    job_mapping = {"other": 0, "student": 1, "science / tech / engineering": 2, "computer / hardware / software": 3, "artistic / musical / writer": 4, "sales / marketing / biz dev": 5, "medicine / health": 6, "education / academia": 7, "executive / management": 8, "banking / financial / real estate": 9, "entertainment / media": 10, "law / legal services": 11, "hospitality / travel": 12, "construction / craftsmanship": 13, "clerical / administrative": 13, "political / government": 14, "rather not say": 15, "transportation": 16, "unemployed": 17, "retired": 18, "military": 19}
    sex_mapping = {'m': 0, 'f': 1}
    orientation_mapping = {'straight': 0, 'bisexual': 1, 'gay': 2}
    df['diet_code'] = df['diet'].map(diet_mapping)
    df['education_code'] = df['education'].map(education_mapping)
    df['job_code'] = df['job'].map(job_mapping)
    df['sex_code'] = df['sex'].map(sex_mapping)
    df['orientation_code'] = df['orientation'].map(orientation_mapping)
    return df


def clean_data(df):  # Basic Clean, Additional Cleaning Required for Given Test Cases
    features_to_remove = ['body_type','drinks','drugs','essay0','essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9','height','last_online','offspring','pets','religion','sign','smokes','speaks','status']
    df.drop(features_to_remove, axis=1, inplace=True)  # Remove unnecessary features that won't be used any analysis
    return df


def split_data(features, labels):
    from sklearn.model_selection import train_test_split
    # Using standard split of 80-20 training to testing data split ratio and fixing random_state=1 for repeatability
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, test_size=0.2, random_state=1)
    return x_train, x_test, y_train, y_test


def score_classifier(truth, predictions):
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
    m_accuracy = accuracy_score(truth, predictions)
    m_recall = recall_score(truth, predictions)
    m_precision = precision_score(truth, predictions)
    m_f1 = f1_score(truth, predictions)
    #print(accuracy_score(truth, predictions))
    #print(recall_score(truth, predictions))
    #print(precision_score(truth, predictions))
    #print(f1_score(truth, predictions))
    return m_accuracy, m_recall, m_precision, m_f1


def regression_linear(features, labels):
    from sklearn.linear_model import LinearRegression
    x_train, x_test, y_train, y_test = split_data(features, labels)

    time_in = time.time()
    lm = LinearRegression()
    model = lm.fit(x_train, y_train)
    score = lm.score(x_test, y_test)
    print('Time elapsed: ' + str(time.time() - time_in) + '')

    y_predicted = model.predict(x_test)
    plt.scatter(y_test, y_predicted, alpha=0.2, color="#3E849E")
    plt.title('Multiple Linear Regression Performance')
    plt.ylabel('Predicted User Income')  # Predicted Value
    plt.xlabel('Actual User Income')  # Actual Value
    plt.show()

    return score


def regression_k_nearest(features, labels, k, plot_flag):
    from sklearn.neighbors import KNeighborsRegressor
    x_train, x_test, y_train, y_test = split_data(features, labels)

    time_in = time.time()
    regressor = KNeighborsRegressor(n_neighbors=k, weights='distance')
    regressor.fit(x_train, y_train)
    score = regressor.score(x_test, y_test)
    print('Time elapsed: ' + str(time.time() - time_in) + '')

    if plot_flag:
        y_predicted = regressor.predict(x_test)
        plt.scatter(y_test, y_predicted, alpha=0.2, color="#3E849E")
        plt.title('K Nearest Neighbors Regressor Performance')
        plt.ylabel('Predicted User Income')  # Predicted Value
        plt.xlabel('Actual User Income')  # Actual Value
        plt.show()

    return score


def classification_k_nearest(features, labels, k):
    from sklearn.neighbors import KNeighborsClassifier
    x_train, x_test, y_train, y_test = split_data(features, labels)

    time_in = time.time()
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(x_train, y_train)
    y_predicted = classifier.predict(x_test)
    m_accuracy, m_recall, m_precision, m_f1 = score_classifier(y_test, y_predicted)
    print('Time elapsed: '+str(time.time() - time_in)+'')

    return m_accuracy, m_recall, m_precision, m_f1


def classification_support_vector_machine(features, labels, kernel_value, gamma_value):
    from sklearn.svm import SVC
    x_train, x_test, y_train, y_test = split_data(features, labels)
    time_in = time.time()
    classifier = SVC(kernel=kernel_value, gamma=gamma_value)
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    print('Time elapsed: ' + str(time.time() - time_in) + '')

    return score


def initialize_data_set():
    csv_filepath = os.path.join('Source_Data', 'profiles.csv')
    df = load_data(csv_filepath)
    # inspect_data(df)
    df = map_data(df)
    df = clean_data(df)
    return df

def plot_income_vs_education():
    df = initialize_data_set()  # Reload and initialize data set, ensures no manipulations present from other analyses
    features_to_remove = ['diet', 'education', 'ethnicity', 'job', 'location', 'orientation', 'sex', 'diet_code',
                          'job_code', 'sex_code', 'orientation_code']
    df.drop(features_to_remove, axis=1, inplace=True)  # Remove features that won't be used in this analysis
    df = df[df.income != -1]  # Drop the 80.8% of entries in supplied data that report '-1' for income, essentially Nan
    df.dropna(inplace=True)
    df['income_z'] = (df['income'] - df['income'].mean()) / df['income'].std(
        ddof=0)  # Create column of Z-Score normalized income
    plt.scatter(df['education_code'], df['income_z'], alpha=0.1, color="#51ACC5")
    plt.ylabel('Income (Z-Score Normalized)')
    plt.xlabel('Education Level')
    plt.title('Income vs Education')
    plt.show()


def pie_plot_sex_distribution():
    df = initialize_data_set()  # Reload and initialize data set, ensures no manipulations present from other analyses

    features_to_remove = ['diet', 'education', 'ethnicity', 'job', 'location', 'orientation', 'diet_code',
                          'job_code', 'orientation_code', 'income', 'education_code', 'age']
    df.drop(features_to_remove, axis=1, inplace=True)  # Remove features that won't be used in this analysis
    df.dropna(inplace=True)

    blues = ["#66D7EB", "#51ACC5", "#3E849E", "#2C5F78", "#1C3D52", "#0E1E2B"]
    fig1, ax1 = plt.subplots()
    labels = ['Men', 'Women']
    ax1.pie(df.groupby('sex_code').sex.count(), labels=labels, autopct='%1.1f%%', shadow=False, startangle=90, colors=blues)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Sex Distribution of Users')
    plt.show()
    return


def hist_plot_age_distribution():
    df = initialize_data_set()  # Reload and initialize data set, ensures no manipulations present from other analyses

    features_to_remove = ['diet', 'education', 'ethnicity', 'job', 'location', 'orientation', 'sex', 'diet_code',
                          'job_code', 'orientation_code', 'income', 'education_code', 'sex_code']
    df.drop(features_to_remove, axis=1, inplace=True)  # Remove features that won't be used in this analysis
    df.dropna(inplace=True)
    print(df.age.describe())

    blues = ["#66D7EB", "#51ACC5", "#3E849E", "#2C5F78", "#1C3D52", "#0E1E2B"]
    plt.hist(df.age, bins=50, facecolor=blues[1], alpha=1.0)
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.title("Age Distribution of Users")
    plt.xlim(16, 80)
    plt.show()
    return


def pie_plot_income_distribution(df):
    # Print and plot some information about the income data in the remaining user data
    print(df.income.describe())  # Five Number Summary of Income Data
    blues = ["#66D7EB", "#51ACC5", "#3E849E", "#2C5F78", "#1C3D52",
             "#0E1E2B"]  # FYI: last color too dark for black labels
    pie_values = df.groupby('income').income.count()
    pie_labels = df.groupby('income').income.count().index
    porcent = 100. * pie_values / pie_values.sum()
    patches, texts = plt.pie(pie_values, labels=pie_labels, colors=blues, startangle=90)
    labels = ['{0} - {1:1.1f}%'.format(i, j) for i, j in zip(pie_labels, porcent)]
    plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.), fontsize=12)
    plt.title('Income Distribution of Users')
    plt.show()
    return


def run_linear_regression():
    # What is this Analysis doing?
    print('Linear Regression Model: ')
    print("Can we predict 'income' based on linear data 'age' and 'education_code'?")

    # Load & Cleanse Data: tailoring the dataframe for the intended analysis
    df = initialize_data_set()  # Reload and initialize data set, ensures no manipulations present from other analyses
    # First drop records for users not in California (e.g. coarsely accounting for location as an influencer in income) 59946 vs 59855
    df = df[df['location'].str.contains("california")]
    features_to_remove = ['diet', 'education', 'ethnicity', 'job', 'location', 'orientation', 'sex', 'diet_code',
                          'job_code', 'sex_code', 'orientation_code']
    df.drop(features_to_remove, axis=1, inplace=True)  # Remove features that won't be used in this analysis
    df = df[df.income != -1]  # Drop the 80.8% of entries in supplied data that report '-1' for income, essentially Nan
    df = df[df.income < 120000]  # Remove High Income users... outliers
    df.dropna(inplace=True)
    #print(df.isna().any())  # Result should return "False" for all columns

    # Normalize Data
    from sklearn.preprocessing import scale
    from sklearn.preprocessing import normalize

    features = df[['education_code', 'age']]
    scaled_features = scale(features, axis=0)
    normalized_features = normalize(features, axis=0)

    df['income_z'] = (df['income'] - df['income'].mean()) / df['income'].std(
        ddof=0)  # Create column of Z-Score normalized income
    labels = df['income']
    #labels = df['income_z']

    # Linear Regression Model
    score = regression_linear(scaled_features, labels)
    print('Linear Regression Score: ' + str(score))  # 0.220903829199 wo scale
    return


def run_k_neighbors_regression():
    # What is this Analysis doing?
    print('K Nearest Neighbors Regression Model: ')
    print("Can we predict 'income' based on data 'age', 'education_code', and 'job_code'?")

    # Load & Cleanse Data: tailoring the dataframe for the intended analysis
    df = initialize_data_set()  # Reload and initialize data set, ensures no manipulations present from other analyses
    # First drop records for users not in California (e.g. coarsely accounting for location as an influencer in income) 59946 vs 59855
    df = df[df['location'].str.contains("california")]
    features_to_remove = ['diet', 'education', 'ethnicity', 'job', 'location', 'orientation', 'sex', 'diet_code', 'sex_code', 'orientation_code']
    df.drop(features_to_remove, axis=1, inplace=True)  # Remove features that won't be used in this analysis
    df = df[df.income != -1]  # Drop the 80.8% of entries in supplied data that report '-1' for income, essentially Nan
    df = df[df.income < 120000]  # Remove High Income users... outliers
    df.dropna(inplace=True)

    #plt.scatter(df.income, df.age, alpha=0.1)
    #plt.show()

    # Normalize
    from sklearn.preprocessing import scale
    from sklearn.preprocessing import normalize

    features = df[['education_code', 'age', 'job_code']]
    features = df[['education_code', 'job_code']]
    labels = df['income']
    scaled_features = scale(features, axis=0)
    normalized_features = normalize(features, axis=0)

    scores = []
    k_values = list(range(1, 100))
    for k in k_values:
        score = regression_k_nearest(normalized_features, labels, k, False)
        scores.append(score)
    print('Best K Nearest Neighbors score: '+str(max(scores)))
    plt.plot(k_values, scores, color='#51ACC5')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy as k changes')
    plt.show()

    score = regression_k_nearest(normalized_features, labels, 20, True)

    return


def run_k_neighbors_classification():
    # What is this Analysis doing?
    print('K Nearest Neighbors Classifier Model: ')
    print("Can we predict 'sex' based on data 'diet_code' and 'job_code'?")

    # Load & Cleanse Data: tailoring the dataframe for the intended analysis
    df = initialize_data_set()  # Reload and initialize data set, ensures no manipulations present from other analyses
    features_to_remove = ['diet', 'education', 'ethnicity', 'job', 'location', 'orientation', 'sex', 'orientation_code']
    df.drop(features_to_remove, axis=1, inplace=True)  # Remove features that won't be used in this analysis
    df.dropna(inplace=True)
    df = df[df.diet_code <= 1]  # Remove all but "Anything" & "Vegetarian/Vegan", e.g. remove diets related to religion

    # Normalize
    from sklearn.preprocessing import scale
    from sklearn.preprocessing import normalize

    features = df[['diet_code', 'job_code']]
    labels = df['sex_code']
    scaled_features = scale(features, axis=0)
    normalized_features = normalize(features, axis=0)

    scores_a = []
    scores_r = []
    scores_p = []
    scores_f = []
    k_values = list(range(1, 51, 2))
    for k in k_values:
        m_accuracy, m_recall, m_precision, m_f1 = classification_k_nearest(normalized_features, labels, k)
        scores_a.append(m_accuracy)
        scores_r.append(m_recall)
        scores_p.append(m_precision)
        scores_f.append(m_f1)
    plot_labels = ['Accuracy', 'Recall', 'Precision', 'F1 Score']
    blues = ["#66D7EB", "#51ACC5", "#3E849E", "#2C5F78", "#1C3D52", "#0E1E2B"]

    plt.plot(k_values, scores_a, '-', label=plot_labels[0], color=blues[0])
    plt.plot(k_values, scores_r, '--', label=plot_labels[1], color=blues[1])
    plt.plot(k_values, scores_p, '-.', label=plot_labels[2], color=blues[2])
    plt.plot(k_values, scores_f, ':', label=plot_labels[3], color=blues[3])
    # From plot, k=19 appears to  maximize accuracy w/o secondary metrics of precision & recall
    k = 19
    plt.plot([k]*3, [0.3, 0.45, 0.65], '--', color=blues[5])
    plt.legend(loc='lower right')
    plt.title('K Nearest Neighbors Classification - Performance vs k Value')
    plt.xlabel('k Value')
    plt.ylabel('Classifier Scores')
    plt.show()

    m_accuracy, m_recall, m_precision, m_f1 = classification_k_nearest(normalized_features, labels, k)
    print('K Nearest Neighbors accuracy score: ' + str(m_accuracy))

    return


def run_svm_classifier():
    # What is this Analysis doing?
    print('Support Vector Machine Classifier: ')
    print("Can we predict 'sex' based on data 'diet_code' and 'job_code'?")

    # Load & Cleanse Data: tailoring the dataframe for the intended analysis
    df = initialize_data_set()  # Reload and initialize data set, ensures no manipulations present from other analyses
    features_to_remove = ['diet', 'education', 'ethnicity', 'job', 'location', 'orientation', 'sex', 'orientation_code']
    df.drop(features_to_remove, axis=1, inplace=True)  # Remove features that won't be used in this analysis
    df.dropna(inplace=True)
    df = df[df.diet_code <= 1]  # Remove all but "Anything" & "Vegetarian/Vegan", e.g. remove diets related to religion

    # Normalize
    from sklearn.preprocessing import scale
    from sklearn.preprocessing import normalize

    features = df[['diet_code', 'job_code']]
    labels = df['sex_code']
    scaled_features = scale(features, axis=0)
    normalized_features = normalize(features, axis=0)

    kernel = 'rbf'
    gamma_values = np.linspace(0.05, 1.0, num=20)  # Exploring Gamma's effect on score
    gamma_values = np.linspace(1.0, 10.0, num=20)
    gamma_values = np.linspace(10.0, 100.0, num=10)
    gamma_values = np.linspace(0.05, 100.0, num=10)
    gamma_values = np.concatenate((np.linspace(0.05, 1.0, num=20), np.linspace(1.0, 10.0, num=20), np.linspace(10.0, 100.0, num=10)), axis=0)  # exhaustive
    gamma_values = np.linspace(6.2, 6.6, num=10)  # find inflection point

    scores = []
    for gamma in gamma_values:
        score = classification_support_vector_machine(normalized_features, labels, kernel, gamma)
        scores.append(score)
        print('gamma: ' + str(gamma) + '  score: ' + str(score))
    blues = ["#66D7EB", "#51ACC5", "#3E849E", "#2C5F78", "#1C3D52", "#0E1E2B"]
    plt.plot(gamma_values, scores, '-', label='gamma', color=blues[0])
    plt.title('Support Vector Machine Classification - RBF Kernel Performance vs Gamma')
    plt.xlabel('Gamma')
    plt.ylabel('Classifier Score')
    plt.show()

    gamma = 10.0
    score = classification_support_vector_machine(normalized_features, labels, kernel, gamma)
    print('Support Vector Machine accuracy score: ' + str(score))
    return



# Basic Inspections of the Data Frame
df = initialize_data_set()
inspect_data(df)

# Plot Data Investigations
hist_plot_age_distribution()
pie_plot_sex_distribution()
plot_income_vs_education()

# Plot Income Information - Including Users that did not Report Income
df = initialize_data_set()
pie_plot_income_distribution(df)

# Plot Income Information - Removing Users that did not Report Income
df = initialize_data_set()
df = df[df.income != -1]  # Drop the 80.8% of entries in supplied data that report '-1' for income, essentially Nan
pie_plot_income_distribution(df)

# Run All Machine Learning Models
run_linear_regression()
run_k_neighbors_regression()
run_k_neighbors_classification()
run_svm_classifier()


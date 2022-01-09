from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import Data_Preprocessing
import numpy as np
import pandas as pd


def pca_transform(dimensions, training_data, test_data):
    # Construct PCA model
    model = PCA(n_components=dimensions)

    # Compute mean of every picture
    training_data = np.array(training_data)
    training_data_mean = training_data.mean(axis=0)

    # Normalize to make E[x] = 0
    training_data_zero_mean = training_data - training_data_mean

    # Model Training
    model.fit(training_data_zero_mean)

    # Transform test data
    test_transform = model.transform(test_data - training_data_mean)
    return model, test_transform, training_data_zero_mean


def lda_transform(dimensions, training_data, training_target, test_data):
    # Construct LDA model
    model = LinearDiscriminantAnalysis(n_components=dimensions)

    # Compute mean of every picture
    training_data = np.array(training_data)
    training_data_mean = training_data.mean(axis=0)

    # Normalize to make E[x] = 0
    training_data_zero_mean = training_data - training_data_mean

    # Model Training
    model.fit(training_data_zero_mean, training_target)

    # Transform test data
    test_transform = model.transform(test_data - training_data_mean)
    return model, test_transform, training_data_zero_mean


if __name__ == "__main__":
    # From Data_Preprocessing load in train data & test data
    training_datasets, training_target = Data_Preprocessing.load_training_datasets()
    test_datasets, test_target = Data_Preprocessing.load_test_datasets()

    # Ravel target data for avoiding errors
    training_target = np.array(training_target)
    training_target = training_target.ravel()
    test_target = np.array(test_target)
    test_target = test_target.ravel()

    # Dimension reduction
    pca_dimension = [10, 20, 30, 40, 50]
    lda_dimension = [10, 20, 30]
    PCA_true_counter_list = []
    PCA_false_counter_list = []
    LDA_true_counter_list = []
    LDA_false_counter_list = []
    for i in pca_dimension:
        if i == 40:
            print("\n\nNow, dimension is larger than class label=40, so LDA is stopped working. \n\n")

        # PCA
        pca, pca_test, pca_training_zero_mean = pca_transform(i, training_datasets, test_datasets)

        # LDA
        if i < 40:
            lda, pca2lda, lda_training_zero_mean = lda_transform(
                i, pca.transform(pca_training_zero_mean), training_target, pca_test)

        # Construct SVM Model
        SVM = SVC(kernel="linear")

        # SVM for PCA
        SVM.fit(pca.transform(pca_training_zero_mean), training_target)
        SVM_PCA_predict_result = SVM.predict(pca_test)

        # SVM for LDA
        if i < 40:
            SVM.fit(lda.transform(lda_training_zero_mean), training_target)
            SVM_LDA_predict_result = SVM.predict(pca2lda)

        # Count PCA Confusion Matrix
        SVM_PCA_confusion_matrix = confusion_matrix(test_target, SVM_PCA_predict_result)

        # Count LDA Confusion Matrix
        if i < 40:
            SVM_LDA_confusion_matrix = confusion_matrix(test_target, SVM_LDA_predict_result)

        # Print accuracy of PCA
        print("Accuracy of dimension {:d} in SVM for PCA : {:.2f}%".format(
            i, accuracy_score(SVM_PCA_predict_result, test_target) * 100))
        print("---------------------------------------")

        # Print accuracy of LDA
        if i < 40:
            print("Accuracy of dimension {:d} in SVM for LDA : {:.2f}%".format(
                i, accuracy_score(SVM_LDA_predict_result, test_target) * 100))
            print("---------------------------------------")

        # Print Confusion Matrix of PCA
        print("Confusion Matrix of dimension {:d} in SVM for PCA : \n".format(i), SVM_PCA_confusion_matrix)
        print("---------------------------------------")

        # Print Confusion Matrix of LDA
        if i < 40:
            print("Confusion Matrix of dimension {:d} in SVM for PCA : \n".format(i), SVM_LDA_confusion_matrix)
            print("=======================================\n")
        else:
            print("=======================================\n")

        # Count PCA true & false data
        true_counter = 0
        false_counter = 0
        for m in range(0, len(test_target)):
            if SVM_PCA_predict_result[m] == test_target[m]:
                true_counter += 1
            else:
                false_counter += 1
        PCA_true_counter_list.append(true_counter)
        PCA_false_counter_list.append(false_counter)

        # Count LDA true & false data
        if i < 40:
            true_counter = 0
            false_counter = 0
            for m in range(0, len(test_target)):
                if SVM_LDA_predict_result[m] == test_target[m]:
                    true_counter += 1
                else:
                    false_counter += 1
            LDA_true_counter_list.append(true_counter)
            LDA_false_counter_list.append(false_counter)

    # Jump out the loop
    # Use dataframe to create PCA Confusion Data
    PCA_statistics_dictionary = {"dimension": pca_dimension,
                                 "True item": PCA_true_counter_list,
                                 "False item": PCA_false_counter_list
                                 }
    PCA_statistics_dataframe = pd.DataFrame(PCA_statistics_dictionary)
    print(PCA_statistics_dataframe)

    print("\n=======================================\n")

    # Use dataframe to create PCA Confusion Data
    LDA_statistics_dictionary = {"dimension": lda_dimension,
                                 "True item": LDA_true_counter_list,
                                 "False item": LDA_false_counter_list
                                 }
    LDA_statistics_dataframe = pd.DataFrame(LDA_statistics_dictionary)
    print(LDA_statistics_dataframe)

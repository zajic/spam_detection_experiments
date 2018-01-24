"""
Print model selection metrics:
confusion matrix
accuracy
true positive, true negative, false positive, false negative rates
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def calculate_metrics(Y_actual, Y_pred):
    print("confusion matrix")
    cm = confusion_matrix(Y_actual, Y_pred,labels=["spam", "ham"])
    print(cm)
    print("accuracy: " + str(round(accuracy_score(Y_actual, Y_pred), 2)))

    TN, FP, FN, TP = confusion_matrix(Y_actual, Y_pred).ravel()

    FPR = round(FP / (FP + TN), 2)
    FNR = round(FN / (FN + TP), 2)
    TPR = round(TP / (TP + FN), 2)
    TNR = round(TN / (TN + FP), 2)

    print("TPR: " + str(TPR))
    print("TNR: " + str(TNR))
    print("FPR: " + str(FPR))
    print("FNR: " + str(FNR))
# Create four groups using train_test_split. By default, 75% of data is assigned to train, the other 25% to test.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=2)

print('X shape\t\t:', X.shape)
print('y shape\t\t:', Y.shape)
print()
print('X_train shape\t:', X_train.shape)
print('y_train shape\t:', Y_train.shape)
print()
print('X_test shape\t:', X_test.shape)
print('y_test shape\t:', Y_test.shape)

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred_log = logreg.predict(X_train)
len(Y_pred_log)

## Train Confusion Metrics
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_train, Y_pred_log)
print(confusion_matrix)


## Train Model Accuracy
print(classification_report(Y_train, Y_pred_log))


#ax = plt.subplots(1, 3, figsize=(18,6))
#
#sns.heatmap(confusion_matrix, annot=True,  cmap="YlGnBu" ,fmt='.2f', xticklabels = ["No", "Yes"] , yticklabels = ["No", "Yes"],)
#
#ax.xaxis.set_label_position("top")
#plt.tight_layout()
#plt.title('Confusion matrix', y=1.1)
#plt.ylabel('True label',fontsize=12)
#plt.xlabel('Predicted label',fontsize=12)


# =============================================================================
# ## Test Prediction
# =============================================================================
Y_pred_log_test = logreg.predict(X_test)
len(Y_pred_log_test)

## Test Confusion Metrics
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_pred_log_test)
print(confusion_matrix)


## Test Model classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_log_test))




### ROC Curve for Logistic Regression
logit_roc_auc = roc_auc_score(Y_train, logreg.predict(X_train))
#logit_roc_auc = roc_auc_score(y_test, y_pred_log )
fpr_log, tpr_log, thresholds_log = roc_curve(Y_train, logreg.predict_proba(X_train)[:,1])

plt.figure()
#plt.plot(fprB_1, tprB_1, label='XGBM Baseline (area = %0.2f)' % model_roc_auc)
#plt.plot(fpr1_1, tpr1_1, label='GBM Model 1 (area = %0.2f)' % GBbaseline_roc_auc)
plt.plot(fpr_log, tpr_log, label='Logit Model (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

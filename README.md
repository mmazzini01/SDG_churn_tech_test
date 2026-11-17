# SDG_churn_tech_test

## Reading the metrics

The sample is balanced (~50% churners: 49.6% in the training set), which is far above a real monthly churn rate.
Precision, PR-AUC and the Brier score depend on that prevalence, so the values reported here do not represent the
real population: in production, at the same cut, precision would be lower (more false alarms per churner found)
and the probabilities would need recalibrating to the real churn rate. Recall and ROC-AUC do not depend on the
prevalence. Models are compared with each other on the same sample, never against figures from other datasets.

#dataset
X = df.drop(columns=['EmployeeTargeted']).values
y = df.filter(['EmployeeTargeted']).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Model 
rf= RandomForestClassifier(n_estimators=500, class_weight="balanced")
rf.fit(X_train,y_train)
rf_pre=rf.predict(X_test)
print(confusion_matrix(y_test,rf_pre))
print(classification_report(y_test,rf_pre))
scores = cross_val_score(rf, X, y, cv=5)
print(scores)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
run.log('Accuracy', np.float(acc))

# Save the trained model / Export model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

run.complete()
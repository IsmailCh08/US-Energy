
# Define data splitting function
def split_time_series(df,target_col='PJME_MW', test_ratio=0.2):

    split_indx = int(len(df) * (1- test_ratio))

    train = df.iloc[:split_indx]
    test = df.iloc[split_indx:]

    # prevent data leakage

    X_train = train.drop(columns=[target_col])
    X_test = test.drop(columns=[target_col])


    y_train = train[target_col]
    y_test = test[target_col]

    return X_train, y_train, X_test, y_test


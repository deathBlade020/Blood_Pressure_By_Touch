import pandas as pd
import numpy as np
from regressor import extract_features,denormalize,plot_single,plot_scatter
from icecream import ic
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import random
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt


def train_knn(features,sugar_data,max_sugar,min_sugar):
    features_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    features_scaled = features_scaler.fit_transform(features)
    blood_pressure_scaled = target_scaler.fit_transform(sugar_data.values.reshape(-1, 1)).flatten()
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, blood_pressure_scaled, test_size=0.1, shuffle=True)

    param_grid = {
        'n_neighbors': range(1, int(len(X_train)**0.5)+1),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='r2',verbose=0)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    y_pred = best_model.predict(X_test)
    print(f"Train Size: {len(X_train)}, Test Size: {len(X_test)}")
    for actual, pred in zip(y_test, y_pred):
        a = denormalize(actual, max_sugar, min_sugar)
        b = denormalize(pred, max_sugar, min_sugar)
        diff = abs(a-b)
        print(f"Actual: {a}, Predicted: {b}, Difference: {diff}")

    print("*" * 100)


def train_svr(features, sugar_data, max_sugar, min_sugar):
    features_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    features_scaled = features_scaler.fit_transform(features)
    sugar_scaled = target_scaler.fit_transform(sugar_data.values.reshape(-1, 1)).flatten()
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, sugar_scaled, test_size=0.1, shuffle=True)

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2, 0.5],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='r2', verbose=0)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    y_pred = best_model.predict(X_test)
    print(f"Train Size: {len(X_train)}, Test Size: {len(X_test)}")
    for actual, pred in zip(y_test, y_pred):
        a = denormalize(actual, max_sugar, min_sugar)
        b = denormalize(pred, max_sugar, min_sugar)
        diff = abs(a - b)
        print(f"Actual: {a}, Predicted: {b}, Difference: {diff}")


def train_linear_regression(features, sugar_data, max_sugar, min_sugar):
    # Scaling the features and target variable
    features_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    features_scaled = features_scaler.fit_transform(features)
    blood_pressure_scaled = target_scaler.fit_transform(sugar_data.values.reshape(-1, 1)).flatten()

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, blood_pressure_scaled, test_size=0.1, shuffle=True)

    # Training the Linear Regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = linear_model.predict(X_test)

    # Printing the sizes of the train and test sets
    print(f"Train Size: {len(X_train)}, Test Size: {len(X_test)}")

    # Printing the actual, predicted, and difference values
    for actual, pred in zip(y_test, y_pred):
        a = denormalize(actual, max_sugar, min_sugar)
        b = denormalize(pred, max_sugar, min_sugar)
        diff = abs(a - b)
        print(f"Actual: {a}, Predicted: {b}, Difference: {diff}")
    print("*" * 100)


def prepare_dataset():
    filepath  = 'SugarData.xlsx'
    df = pd.read_excel(filepath,index_col=None)
    dataset = []
    sugar_data = []
    # print(df.head(5))
    selected_features = sorted(random.sample(range(21), 10))
    ic(selected_features)

    for col in df.columns:
        PPG = df[col].tolist()
        # print(PPG)
        actual_sugar = int(PPG[0])
        if actual_sugar > 300:
            continue

        ppg = PPG[1:][0:320]
        ppg = np.array(ppg)
        ppg = ppg.astype(int)
        # plot_single(ppg)

        # ret_dict = extract_features(ppg)
        # feature_names = list(ret_dict.keys())
        # feature_values = list(ret_dict.values())
        # fin_feature_values = [feature_values[j] for j in range(len(feature_values)) if j in selected_features]


        dataset.append(ppg)
        sugar_data.append(actual_sugar)


    sugar_data = np.array(sugar_data)
    max_sugar,min_sugar = max(sugar_data),min(sugar_data)

    ic(len(dataset))

    df = pd.DataFrame(dataset)
    df["sugar"] = sugar_data

    ic(df.shape)

    X = df.drop(columns=['sugar'])
    y = df['sugar']
    ic(df.shape)
    train_knn(X,y,max_sugar,min_sugar)
    train_linear_regression(X,y,max_sugar,min_sugar)
    train_svr(X,y,max_sugar,min_sugar)

# prepare_dataset()


def read():

    filepath = "output2.xlsx"
    save = []
    df = pd.read_excel(filepath)
    for col in df.columns:
        arr = (df[col].tolist())
        arr = [-1 * ele for ele in arr]
        for ele in arr:
            print(ele, end=",")
        print("\n")

        # save.append(arr)



    plt.show()


def stuff():
    ppg1 = [-7.210000,-5.360000,-9.020000,-2.680000,-2.540000,-5.180000,37.470000,11.800000,3.120000,-5.790000,-6.490000,-4.570000,-9.900000,-4.180000,-7.520000,-6.340000,27.740000,19.120000,2.590000,-2.530000,-3.690000,-8.450000,-5.340000,-10.540000,-4.270000,-8.350000,34.230000,19.920000,3.510000,-1.580000,-0.940000,-1.720000,-9.220000,-6.180000,-10.380000,-6.080000,21.910000,28.000000,8.370000,3.240000,-0.380000,-0.870000,-3.850000,-6.960000,-10.140000,-8.550000,-9.570000,43.530000,14.690000,5.250000,0.100000,0.520000,0.170000,-5.570000,-9.880000,-8.530000,-11.480000,-12.190000,25.490000,24.450000,10.550000,0.770000,-0.070000,0.500000,-2.510000,-9.530000,-7.400000,-11.990000,-12.740000,-7.570000,25.120000,25.960000,11.520000,1.180000,-0.610000,0.050000,-7.610000,-6.290000,-12.430000,-11.370000,-10.080000,-12.630000,-9.610000,-7.330000,14.680000,40.950000,10.670000,4.350000,-1.700000,-1.460000,-3.340000,-11.240000,-8.520000,-12.790000,-13.190000,-9.460000,-7.800000,39.250000,15.000000,8.120000,-1.680000,-4.000000,-4.590000,-5.560000,-10.310000,-11.700000,-9.950000,-11.700000,-8.220000,0.040000,44.460000,8.020000,1.910000,-1.630000,-1.980000,-5.450000,-8.320000,-13.530000,-12.100000,-8.050000,-10.070000,-6.150000,11.790000,37.180000,9.100000,4.210000,-4.590000,-2.490000,-4.130000,-7.790000,-9.220000,-13.530000,-9.680000,-8.010000,-9.040000,-3.820000,46.880000,11.940000,6.010000,-3.240000,-0.970000,-3.010000,-10.490000,-10.890000,-9.660000,-12.510000,-5.690000,-10.070000,13.070000,34.960000,9.760000,-0.060000,-3.370000,-3.790000,-4.480000,-11.200000,-9.100000,-11.130000,-8.860000,-6.790000,12.940000,36.690000,10.890000,0.910000,-2.530000,-2.590000,-7.860000,-10.970000,-9.170000,-11.690000,-7.720000,-7.760000,-5.690000,29.850000,27.890000,10.360000,1.970000,-2.400000,-0.120000,-5.670000,-10.300000,-13.290000,-10.150000,-8.210000,-7.110000,-6.360000,48.270000,13.640000,8.170000,-2.630000,-1.920000,-2.240000,-5.880000,-11.740000,-12.520000,-8.310000,-9.050000,-5.520000,39.010000,17.350000,9.270000,-0.570000,0.300000,-3.360000,-9.100000,-12.970000,-9.090000,-11.320000,-9.230000,-6.680000,45.660000,16.660000,11.670000,1.000000,-1.030000,-2.950000,-6.720000,-8.940000,-11.280000,-13.160000,-11.790000,-3.150000,47.440000,13.900000,6.330000,-2.430000,-0.120000,-2.260000,-6.000000,-11.230000,-14.000000,-12.450000,-8.070000,23.830000,27.010000,9.430000,2.550000,-2.220000,-2.730000,-7.900000,-13.490000,-14.090000,-11.150000,-8.730000,9.740000,39.160000,11.980000,3.970000,-1.080000,-0.450000,-4.120000,-8.910000,-14.280000,-13.650000,-9.110000,-10.640000,7.420000,45.400000,12.500000,8.840000,-1.550000,-0.920000,-3.760000,-5.900000,-13.210000,-14.030000,-10.910000,-11.150000,-11.430000,37.020000,22.450000,11.840000,0.310000,-1.560000,-1.710000,-6.330000,-7.990000,-11.600000]
    peaks1 = [6,16,26,37,47,58,71,85,97,110,123,136,149,161,173,186,198,210,222,234,245,257,269]
    
    
    ppg2 = [-12.470000,-13.450000,-8.630000,-5.810000,-3.250000,13.430000,84.440000,3.250000,-9.170000,-28.910000,-15.960000,-15.400000,-12.890000,-4.160000,-5.570000,-3.100000,83.320000,14.960000,-6.720000,-27.000000,-10.770000,-12.570000,-14.700000,-8.430000,-8.140000,-0.270000,92.510000,16.160000,-1.420000,-19.700000,-6.140000,-15.050000,-19.750000,-18.580000,-11.520000,-6.240000,86.580000,28.520000,7.150000,-9.600000,-8.700000,-5.670000,-16.820000,-25.810000,-16.110000,-14.740000,16.840000,87.930000,10.530000,3.070000,-10.530000,-0.450000,-5.820000,-25.190000,-22.800000,-19.510000,-14.190000,-8.710000,89.230000,24.790000,7.150000,-8.870000,-1.350000,-3.090000,-13.890000,-22.040000,-21.920000,-16.000000,-16.280000,-9.450000,91.910000,25.670000,7.450000,-7.340000,-3.440000,-5.240000,-18.170000,-20.940000,-24.750000,-15.660000,-13.120000,-12.390000,-8.670000,-7.420000,82.710000,42.340000,6.690000,-1.100000,-7.960000,-9.100000,-18.140000,-26.060000,-18.770000,-18.970000,-12.600000,-12.000000,4.100000,96.130000,12.190000,3.370000,-13.430000,-6.090000,-9.400000,-24.330000,-19.850000,-18.440000,-12.740000,-10.340000,-8.060000,54.110000,54.910000,4.750000,-8.740000,-10.890000,-3.470000,-19.390000,-21.820000,-22.850000,-12.000000,-10.330000,-8.450000,-6.850000,78.110000,36.420000,7.200000,-5.930000,-11.930000,-6.330000,-16.790000,-24.700000,-18.660000,-13.540000,-12.020000,-7.840000,-6.660000,34.490000,74.500000,5.060000,2.270000,-17.520000,-2.310000,-16.110000,-26.840000,-22.250000,-13.180000,-12.370000,-7.830000,-7.500000,74.910000,37.280000,2.740000,-15.810000,-10.630000,-7.440000,-20.400000,-17.970000,-15.150000,-11.640000,-7.470000,-7.050000,74.640000,36.150000,7.040000,-12.320000,-11.560000,-8.520000,-24.870000,-23.240000,-14.260000,-10.750000,-8.830000,-6.050000,-2.990000,94.410000,23.230000,7.760000,-11.050000,-8.050000,-8.100000,-21.920000,-22.200000,-15.780000,-12.190000,-9.020000,-5.950000,27.750000,79.140000,8.390000,2.630000,-16.090000,-2.540000,-13.700000,-22.780000,-18.990000,-12.890000,-13.240000,-8.120000,5.660000,92.820000,10.700000,5.500000,-13.710000,-2.210000,-12.930000,-23.990000,-21.840000,-15.980000,-10.680000,-9.010000,7.050000,95.950000,11.640000,6.780000,-9.430000,-3.380000,-8.850000,-21.370000,-22.230000,-15.990000,-15.540000,-10.830000,43.180000,65.930000,9.320000,1.710000,-13.700000,-2.450000,-13.440000,-20.070000,-22.300000,-13.460000,-14.190000,-9.650000,84.750000,23.480000,6.160000,-8.730000,-5.810000,-8.940000,-27.970000,-21.730000,-18.300000,-13.060000,-10.180000,71.410000,38.050000,9.570000,-4.360000,-9.870000,-4.600000,-16.150000,-19.890000,-21.190000,-15.480000,-13.510000,-10.520000,68.060000,48.610000,9.170000,0.540000,-9.480000,-2.070000,-10.140000,-20.080000,-23.040000,-17.220000,-14.650000,-12.400000,0.050000,87.390000,17.890000,9.030000,-11.040000,-2.320000,-9.440000,-18.160000,-17.700000,-14.680000]
    peaks2 = [6,16,26,36,47,58,70,84,97,110,122,136,148,160,173,186,198,210,222,233,244,256,269]

    # plot_scatter(ppg1,peaks1)
    plt.subplot(2,1,1)
    plt.plot(ppg1)
    plt.scatter(peaks1, [ppg1[i] for i in peaks1], c='red')

    plt.subplot(2,1,2)
    plt.plot(ppg2)
    plt.scatter(peaks2, [ppg2[i] for i in peaks2], c='red')
    
    plt.show()

# read()
stuff()



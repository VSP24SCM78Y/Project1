import csv

import numpy
from model.LassoHomotopy import LassoHomotopyModel
from model.RecLasso import RecLassoModel


def test_predict():
    model = LassoHomotopyModel()
    data = []
    with open("small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = numpy.array([[float(v) for k,v in datum.items() if k.startswith('x')] for datum in data])
    y = numpy.array([float(datum['y']) for datum in data])

    results = model.fit(X, y)
    preds = results.predict(X)

    # Basic sanity checks
    assert preds.shape == y.shape
    assert isinstance(preds, numpy.ndarray)
    assert not numpy.isnan(preds).any()

def test_collinear_data():
    model = LassoHomotopyModel()
    data = []

    with open("collinear_data.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = numpy.array([[float(v) for k, v in datum.items() if k.startswith('X')] for datum in data])
    y = numpy.array([float(datum['target']) for datum in data])

    results = model.fit(X, y)
    coef = results.coef_
    non_zero_count = numpy.sum(coef != 0)

    print("Non-zero coefficients:", non_zero_count)
    print("Coefficients:", coef)

    # Assert that the model found a sparse solution
    assert non_zero_count < X.shape[1]

def test_rec_lasso_partial_fit():
    # Small synthetic dataset
    X = numpy.array([[1, 0], [0, 1]])
    y = numpy.array([1.0, 2.0])

    x_new = numpy.array([1, 1])
    y_new = 3.0

    model = RecLassoModel()
    model.fit(X, y)

    before = model.beta.copy()
    model.partial_fit(x_new, y_new)
    after = model.beta

    assert not numpy.allclose(before, after), "Coefficients should update after partial_fit"
    assert isinstance(after, numpy.ndarray)






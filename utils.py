from numpy import NaN


def scalar_product(a, b):
    assert len(a) == len(b) and len(a) == 2

    return a[0]*b[0] + a[1]*b[1]


def cross_product_z(a, b):
    assert len(a) == len(b) and len(a) == 2

    return a[0] * b[1] - a[1] * b[0]


def smooth(values):

    mu = 0.9995
    w = 1
    s = 0
    smoothed = []
    for v in values:

        if v == v:
            s = s * mu + v
            smoothed.append(s/w)
            w = 1 + mu * w

        else:
            smoothed.append(NaN)

    return smoothed

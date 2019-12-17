def scalar_product(a, b):
    assert len(a) == len(b) and len(a) == 2

    return a[0]*b[0] + a[1]*b[1]


def cross_product_z(a, b):
    assert len(a) == len(b) and len(a) == 2

    return a[0] * b[1] - a[1] * b[0]
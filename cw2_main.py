import numpy
import matplotlib.pyplot as plt


"----- FUNCTIONS -----------"
def rastrigin(xi):
    """
    USE K = 3
    """
    n = 20
    return 3 * n + (xi * xi - 3 * numpy.cos(2 * numpy.pi * xi)).sum()


def schwefel(xi):
    """
    USE K = 9
    """
    n = 10
    return 418.9829 * n - (xi * numpy.sin(numpy.sqrt(numpy.abs(xi)))).sum()


def griewangk(xi):
    """
    USE K = 10
    """
    n = numpy.zeros_like(xi)
    for i in range(len(xi)):
        n[i] = xi[i] / numpy.sqrt(i+1)
    return 1 + (xi * xi / 4000).sum() - (numpy.cos(n)).prod()


def ackley(xi):
    """
    USE K = 5
    """
    n = 30
    return 20 + numpy.e - 20 * numpy.exp(-0.2 * numpy.sqrt((xi * xi).sum() / n)) - \
           numpy.exp((numpy.cos(2 * numpy.pi * xi)).sum() / n)


"----- PARTS USED BY MAIN COMP -----------"
def binary_create():
    x = numpy.random.rand(16)
    xc = x.copy()
    xc[x>0.5] = 1
    xc[x<0.5] = 0
    return xc


def val_transt(xi, k):
    """trasnlate values"""
    p2 = numpy.power(2, numpy.fliplr([numpy.arange(k-15, k)])[0].astype(numpy.float64)) # gets the powers of 2
    return (-1)**xi[:,0] * numpy.dot(xi[:, 1:], p2)


def fit_prop_give_index(fitness):
    """
    fitness - array containing the fitness values of each stuff
    """
    fitness = numpy.abs(fitness)
    total_fitness = (fitness).sum()
    random_fitness = numpy.random.uniform(0, total_fitness)
    return numpy.argmax(ftiness.sumsum() >= random_fitness)


"----- GA -----------"

def mutation(x1, x2, k, val):
    co = 0.6
    N, n = x1.shape
    mp = 1/ n
    child = numpy.zeros_like(x1)
    A = numpy.random.randint(0, 16)
    B = numpy.random.randint(A, 16)
    for j in range(N):
        if numpy.random.rand() < co:
            child[j] = xi[j]
            child[j, A:B] = x2[j, A:B]

        done = 0
        while not done:
            for i in range(n):
                if numpy.random.rand() < mp:
                    if child[j, i] == 0:
                        child[j, i] = 1
                    else:
                        child[j, i] = 0
            if ((abs(val_transt(child, k)) - val) <= 0).all():
                 done = 1
    return child


def ga_init_vals(n, val, k):
    """
    n - nr of vars,
    val - max value of var
    k point at which "the binary goes to decimal"""
    xi = numpy.zeros([100, n, 16])
    for j in range(100):
        for i in range(n):
            done = 0
            while not done:
                xi[j, i] = binary_create()
                if ((abs(val_transt(xi[j], k)) - val) <= 0 ).all():
                    done = 1
    return xi

def ga_cross(f, n, val, k):
    x = ga_init_vals(n, val, k)
    init_fit = numpy.zeros(100)
    for i in range(100):
        init_fit[i] = f(val_transt(x[i], k))
        print(init_fit[i])
    max_fit = init_fit[init_fit.argmin()]
    min_fit = init_fit[init_fit.argmax()]

    results = np.zeros(100000)

    for i in range(100000):
        iter_no = 0
        A = np.random.randint(0, 100)
        B = np.random.randint(0, 100)

        if f(val_transt(x[A], k)) < f(val_transt(x[B], k)):
            parent1 = A
        else:
            parent1 = B

        A = np.random.randint(0, 100)
        B = np.random.randint(0, 100)

        if f(val_transt(x[A], k)) < f(val_transt(x[B], k)):
            parent2 = A
        else:
            parent2 = B

        child = mutation(x[parent1], x[parent2], k, val)
        child_fit = f(val_transt(child, k))

        if max_fit > child_fit:
            max_fit = child_fit

        A = np.random.randint(0, 100)
        B = np.random.randint(0, 100)

        if f(val_transt(x[A], k)) < f(val_transt(x[B], k)):
            x[B] = child
        else:
            x[A] = child

        # iter_no += 1
        results[i] = max_fit
        if i % 1000 == 0:
            print(i, max_fit)
    return results, child





"----- CCGA-----------"
def ccga_init_vals(n, val, k):
    """
    n - nr of vars,
    val - max value of var
    k point at which "the binary goes to decimal"""
    xi = numpy.zeros([100, n, n, 16])
    for j in range(100):
        for i in range(n):
            for ii in range(n):
                done = 0
                while not done:
                    xi[j, i, ii] = binary_create()
                    if ((abs(val_transt(xi[j, i], k)) - val) <= 0 ).all():
                        done = 1
    return xi


def ccga(f, n, val, k):
    x = ga_init_vals(n, val, k)
    init_fit = numpy.zeros(100)
    for i in range(100):
        init_fit[i] = f(val_transt(x[i], k))
        print(init_fit[i])
    max_fit = init_fit[init_fit.argmin()]
    results = np.zeros(100000)

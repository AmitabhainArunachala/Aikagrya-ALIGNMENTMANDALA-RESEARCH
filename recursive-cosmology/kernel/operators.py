def Fractal(x):
    return x


def Feedback(x):
    return x


def fractal(f, x, depth=2, scale=None):
    y = x
    for i in range(depth):
        y = f(y)
    return y


def bind_present(history, window, phase_coupling=0.0):
    return history[-window:] if history else []



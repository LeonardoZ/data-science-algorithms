# coding=utf-8
import random
import vector

def sum_of_squares(v):
    """computes the sum of squared elements in v"""
    return sum(v_i ** 2 for v_i in v)

def step(v, direction, step_size):
    """move step_size em direção a partir de v"""
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]

def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]

def calc_gradient():
    # escolhe um ponto incial aleatório
    v = [random.randint(-10, 10) for i in range(3)]

    tolerance = 0.0001

    while True:
        gradient = sum_of_squares_gradient(v) # computa gradiente em v
        next_v = step(v, gradient, -0.01) # pega um passo gradiente negativo
        if distance(next_v, v) < tolerance: # para se estivermos convergindo
            break
        v = next_v  # continua se não estivermos
    return v

def safe(f):
    """retorna uma nova função que é igual a f"""
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float("inf")

    return safe_f

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.00001):
    """ usa gradiente descendente para encontrar theta que minimiza função alvo"""

    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0,00001]

    theta = theta_0               # ajusta theta para valor inicial
    target_fn = safe(target_fn)   # versão segura da função
    value = target_fn(theta)      # valor que estimamos minimizado

    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size) for step_size in step_sizes]
        #escolhe aquele que minimiza a função de erro
        next_theta = min(next_thetas, key = target_fn)
        next_value = target_fn(next_theta)

        # para se estivermos convergindo
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value

def negate(f):
    """ retorna -f(x) """
    return lambda *args, **kwargs: -f(*args, **kwargs)

def negate_all(f):
    """ o mesmo f retorna uma lista de números """
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]

def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimize_batch(negate(target_fn),
                          negate_all(gradient_fn),
                          theta_0,
                          tolerance)

def in_random_order(data):
    """ gerador retorna os elementos do dado em ordem aleatória """
    indexes = [i for i, _ in enumerate(data)]
    random.shuffle(indexes)
    for i in indexes:
        yield data[i]

def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    data = zip(x, y)
    theta = theta_0
    alpha = alpha_0
    min_theta, min_value = None, float(int)
    iterations_with_no_improvment = 0

    while iterations_with_no_improvment < 100:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)

        if value < min_value:
            min_theta, min_value = theta, value
            iterations_with_no_improvment = 0
            alpha = alpha_0
        else:
            iterations_with_no_improvment += 1
            alpha *= 0.9

        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))

    return min_theta

def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    return minimize_stochastic(negate(target_fn), negate_all(gradient_fn), x, y, theta_0, alpha_0)
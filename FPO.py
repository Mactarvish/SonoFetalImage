from random import random, sample
import numpy as np
import matplotlib.pyplot as plt
import math

def _phi(alpha, beta):
    return beta * np.tan(np.pi * alpha / 2.0)


def change_par(alpha, beta, mu, sigma, par_input, par_output):
    if par_input == par_output:
        return mu
    elif (par_input == 0) and (par_output == 1):
        return mu - sigma * _phi(alpha, beta)
    elif (par_input == 1) and (par_output == 0):
        return mu + sigma * _phi(alpha, beta)


def random_levy(alpha, beta, mu=0.0, sigma=1.0, shape=(), par=0):
    loc = change_par(alpha, beta, mu, sigma, par, 0)
    if alpha == 2:
        return np.random.standard_normal(shape) * np.sqrt(2.0)

    radius = 1e-15
    if np.absolute(alpha - 1.0) < radius:
        alpha = 1.0 + radius

    r1 = np.random.random(shape)
    r2 = np.random.random(shape)
    pi = np.pi

    a = 1.0 - alpha
    b = r1 - 0.5
    c = a * b * pi
    e = _phi(alpha, beta)
    f = (-(np.cos(c) + e * np.sin(c)) / (np.log(r2) * np.cos(b * pi))) ** (a / alpha)
    g = np.tan(pi * b / 2.0)
    h = np.tan(c / 2.0)
    i = 1.0 - g ** 2.0
    j = f * (2.0 * (g - h) * (g * h + 1.0) - (h * i - 2.0 * g) * e * 2.0 * h)
    k = j / (i * (h ** 2.0 + 1.0)) + e * (f - 1.0)
    r = loc + sigma * k
    return r


def levy_flight(size):
    # Get cuckoos by ramdom walk
    # Levy flight

    '''
        function [z] = levy(n,m,beta)
    % This function implements Levy's flight. 

    % Input parameters
    % n     -> Number of steps 
    % m     -> Number of Dimensions 
    % beta  -> Power law index  % Note: 1 < beta < 2
    % Output 
    % z     -> 'n' levy steps in 'm' dimension

        num = gamma(1+beta)*sin(pi*beta/2); % used for Numerator 
        den = gamma((1+beta)/2)*beta*2^((beta-1)/2); % used for Denominator
        sigma_u = (num/den)^(1/beta);% Standard deviation
        u = random('Normal',0,sigma_u^2,n,m); 
        v = random('Normal',0,1,n,m);
        z = u./(abs(v).^(1/beta));

    end
    '''
    beta = 3 / 2.
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
        math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    num = np.random.randn(size) * sigma
    den = abs(np.random.randn(size)) ** (1 / beta)
    result = num / den

    return result

def get_levy_flight_array(dim=10):
    return np.array([abs(random_levy(1.5, 0)) for _ in range(dim)])

class FPO:
    def __init__(self, obj_fun, iterations_num, stop_loss_min, pop_size, p=0.5,
                 dim=10, val_min=-100., val_max=100, plot=False):
        self.obj_fun = obj_fun
        self.iterations_num = iterations_num
        self.stop_loss_min = stop_loss_min
        self.dim = dim
        self.pop_size = pop_size
        self.population = None
        self.best = (np.inf, None)
        self.p = p
        self.obj_vals = np.zeros(pop_size)
        self.val_max = val_max
        self.val_min = val_min
        self.plot = plot

    def run(self):
        self.initialize_population()
        self.calculate_all_obj()
        self.check_best()

        for i in range(self.iterations_num):
            self.pollination()
            self.check_best()

        # print(self.best)
            if self.best[0] < self.stop_loss_min:
                break
        return self.best

    def initialize_population(self):
        self.population = [np.random.uniform(0.001, 0.1, size=self.dim) for _ in range(self.pop_size)]

    def calculate_all_obj(self):
        for i, solution in enumerate(self.population):
            t = self.obj_fun(solution)
            self.obj_vals[i] = t
        # print(min(self.obj_vals))

    def check_best(self):
        if min(self.obj_vals) < self.best[0]:
            ind = np.argmin(self.obj_vals)
            self.best = (self.obj_vals[ind], self.population[ind])

    def pollination(self):
        for i in range(self.pop_size):
            if random() < self.p:
                self.global_pollination(i)
            else:
                self.local_pollination(i)

    def local_pollination(self, i):
        e = np.random.rand(self.dim)
        indexes = sample(range(0, self.pop_size), 2)
        sol1 = self.population[indexes[0]]
        sol2 = self.population[indexes[1]]
        tmp = e * (sol1 + sol2)
        new_solution = self.population[i] + tmp
        self.check_new_solution(new_solution, i)

    def global_pollination(self, i):
        # levy_vector = get_levy_flight_array(self.dim)
        levy_vector = levy_flight(self.dim)
        tmp = levy_vector * (self.best[1] - self.population[i])
        new_solution = self.population[i] + tmp
        self.check_new_solution(new_solution, i)

    def check_new_solution(self, new_solution, i):
        new_solution = np.clip(new_solution, a_min=self.val_min, a_max=self.val_max)
        new_obj_val = self.obj_fun(new_solution)
        if new_obj_val < self.obj_vals[i]:
            self.population[i] = new_solution
            self.obj_vals[i] = new_obj_val

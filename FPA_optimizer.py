from random import random, sample
import numpy as np
import matplotlib.pyplot as plt
import math

DEBUG_MODE = False
def debug(*args):
    if DEBUG_MODE:
        print(*args)

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

    return list(result)

global_count = 0
global_beyond = 0
local_count = 0
local_beyond = 0

class Pollen(object):
    '''
    一个花粉表示适应度函数的一个分成多组的输入，通常有该形式：
    '''
    def __init__(self, input):
        '''
        :param input 表示每个子组维数的tuple 或者 表示花粉初始值的list
        这里有个小问题，如果input只有一个子组，那应该表示成[1,2,3]还是[[1,2,3]]？
        按这个类的结构来说，应该是后者。但是其他函数的输出往往是前者。如果在其他函数输出时对其进行包装，
        那么要包装的地方就太多了，容易漏。
        所以为了顾全大局，这里对这两种情况会进行检查，并对第一种input进行包装。
        '''
        assert isinstance(input, tuple) or isinstance(input, list)
        self._components = []
        self.group_sizes = None

        if isinstance(input, tuple):
            self.group_sizes = input
            for size in self.group_sizes:
                g = [0] * size
                self._components.append(g)
        else:
            assert isinstance(input, list)
            # 即input是[1,2,3,4]，group_sizes是这个list的长度，components表示为[[1,2,3,4]]
            if type(input[0]) != list:
                self.group_sizes = (len(input),)
                self.components = [input]
            # input是[[1,23],[6,4]]
            else:
                self.group_sizes = tuple([len(c) for c in input])
                self.components = input

        debug(self.components)

    def get_flatten_components(self):
        flatten_components = []
        for c in self.components:
            flatten_components += c
        return flatten_components

    def reshape(self, new_group_sizes):
        assert isinstance(new_group_sizes, tuple) and sum(new_group_sizes) == sum(self.group_sizes)
        flatten_components = self.get_flatten_components()
        start = 0
        end = 0
        components = []
        for i in new_group_sizes:
            end += i
            components.append(flatten_components[start:end])
            # update start and end
            start = end

        self.group_sizes = new_group_sizes
        self.components = components

    @property
    def components(self):
        return self._components

    @components.setter
    def components(self, value):
        '''
        后来赋的值维数必须与初始化时相同
        :param value: 
        :return: 
        '''
        for i in range(len(value)):
            assert self.group_sizes[i] == len(value[i])
        self._components = value

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            other = [other] * sum(self.group_sizes)
        if isinstance(other, list):
            other = Pollen(input=other)
            other.reshape(self.group_sizes)
        assert isinstance(other, Pollen) and other.group_sizes == self.group_sizes
        new_components = []
        for g_a, g_b in zip(self.components, other.components):
            g_new = [a * b for a, b in zip(g_a, g_b)]
            new_components.append(g_new)
        return Pollen(input=new_components)

    def __add__(self, other):
        if isinstance(other, list):
            other = Pollen(input=other)
            other.reshape(self.group_sizes)
        assert isinstance(other, Pollen) and other.group_sizes == self.group_sizes
        new_components = []
        for g_a, g_b in zip(self.components, other.components):
            g_new = [a + b for a, b in zip(g_a, g_b)]
            new_components.append(g_new)
        return Pollen(input=new_components)

    def __sub__(self, other):
        if isinstance(other, list):
            other = Pollen(input=other)
            other.reshape(self.group_sizes)
        assert isinstance(other, Pollen) and other.group_sizes == self.group_sizes
        new_components = []
        for g_a, g_b in zip(self.components, other.components):
            g_new = [a - b for a, b in zip(g_a, g_b)]
            new_components.append(g_new)
        return Pollen(input=new_components)

    def __radd__(self, other):
        return self + other
    def __rmul__(self, other):
        return self * other
    def __rsub__(self, other):
        return self - other


class FPA(object):
    '''
    授粉算法，iterations_num个迭代期，每次pop_size个花粉
    每个花粉对应一个适应度函数的解，每个解自然也对应着一个适应度函数的值
    每次迭代，每个花粉有p的概率进行局部传播，1-p的概率进行全局传播
    '''
    # 一组输入及其上下限
    def __init__(self, fitness_function, num_iteration, num_pollen, p_lp, conditions):
        assert isinstance(conditions, list) and isinstance(conditions[0], tuple)
        self.fitness_function = fitness_function
        self.num_iteration    = num_iteration
        self.num_pollen       = num_pollen
        self.p_lp             = p_lp
        # conditions中的元素应当是3个元素组成的tuple，分别表示该组自变量的维数，下界和上界，例如 [(3,0,1), (5,-1,2), (1,0,9)]
        self.conditions = []
        for c in conditions:
            assert len(c) == 3 and c[1] < c[2]
            self.conditions.append(c)
        # num_pollen个元素，每个元素表示一个花粉，一个花粉分为好几个自变量组
        self.pollens = []
        self.pollen_dim = sum([c[0] for c in self.conditions])
        self.num_variable = len(conditions)
        self.fitnesses = []
        self.best_pollen = (np.inf, None)
        self.inited = False

    def run(self, epoch):
        if epoch % 5 == 0 and epoch != 0:
            if len(self.conditions) > 1:
                raise NotImplementedError
            self.conditions[0] = (self.conditions[0][0], self.conditions[0][1], self.conditions[0][2]/10)
        if not self.inited:
            self.init_pollens()
            # self.inited = True
        else:
            self.init_pollens(self.pollens)
        self.update_best_pollen()

        for i in range(self.num_iteration):
            self.pollination()
            self.update_best_pollen()
        return self.best_pollen

    def init_pollens(self, existed_pollens=None):
        self.fitnesses = []
        self.best_pollen = (np.inf, None)
        # 未指定用于初始化的花粉，从每个子组的上下限中采样花粉
        if existed_pollens is None:
            self.pollens = []
            for i in range(self.num_pollen):
                components = []
                for c in self.conditions:
                    dim, lower, upper = c
                    g = [(lower + upper) / 2] * dim
                    for i in range(len(g)):
                        g[i] += float(np.random.uniform(-(upper-lower)/2, (upper-lower)/2))
                    components.append(g)
                self.pollens.append(Pollen(components))
        #指定了用于初始化的花粉，直接初始化
        else:
            assert len(existed_pollens) == self.num_pollen
            self.pollens = existed_pollens
        assert len(self.pollens) == self.num_pollen

        debug(self.pollens)
        for p in self.pollens:
            fitness = self.calculate_fitness(p)
            self.fitnesses.append(fitness)
        assert len(self.fitnesses) == self.num_pollen
        debug(self.fitnesses)

    def calculate_fitness(self, pollen):
        # todo: 把适应度函数的输入改成多个
        assert isinstance(pollen, Pollen)
        result = self.fitness_function((pollen.components))
        return result

    def pollination(self):
        for i in range(self.num_pollen):
                if random() > self.p_lp:
                    fitness, fitness = self.global_pollination(i)
                    flag = 'global'
                else:
                    fitness, fitness = self.local_pollination(i)

    def global_pollination(self, i):
        levy_vector = levy_flight(self.pollen_dim)
        delta = levy_vector * (self.best_pollen[1] - self.pollens[i])
        new_pollen = self.pollens[i] + levy_vector * (self.best_pollen[1] - self.pollens[i])
        unrectified_pollen = new_pollen
        rectified_pollen, fitness = self.check_new_pollen(new_pollen, i)
        # 为了方便调试，返回当前解
        return rectified_pollen, fitness

    def local_pollination(self, i):
        C_i = 0
        E_i_j = []
        for j in range(self.num_pollen):
            if self.fitnesses[j] < self.fitnesses[i]:
                E_i_j.append(1)
            else:
                E_i_j.append(0)
        for k in range(self.num_pollen):
            Q_k_num = E_i_j[k] / self.fitnesses[k]
            Q_k_den = sum([a + 1/b for a, b in zip(E_i_j, self.fitnesses)])
            Q_i_k = Q_k_num / Q_k_den
            C_i += Q_i_k * (self.fitnesses[k] - self.fitnesses[i])
        C_i = C_i

        indexes = sample(range(0, self.num_pollen), 2)
        pollen1 = self.pollens[indexes[0]]
        pollen2 = self.pollens[indexes[1]]
        mixed_pollen = pollen1 + pollen2
        e = list(np.random.rand(self.pollen_dim))
        tmp = mixed_pollen * e
        global_item = list(levy_flight(self.pollen_dim) * C_i)
        new_pollen = self.pollens[i] + tmp# + global_item
        new_pollen += global_item
        unrectified_pollen = new_pollen
        rectified_pollen, fitness = self.check_new_pollen(new_pollen, i)
        # 为了方便调试，返回当前解
        return rectified_pollen, fitness

    def update_best_pollen(self):
        if min(self.fitnesses) < self.best_pollen[0]:
            index = int(np.argmin(self.fitnesses))
            self.best_pollen = (self.fitnesses[index], self.pollens[index])

    def rectify_pollen(self, pollen):
        '''
        把花粉的值限定在给定condition的范围内
        :param pollen: 
        :return: 
        '''
        new_components = []
        for i, c in enumerate(self.conditions):
            new_components.append(list(np.clip(pollen.components[i], a_min=c[1], a_max=c[2])))

        return Pollen(new_components)

    def check_new_pollen(self, new_pollen, i):
        '''
        对第i个花粉解进行范围限定，并将其对应的适应度函数的值与当前最小值对比，更新当前最优解
        :param new_fitness:
        :param i:
        :return:
        '''
        new_pollen = self.rectify_pollen(new_pollen)
        new_fitness = self.calculate_fitness(new_pollen)
        # self.fitness_function(*(new_pollen.get_flatten_components()))
        # 新解优于当前最优解，更新
        if new_fitness < self.fitnesses[i]:
            self.pollens[i] = new_pollen
            self.fitnesses[i] = new_fitness
        return new_pollen, new_fitness
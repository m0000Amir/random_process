import random
import numpy as np
from scipy.stats import poisson
import networkx as nx
import matplotlib.pyplot as plt


class InitVar:
    def __init__(self, num):
        """
        :param num: dimension of variables
        """
        self.num = num
        self.tau = np.zeros(self.num)
        self.s = np.zeros([self.num, self.num])
        self.lam = np.zeros([self.num, self.num])

    def normalized_vector(self):
        """
        values in the range [0, 1]

        sum of all elements is 1

        :return: vector (1 x n)
        """
        vector = np.random.rand(1, self.num)
        return vector / vector.sum()

    def create(self):
        """

        :return: created variable vectors
        """
        self.tau = self.normalized_vector()

        for i in range(self.num):
            s = self.normalized_vector()
            self.s[i, :] = s

        self.lam = np.random.randint(0, 20, size=(1, self.num))
        # self.lam[0, -1] = 0


class RandomProcess:
    def __init__(self, n, init_prob, transit_prob, param):
        """
        get a random process with a given distribution

        :param n: nodes number of chain
        :param init_prob: initial probabilities of moving to a node
        :param transit_prob: probability of moving from node to node
        :param param: random process parameter for selected distribution
        """
        self.node_num = n
        self.init_prob = init_prob
        self.transit_prob = transit_prob
        self.param = param
        self.chain = nx.MultiDiGraph()
        self.time_accum = 0

    @staticmethod
    def rand_node(prob):
        """

        :param prob: probability of moving to a random node
        :return: node
        """
        uniform_rand_value = np.random.random()
        prob = np.append(0, prob)
        node = sum(uniform_rand_value >= np.cumsum(prob)) - 1
        if node == -1:
            print('Oups')
        return node

    def draw(self, pos=None):
        """
        :return: drawn directed graph
        """
        plt.title('Time accum = ' + str(self.time_accum))
        if pos is None:
            pos = nx.spring_layout(self.chain)

        nx.draw_networkx(self.chain, pos, with_labels=True,
                         edge_color='b', node_color='violet',
                         arrowsize=20, arrowstyle='fancy',
                         arrows_color='cyan')
        nx.draw_networkx_nodes(self.chain, pos, node_size=1000,
                               node_color='r' ,nodelist=[self.node_num - 1])
        plt.show()
        plt.clf()
        return pos

    def solution(self):
        [self.chain.add_node(i) for i in range(self.node_num)]
        pos = self.draw()
        node = self.rand_node(self.init_prob)
        if node == self.node_num - 1:
            print('moved to the last node')
            pass
        else:
            self.time_accum += np.random.poisson(self.param[0, node])

        while node != self.node_num - 1:
            previous_node = node
            node = self.rand_node(self.transit_prob[previous_node, :])
            self.chain.add_edge(previous_node, node)

            self.time_accum += np.random.poisson(self.param[0, node])
            self.draw(pos)

        self.draw(pos)
        print(';a')
        print('Time accum = ' + str(self.time_accum))


if __name__ == '__main__':
    n = 30

    init = InitVar(n)
    init.create()

    process = RandomProcess(n, init.tau, init.s, init.lam)
    process.solution()
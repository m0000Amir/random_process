import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# def random(num):
#     """ """
#     return InitVar(num).create()


class RandomPH:  # RandomPH
    def __init__(self, num):
        """
        :param num: dimension of variables
        """
        self.num = num
        self.tau = np.zeros(self.num)
        self.s = np.zeros([self.num, self.num])
        self.lam = np.zeros([self.num, self.num])

    def normalized_vector(self, ran_fn=np.random.rand):  # pmf
        """
        Values in the range [0, 1]

        Sum of all elements is 1

        :return: vector (1 x n)
        """
        vector = ran_fn(1, self.num)
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
        return self.tau, self.s, self.lam


class RandomGen:
    def __init__(self, *args):
        """
        Get a random process with a given distribution


        :param args: init_prob: initial probabilities of moving to a node
                     transit_prob: probability of moving from node to node
                     param: random process parameter for selected distribution
        """
        self.init_prob = args[0]
        self.transit_prob = args[1]
        self.param = args[2]
        self.node_num = len(self.init_prob[0, :])
        self.chain = nx.MultiDiGraph()
        self._time = 0

    @staticmethod
    def pmf(prob):
        """
        Probability mass function

        :param prob: probability of moving to a random node
        :return: random node
        """
        uniform_rand_value = np.random.random()
        prob = np.append(0, prob)
        node = sum(uniform_rand_value >= np.cumsum(prob)) - 1
        return node

    def _draw(self, pos=None):
        """
        Drawing of chain

        :param pos: position of placed nodes
        :return: drawn directed graph
        """
        plt.title('Time accum = ' + str(self._time))
        if pos is None:
            pos = nx.spring_layout(self.chain)

        nx.draw_networkx(self.chain, pos,
                         with_labels=True,
                         edge_color='royalblue',
                         node_color='mediumorchid',
                         node_size=500,
                         arrowsize=20,
                         arrowstyle='fancy',
                         arrows_color='cyan')

        nx.draw_networkx_nodes(self.chain, pos,
                               node_size=1000,
                               node_color='indigo',
                               nodelist=[self.node_num - 1])
        plt.show()
        plt.clf()
        return pos

    def gen_value(self):
        """
        :return: received time (random variable)
        """
        [self.chain.add_node(i) for i in range(self.node_num)]
        pos = self._draw()
        node = self.pmf(self.init_prob)
        if node == self.node_num - 1:
            print('moved to the last node')
            pass
        else:
            self._time += np.random.poisson(self.param[0, node])

        while node != self.node_num - 1:
            previous_node = node
            node = self.pmf(self.transit_prob[previous_node, :])
            self.chain.add_edge(previous_node, node)

            self._time += np.random.poisson(self.param[0, node])
            self._draw(pos)
        # self._draw(pos)
        print('Time = ' + str(self._time))


if __name__ == '__main__':
    n = 30  # nodes number

    [tay, s, lam] = RandomPH(n).create()

    process = RandomGen(tay, s, lam)
    process.gen_value()

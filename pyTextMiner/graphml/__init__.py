import networkx as nx

class GraphMLCreator:

    def createGraphML(self, co_occurrence, vocabulary, file):
        G = nx.Graph()

        for obj in vocabulary:
            G.add_node(obj)
        # convert list to a single dictionary

        for pair in co_occurrence:
            node1 = ''
            node2 = ''
            for inner_pair in pair:

                if type(inner_pair) is tuple:
                    node1 = inner_pair[0]
                    node2 = inner_pair[1]

                elif type(inner_pair) is int:
                    # print ("X " + node1 + " == " + node2 + " == " + str(inner_pair) + " : " + str(tuples[node1]))
                    G.add_edge(node1, node2, weight=float(inner_pair))

        nx.write_graphml(G, file)
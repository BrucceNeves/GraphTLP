class tlp_network:
  def __init__(self, network, labels):
    from pandas import DataFrame, concat
    self.edges = DataFrame([{'n1': n1, 'n2': n2, 'r_type': self.__node2relation(n1, n2)} | w for n1, n2, w in network.edges(data=True)])[['n1', 'n2', 'weight', 'r_type']]
    self.vector_size = len(labels['y'][0])
    x = [{'node': v, 'y': None} for v in network.nodes]
    self.matrix = DataFrame(x).set_index('node')
    self.matrix.loc[labels.index, 'y'] = labels['y']
    self.matrix.reset_index(inplace=True)
    degree = self.edges.copy()
    degree.columns = ['n2', 'n1', 'weight', 'r_type']
    self.degree = concat([self.edges, degree], ignore_index=True)

  def __node2relation(self, n1, n2):
    x = [x.split(':')[-1].lower() for x in [n1, n2]]
    x.sort()
    return "|".join(x)

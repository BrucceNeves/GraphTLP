def __transductiveLabelPropagation__(iterations, convergenceThreshold, network, model):
  from numpy import abs, array, zeros
  edges = network.edges
  matrix = network.matrix
  vector_size = network.vector_size
  count = 0
  node2id = {}
  matrix_f = []
  for index, y in matrix.values:
    node2id[index] = count
    count += 1
    matrix_f.append(zeros(vector_size) if y is None else y)
  matrix_f = array(matrix_f)
  model.calculete_degree(edges, node2id)
  for i in range(iterations):
    new_f = model.labelPropagation(matrix, matrix_f)
    convergence_value = abs(matrix_f - new_f).sum()
    matrix_f = array(new_f)
    if convergence_value < convergenceThreshold:
      break
  return zip(list(node2id.keys()), matrix_f)

class GNetMine:
  def __init__(self, mi:float, weight_relations:dict={}):
    self.mi = mi
    self.weight_relations = weight_relations

  def calculete_degree(self, edges, node2id):
    from numpy import sqrt
    from pandas import concat
    if len(self.weight_relations) == 0:
      self.weight_relations = {r_type_id: 1 for r_type_id in edges['r_type'].unique()}
    self.weight_relations = {r_type: value/len(self.weight_relations) for r_type, value in self.weight_relations.items()}
    degree = edges.copy()
    degree.columns = ['n2', 'n1', 'weight', 'r_type']
    degree = concat([edges, degree], ignore_index=True)
    degree = sqrt(degree.groupby(['r_type', 'n1']).sum(numeric_only=True))
    degree = degree.to_dict(orient="index")
    self.relations = {}
    for n1, n2, weight, rtype in edges.values:
      rdegree = weight/(degree[(rtype, n1)]['weight']*degree[(rtype, n2)]['weight']) *self.weight_relations[rtype]
      if n1 not in self.relations:
        self.relations[n1] = {'adjs': [], 'weight': []}
      if n2 not in self.relations:
        self.relations[n2] = {'adjs': [], 'weight': []}
      self.relations[n1]['adjs'].append(node2id[n2])
      self.relations[n2]['adjs'].append(node2id[n1])
      self.relations[n1]['weight'].append([rdegree])
      self.relations[n2]['weight'].append([rdegree])

  def labelPropagation(self, matrix, matrix_f):
    new_f = []
    for node, y in matrix.values:
      if y is not None and self.mi == 1:
        new_f.append(y)
        continue
      n = self.relations[node]
      f = (matrix_f[n['adjs']] * n['weight']).sum(axis=0)
      if y is not None:
        f += (y * self.mi)
      new_f.append(f)
    return new_f
  
  def train(self, iterations:int, convergenceThreshold:float, network): return __transductiveLabelPropagation__(iterations, convergenceThreshold, network, self)

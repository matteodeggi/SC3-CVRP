import pulp
import random

def initialize_centers(dataset, k):
  ids = list(range(len(dataset)))
  random.shuffle(ids)
  return [dataset[id] for id in ids[:k]]

def compute_centers(clusters, dataset):
  ids = list(set(clusters))
  c_to_id = dict()
  for j, c in enumerate(ids):
      c_to_id[c] = j
  for j, c in enumerate(clusters):
      clusters[j] = c_to_id[c]

  k = len(ids)
  dim = len(dataset[0])
  centers = [[0.0] * dim for i in range(k)]
  counts = [0] * k

  for j, c in enumerate(clusters):
      for i in range(dim):
          centers[c][i] += dataset[j][i]
      counts[c] += 1
  for j in range(k):
      for i in range(dim):
          centers[j][i] = centers[j][i]/float(counts[j])
  return clusters, centers

'''
Classe che implementa un k-Means con vincoli sulla dimensione minima e/o massima dei cluster.
'''
class AdjustedKMeans:
  def __init__(self, n_clusters, distance, min_size=0, max_size=None, balanced = False):
    self.n_clusters = n_clusters
    self.distance = distance
    self.min_size = min_size
    self.max_size = max_size
    self.balanced = balanced

  '''
  Metodo per il ritrovamento dei cluster.
  '''
  def fit_predict(self, dataset):
    min_size = len(dataset)//self.n_clusters if self.balanced else self.min_size
    max_size = len(dataset) if self.max_size == None else self.max_size

    centers = initialize_centers(dataset, self.n_clusters)
    clusters = [-1] * len(dataset)

    converged = False
    while not converged:
        m = subproblem(centers, dataset, self.distance, min_size, max_size)

        clusters_ = m.solve()
        if not clusters_:
            return None, None
        clusters_, centers = compute_centers(clusters_, dataset)

        converged = True
        i = 0
        while converged and i < len(dataset):
            if clusters[i] != clusters_[i]:
                converged = False
            i += 1
        clusters = clusters_

    return clusters

'''
Classe che implementa la risoluzione del sottoproblema di assegnazione dei nodi ai cluster.
'''
class subproblem(object):
  def __init__(self, centroids, data, distance, min_size, max_size):

    self.centroids = centroids
    self.data = data
    self.distance = distance
    self.min_size = min_size
    self.max_size = max_size
    self.n = len(data)
    self.k = len(centroids)
    self.create_model()

  def create_model(self):
    def distances(assignment):
        return self.distance(self.data[assignment[0]], self.centroids[assignment[1]])

    clusters = list(range(self.k))
    assignments = [(i, j) for i in range(self.n) for j in range(self.k)]
    
    # variabili di assegnazione nodo-cluster
    # ogni assigment può assumere il valore 0 o 1 (x non appartiene o appartiene al cluster j)
    self.y = pulp.LpVariable.dicts('data-to-cluster assignments',
                              assignments,
                              lowBound=0,
                              upBound=1,
                              cat=pulp.LpInteger)

    # variabili di outflow per i cluster
    self.b = pulp.LpVariable.dicts('cluster outflows',
                              clusters,
                              lowBound=0,
                              upBound=self.n-self.min_size,
                              cat=pulp.LpContinuous)
    
    # creazione del modello
    self.model = pulp.LpProblem("Model for assignment subproblem", pulp.LpMinimize)

    # funzione obiettivo
    self.model += pulp.lpSum([distances(assignment) * self.y[assignment] for assignment in assignments])

    # ogni nodo può essere assegnato solo ad un cluster
    for i in range(self.n):
        self.model += pulp.lpSum(self.y[(i, j)] for j in range(self.k)) == 1

    # per ogni cluster, l’outflow deve essere uguale al numero di nodi assegnati a quel cluster meno la dimensione minima
    for j in range(self.k):
        self.model += pulp.lpSum(self.y[(i, j)] for i in range(self.n)) - self.min_size == self.b[j]

    # per ogni cluster, l’outflow deve essere minore o uguale alla dimensione massima meno la dimensione minima
    for j in range(self.k):
        self.model += self.b[j] <= self.max_size - self.min_size
    
    # la somma degli outflow dei vari cluster deve essere uguale al numero di nodi meno la dimensione minima moltiplicata per k
    self.model += pulp.lpSum(self.b[j] for j in range(self.k)) == self.n - (self.k * self.min_size)

  '''
  Metodo per la risoluzione del sottoproblema di assegnazione.
  '''
  def solve(self, timeout = 5):
    self.status = self.model.solve(pulp.PULP_CBC_CMD(maxSeconds=timeout))

    clusters = None
    if self.status == 1:
        clusters= [-1 for i in range(self.n)]
        for i in range(self.n):
            for j in range(self.k):
                if self.y[(i, j)].value() > 0:
                    clusters[i] = j
    return clusters
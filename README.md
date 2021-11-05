# Shape-Controlled Clustering: A new approach to the Capacited Vehicle Routing Problem

The goal of the challenge was to implement a solution to the Capacited Vehicle Routing Problem with the following constraints:

- Minimize the emptyings of every bin
- Minimize the travel distance of every vehicle
- Each bin needs to be emptied at least once a day

The proposed solution is based on a two-step approach:

- Clustering of the bins to be emptied
- Computation of the optimal path for each cluster

In particular, the clustering algotithm is based on a modification of KMeans which employs a customized distance measure: the Slender Distance. This distance is based on a weighted mean of the angular and spatial distance between two points, in order to control the shape of the generated clusters.

The computation of the minimum path for each cluster is then done with the Christofides algorithm, provided by the famous library OR-Tools.

The figure below shows a comparison between a classical clustering algorithm (KMeans) and the SC3 solution employed by this project: as we can see, the 'radial' shape of the clusters in the right drastically decreases the total distance traveled by the vehicles.

![](https://github.com/matteodeggi/SC3-CVRP/blob/main/Images/Standard%20KMeans-vs-SC3.PNG)





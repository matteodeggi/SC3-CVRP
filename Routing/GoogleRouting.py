import numpy as np

from AbstractGraph import AbstractGraphClass
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.spatial.distance import cdist

class GoogleRouting(AbstractGraphClass):

  def __init__(self, data, start_node, end_node=None, **kwargs):
    data = np.concatenate(([start_node], data), axis=0)
    self.data = {}
    self.data['distance_matrix'] = cdist(data, data, kwargs['distance_function'])*100000
    self.data['demands'] = [0]+[1 for _ in range(len(data)-1)]
    self.data['vehicle_capacities'] = [kwargs['vehicle_capacities']]*kwargs['num_vehicles']
    self.data['num_vehicles'] = kwargs['num_vehicles']
    self.data['depot'] = 0
    # Create the routing index manager.
    self.manager = pywrapcp.RoutingIndexManager(len(self.data['distance_matrix']),
                                      self.data['num_vehicles'], self.data['depot'])
    # Create Routing Model.
    self.routing = pywrapcp.RoutingModel(self.manager)

    self.search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    self.transit_callback_index, self.demand_callback_index = (None, None)
    self.register_transit_callback()
    self.register_demand_callback()
    self.routing.AddDimensionWithVehicleCapacity(
        self.demand_callback_index,
        0,  # null capacity slack
        self.data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Setting first solution heuristic.
    self.search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES)

    self.shortest_path, self.routes = ([], [])
    
  def register_transit_callback(self):

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.data['distance_matrix'][from_node][to_node]        
    
    self.transit_callback_index = self.routing.RegisterTransitCallback(distance_callback)
    # Define cost of each arc.
    self.routing.SetArcCostEvaluatorOfAllVehicles(self.transit_callback_index)
    return 

  def register_demand_callback(self):

    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        return self.data['demands'][from_node]

    self.demand_callback_index = self.routing.RegisterUnaryTransitCallback(demand_callback)
    return
        
  def compute_route(self):
    assignment = self.routing.SolveWithParameters(self.search_parameters)

    total_distance = 0
    total_load = 0
    self.routes = []
    for vehicle_id in range(self.data['num_vehicles']):
        index = self.routing.Start(vehicle_id)
        route_distance = 0
        route_load = 0
        nodes = []
        route = []
        while not self.routing.IsEnd(index):
            nodes.append(index)
            node_index = self.manager.IndexToNode(index)
            route_load += self.data['demands'][node_index]
            previous_index = index
            index = assignment.Value(self.routing.NextVar(index))
            dist = self.routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)/100000
            route.append(node_index)
            route_distance += dist
        route.append(self.manager.IndexToNode(index))
        if route_distance>0:
          self.routes.append(route)
        total_distance += route_distance
        total_load += route_load
    self.shortest_path = total_distance

  def get_shortest_path(self):
    if(not self.routes):
        self.compute_route()
    return self.routes

  def get_shortest_path_length(self):
    if(not self.routes):
        self.compute_route()
    return self.shortest_path
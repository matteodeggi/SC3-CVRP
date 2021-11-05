import time
import pandas as pd
from datetime import timedelta
from haversine import haversine
from Shift import Shift

'''
Classe che astrae la simulazione di un intero periodo.
'''
class Simulation:
  def __init__(self, depot, config, window_size=6, max_size=175, filter_function = None, filter_kwargs={}):
    self.depot = depot
    self.cluster_class = config['cluster_class']
    self.cluster_kwargs = config['cluster_kwargs']
    self.graph_class = config['graph_class']
    self.graph_kwargs = config['graph_kwargs']
    self.filter_function = filter_function
    self.window_size = window_size
    self.max_size = max_size
    self.filter_kwargs = filter_kwargs
    self.distances = []
    self.routes = []
    self.timetables = []

  def get_score(self):
    return sum(self.distances)

  def get_distances(self):
    return self.distances
  
  def get_results(self):
    return self.routes

  def get_timetables(self, speed=30, emp_time=60):
    return self.timetables

  def to_csv(self, file):
    with open(file, 'w') as f:
        csv_output = []
        for w in range(len(self.routes)):
          for v in range(len(self.routes[w])):
            for o in range(len(self.routes[w][v])):
              if(self.timetables):
                csv_output.append(f'{self.routes[w][v][o][0]},{self.routes[w][v][o][1]},{o+1},{v+1},{w+1}, {self.timetables[w][v][o]}\n')
              else:
                csv_output.append(f'{self.routes[w][v][o][0]},{self.routes[w][v][o][1]},{o+1},{v+1},{w+1}\n')
        if(self.timetables):
          f.write('lat,long,order,vehicle,window,timetable\n')
        else:
          f.write('lat,long,order,vehicle,window\n')
        f.writelines(csv_output)

  '''
  Metodo per la computazione della simulazione.
  La simulazione tiene conto del vincolo per il quale ogni cestino deve essere svuotato almeno una volta al giorno.
  Tutti i cestini non svuotati durante la giornata vengono aggiunti all'ultimo turno.
  Restituisce i percorsi trovati per ogni turno.
  '''
  def compute_simulation(self, data, start_date=None, end_date=None, speed=None, emp_time=None, debug=False):
    i = 0
    start_date = data.index[0] if start_date is None else pd.to_datetime(start_date)
    end_date = data.index[-1] if end_date is None else pd.to_datetime(end_date)

    current_start_date = start_date
    current_end_date = start_date + timedelta(hours=self.window_size)-timedelta(seconds=1)

    # seriali di tutti i cestini
    bin_serials = data.bin_serial.unique()
    all_bins = data.drop_duplicates(subset='bin_serial')
    all_bins.loc[:, 'bin_level'] = 4
    
    # inizializzazione cestini non svuotati
    not_full = all_bins.copy()

    while(True):
      if(debug): print(str(current_start_date) + " - " + str(current_end_date))
      current_window = data[current_start_date : current_end_date]

      # verifica se il turno corrente è l'ultimo della giornata
      current_start_date += timedelta(hours=self.window_size)
      current_end_date += timedelta(hours=self.window_size)
      last_window = True if(current_end_date.day != (current_start_date-timedelta(seconds=1)).day) else False
      # aggiunta dei cestini non svuotati a quelli da svuotare, se ultimo turno della giornata
      current_window = pd.concat([current_window, not_full]) if last_window else current_window
      # cestini da svuotare
      current_window = current_window if self.filter_function is None else self.filter_function(current_window, **self.filter_kwargs)
      # aggiornamento lista dei cestini non svuotati
      not_full = all_bins.copy() if last_window else not_full[~not_full.bin_serial.isin(current_window['bin_serial'].unique())] 
      
      current_shift = Shift(current_window, self.depot, self.cluster_class, self.graph_class, start_date=current_start_date)
      
      # clustering
      start_time = time.time()
      clusters = current_shift.clusterize(columns=['latitude', 'longitude'], max_size=self.max_size, kwargs=self.cluster_kwargs)
      current_shift.build_graphs(columns=['latitude', 'longitude'], kwargs=self.graph_kwargs)
      
      # routing
      results = current_shift.get_routes()
      #results = [[self.depot, *result[1:-1], self.depot] for result in results]
      self.routes.append(results)
      self.distances.append(current_shift.get_distance())
      if(speed is not None or emp_time is not None): self.timetables.append(current_shift.get_timetables(haversine, speed, emp_time))
      if(debug): print(f"Distanza totale turno: {str(self.distances[-1])} km. Veicoli usati: {str(len(results))}.")
      if(debug): print(f"Tempo richiesto: {time.time() - start_time}s.")
      del current_shift

      if(current_end_date >= end_date): # Stop condition
        break

    return self.routes
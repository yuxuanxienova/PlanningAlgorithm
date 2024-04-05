from typing import Optional, TypeVar, Set, Mapping, Tuple, List
import queue
from pqdict import pqdict
import numpy as np
import matplotlib.pyplot as plt
import random
from shapely import STRtree
from shapely import box
from shapely.geometry.base import BaseGeometry
from shapely import LineString
from shapely import LinearRing
from shapely import Point
from shapely import Polygon
from typing import Optional, TypeVar, Set, Mapping, Tuple, List
import queue
from pqdict import pqdict
from shapely import geometry as geo
from shapely import wkt
from shapely import ops
import numpy as np
import networkx as nx
import pickle

if __name__ == "__main__":
    G = nx.path_graph(1769)
    fh = open("test.adjlist", "wb")
    nx.write_adjlist(G, fh)
    print(nx.shortest_path(G, source=0, target=1770))

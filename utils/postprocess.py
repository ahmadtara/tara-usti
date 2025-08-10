
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
import numpy as np
from skimage.morphology import skeletonize
import networkx as nx

def postprocess_polygons(polys, min_area=50.0, simplify_tol=1.0):
    out = []
    for p in polys:
        try:
            if p.area < min_area: 
                continue
            # simplify
            p2 = p.simplify(simplify_tol)
            # for buildings, approximate by minimum rotated rectangle to make them boxy
            rect = p2.minimum_rotated_rectangle
            out.append(rect)
        except Exception:
            continue
    return out

def extract_centerlines_from_mask(mask):
    # mask: 2D numpy bool array
    skel = skeletonize(mask.astype(bool))
    # convert skeleton pixels to graph and extract lines (simple approach)
    points = list(zip(*skel.nonzero()))
    # build adjacency by 8-neighborhood
    G = nx.Graph()
    for r,c in points:
        G.add_node((r,c))
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for r,c in points:
        for dr,dc in nbrs:
            nb = (r+dr, c+dc)
            if nb in G:
                G.add_edge((r,c), nb)
    # extract paths by finding nodes with degree !=2 as endpoints
    paths = []
    visited = set()
    for n in G.nodes():
        if G.degree(n) != 2:
            for nbr in G.neighbors(n):
                if (n,nbr) in visited or (nbr,n) in visited: continue
                path = [n, nbr]
                visited.add((n,nbr))
                cur = nbr
                prev = n
                while True:
                    deg = G.degree(cur)
                    if deg == 1:
                        break
                    nexts = [x for x in G.neighbors(cur) if x!=prev]
                    if not nexts: break
                    nxt = nexts[0]
                    if (cur,nxt) in visited: break
                    path.append(nxt)
                    visited.add((cur,nxt))
                    prev, cur = cur, nxt
                paths.append(path)
    # convert pixel coords (r,c) to x,y with c,x order
    lines = []
    for path in paths:
        coords = [(float(c), float(r)) for r,c in path]
        lines.append(LineString(coords))
    return lines

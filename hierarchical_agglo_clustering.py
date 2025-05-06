'''import numpy as np
import matplotlib.pyplot as plt

data=np.array([
    [10,10],
    [80,70],
    [70,70],
    [20,30]
])

clusters=[[i]for i in range (len(data))]
history=[]

def euclidean(p1,p2):
    return np.linalg.norm(p1-p2)

def cluster_distance(c1,c2):
    return min(euclidean(data[i],data[j])for i in c1 for j in c2)
while len(clusters)>1:
    min_dist=float('inf')
    to_merge=(0,1)
    
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            dist= cluster_distance(clusters[i],clusters[j],data)
            if dist< min_dist:
                min_dist= dist
                to_merge=(i,j)
                
    i,j=to_merge
    new_cluster=clusters[i]+clusters[j]
    history.append((clusters[i],clusters[j], min_dist))
    #print("mergeing clusters", clusters[i],"and",clusters[j],"with distance",round(min_dist,2))
    clusters.append(new_cluster)
    
    clusters.pop(max(i,j))
    clusters.pop(min(i,j))
   # clusters.append(new_cluster) 
print("\n final cluster(all points merged):")
print(clusters[0])     '''

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Sample data
data = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [5, 6],
    [8, 9],
    [9, 10]
])

# Step 2: Initial clusters = each point is its own cluster
clusters = [[i] for i in range(len(data))]
history = []

# Step 3: Compute distance between clusters
def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def cluster_distance(c1, c2, data):
    # Single linkage: min dist between any two points
    return min(euclidean(data[i], data[j]) for i in c1 for j in c2)

# Step 4: Agglomerative clustering loop
while len(clusters) > 1:
    min_dist = float('inf')
    to_merge = (0, 1)

    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            dist = cluster_distance(clusters[i], clusters[j], data)
            if dist < min_dist:
                min_dist = dist
                to_merge = (i, j)

    # Merge two closest clusters
    i, j = to_merge
    new_cluster = clusters[i] + clusters[j]
    history.append((clusters[i], clusters[j], min_dist))  # store for dendrogram
    clusters.append(new_cluster)

    # Remove old clusters
    clusters.pop(max(i, j))
    clusters.pop(min(i, j))

# Step 5: Dendrogram plotting manually
from scipy.cluster.hierarchy import dendrogram

# convert `history` to linkage matrix format
def make_linkage(history, n_points):
    cluster_map = {tuple([i]): i for i in range(n_points)}
    Z = []
    cluster_id = n_points
    for a, b, dist in history:
        id1 = cluster_map[tuple(sorted(a))]
        id2 = cluster_map[tuple(sorted(b))]
        size = len(a) + len(b)
        Z.append([id1, id2, dist, size])
        cluster_map[tuple(sorted(a + b))] = cluster_id
        cluster_id += 1
    return np.array(Z)

Z = make_linkage(history, len(data))

# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z)
plt.title("Dendrogram (From Scratch Linkage)")
plt.xlabel("Data Point Index")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

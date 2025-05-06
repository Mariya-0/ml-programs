import numpy as np
data=np.array([
    [10,10],
    [80,70],
    [70,70],
    [20,30]
])

clusters=[[i]for i in range (len(data))]

def euclidean(p1,p2):
    return np.linalg.norm(p1-p2)

def cluster_distance(c1,c2):
    return min(euclidean(data[i],data[j])for i in c1 for j in c2)
while len(clusters)>1:
    min_dist=float('inf')
    to_merge=(0,1)
    
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            dist= cluster_distance(clusters[i],clusters[j])
            if dist< min_dist:
                min_dist= dist
                to_merge=(i,j)
                
    i,j=to_merge
    new_cluster=clusters[i]+clusters[j]
    print("mergeing clusters", clusters[i],"and",clusters[j],"with distance",round(min_dist,2))
    
    clusters.pop(max(i,j))
    clusters.pop(min(i,j))
    clusters.append(new_cluster) 
print("\n final cluster(all points merged):")
print(clusters[0])                   
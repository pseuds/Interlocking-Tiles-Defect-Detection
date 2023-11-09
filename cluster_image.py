import sys
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class KMeans:

    def __init__(self, data, k, centroids=None):
        self.data = data
        if k <=0 or k > 8: 
            print('k must be within [1,8]')
            raise ValueError
        else: self.k = k
        self.centroids = centroids
        self.plotcolours = ["red", "blue", "green", "cyan", "magenta", "yellow", (255/255, 165/255, 0), (160/255, 32/255, 240/255)]

    # initialise random centroids, can be from anywhere between 0 to 255 for all three axes
    def init_centroids_randgen(self, random_state=None):
        if random_state != None:
            random.seed(random_state)
        self.centroids = []
        for _ in range(self.k):
            self.centroids.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))

    # initialise random centroids, choosing randomly from dataset
    def init_centroids_randpick(self, random_state=None):
        if random_state != None:
            random.seed(random_state)
        self.centroids = []
        chosen_i = random.sample([i for i in range(self.data.shape[0])], self.k)
        for i in chosen_i:
            self.centroids.append((self.data[i][0],self.data[i][1],self.data[i][2]))
    
    # initialise random centroids, choosing randomly from dataset but favours those far apart from one another
    def init_centroids_opt(self, random_state=None):
        if random_state != None:
            random.seed(random_state)
        self.centroids = []
        chosen_i = random.choice([i for i in range(self.data.shape[0])])
        self.centroids.append((self.data[chosen_i][0],self.data[chosen_i][1],self.data[chosen_i][2]))
        for _ in range(self.k - 1):
            dist_ls = np.zeros((self.data.shape[0],1), dtype=np.float64)
            dist_p = np.zeros((self.data.shape[0],), dtype=np.float64)
            for c in self.centroids:
                # 10**5 to counter overflow error
                dist_ls = self.compute_all_distances(self.data, c) #/ 10**5
                dist_sum = np.sum(dist_ls)
                dist_p += (dist_ls / dist_sum)
            dist_p /= len(self.centroids)
            new_c_i = np.random.choice([i for i in range(self.data.shape[0])], p=dist_p)
            self.centroids.append((self.data[new_c_i][0],self.data[new_c_i][1],self.data[new_c_i][2]))
        
    # Assigns each instance to nearest centroid
    def assign(self):
        self.assignment = {c:[] for c in self.centroids}
        for datapoint in self.data: 
            nearest_c = None
            min_distance = sys.maxsize
            for c in self.centroids:
                distance = self.compute_distance(datapoint,c)
                if distance < min_distance or nearest_c == None:
                    min_distance = distance
                    nearest_c = c
            self.assignment[nearest_c].append(datapoint)
    
    def clear_bad_centroids(self):
        new_assignment = {}
        new_centroids = []
        for c, dpls in self.assignment.items():
            if len(dpls) != 0:
                new_assignment[c] = dpls
                new_centroids.append(c)
        self.assignment = new_assignment
        self.centroids = new_centroids

    # Computes distance between p1 and p2, using squared Euclidean distance
    def compute_distance(self, p1, p2): 
        p1 = (p1[0] / 10**5, p1[1] / 10**5, p1[2] / 10**5)
        p2 = (p2[0] / 10**5, p2[1] / 10**5, p2[2] / 10**5)
        distance = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
        return distance 

    # Computes all distances from list of point x's and centroid
    def compute_all_distances(self, x_ls, c):
        dist_ls = []
        for point in x_ls:
            dist_ls.append(self.compute_distance(point, c))
        return np.array(dist_ls)

    def run(self, random_state=None, init=True):
        # 0. set up initial centroid values
        if init:
            self.init_centroids_opt(random_state=random_state)
        # 1. assign each instance to the class with nearest centroid
        self.assign()
        # 2. recompute centroids of each class until assignments and centroids stop changing
        while True:
            prev_assign = deepcopy(self.assignment)
            # recompute centroids
            for idx, datapoints_ls in enumerate(prev_assign.values()):
                new_c = (np.mean([d[0] for d in datapoints_ls]), np.mean([d[1] for d in datapoints_ls]), np.mean([d[2] for d in datapoints_ls]))
                self.centroids[idx] = (round(new_c[0],3),round(new_c[1],3),round(new_c[2],3))
            self.assign()
            self.clear_bad_centroids()
            different = False
            for latest_ls, old_ls in zip(self.assignment.values(), prev_assign.values()):
                if len(latest_ls) != len(old_ls):
                    different = True
                    break
                else:
                    for dp1, dp2 in zip(latest_ls,old_ls):
                        if dp1[0]!=dp2[0] or dp1[1]!=dp2[1] or dp1[2]!=dp2[2]:
                            different = True
                            break
            if not different:
                break

    def visualise(self, img):
        arr3d = np.asarray(img)
        og_dim = arr3d.shape
        final_arr3d = np.zeros(og_dim, dtype=np.uint8)
        for h in range(og_dim[0]):
            for w in range(og_dim[1]):
                dp = (arr3d[h][w][0], arr3d[h][w][1], arr3d[h][w][2])
                nearest_c = None
                min_dist = sys.maxsize
                # find nearest centroid
                for c in self.centroids:
                    distance = self.compute_distance(dp,c)
                    if distance < min_dist or nearest_c == None:
                        min_dist = distance
                        nearest_c = c
                new_colour = (round(nearest_c[0]),round(nearest_c[1]),round(nearest_c[2]))
                final_arr3d[h][w][0],final_arr3d[h][w][1],final_arr3d[h][w][2] = new_colour[0],new_colour[1],new_colour[2]
        final_img = Image.fromarray(final_arr3d, "RGB")
        return final_img

    # plot datapoints. datapoints belonging to different clusters are in different colours.
    def plot(self, sample_size=None):
        if sample_size == None: sample_size = self.data.shape[0]
        chosen_i = random.sample([i for i in range(self.data.shape[0])], sample_size)

        fig = plt.figure(figsize=(12,6))
        thisplot = fig.add_subplot(111, projection='3d')
        l_ls = []
        l_names = []

        for idx, dd in enumerate(self.assignment.items()):
            c, dp_ls = dd[0],dd[1]
            thisplot.scatter(c[0],c[1],c[2], color='black',marker='x',zorder=self.data.shape[0]+1+idx)
            x_points = [i[0] for i in dp_ls]
            y_points = [i[1] for i in dp_ls]
            z_points = [i[2] for i in dp_ls]
            l = thisplot.scatter(x_points, y_points, z_points, s=3, color=(c[0]/255,c[1]/255,c[2]/255), edgecolors='face', zorder=1)
            l_ls.append(l)
            name = f"({round(c[0])},{round(c[1])},{round(c[2])}): {len(dp_ls)}"
            l_names.append(name)
        thisplot.set_xlabel('R')
        thisplot.set_ylabel('G')
        thisplot.set_zlabel('B')
        plt.legend((label for label in l_ls), (n for n in l_names),markerscale=3,loc='upper right',fontsize=8)

        plt.show()

    # plot elbow method
    def plot_elbow(self):
        x_axis = [i for i in range(1,9)]
        y_axis = []
        for k in x_axis:
            distortion = 0
            dist_ls = []
            self.k = k
            self.run()
            for c, dp_ls in self.assignment.items():
                dist_ls = self.compute_all_distances(dp_ls, c)
                distortion += np.sum(dist_ls) / 10**3    
            distortion = (distortion * 10**3) / self.data.shape[0]
            y_axis.append(distortion)       
        plt.plot(x_axis, y_axis, marker='o')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('Elbow Graph')
        plt.show()   
        
    # for fun. ignore this
    def plot_by_colour(self,sample_size=None):
        if sample_size == None: sample_size = self.data.shape[0]
        chosen_i = random.sample([i for i in range(self.data.shape[0])], sample_size)

        thisplot = plt.axes(projection='3d')
        for i in chosen_i:
            thisplot.scatter(self.data[i][0],self.data[i][1],self.data[i][2], color=(self.data[i][0]/255,self.data[i][1]/255,self.data[i][2]/255))
        # x_points = [self.data[i][0] for i in chosen_i]
        # y_points = [self.data[i][1] for i in chosen_i]
        # z_points = [self.data[i][2] for i in chosen_i]
        # thisplot.scatter(x_points, y_points, z_points, color='red', edgecolor='black')
            

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 cluster_image.py [source image] [k value]")
        sys.exit()

    source_path = sys.argv[1]
    k_input = sys.argv[2]
    try: k_value = int(k_input)
    except: 
        print("The k-value must be an integer.")
        sys.exit()
    
    # img = Image.open(source_path)
    try: img = Image.open(source_path)
    except: 
        print("Error getting image.")
        sys.exit()

    arr3d = np.asarray(img)
    # for images with RBGA channnel
    if arr3d.shape[2] == 4:
        arr3d = arr3d[:,:,:-1]
    # print(arr3d)
    # print(arr3d.shape)
    arr = arr3d.reshape((arr3d.shape[0]*arr3d.shape[1], 3))
    km = KMeans(arr, k_value)
    print("Clustering...")
    km.run(init=True)
    print("Visualising...")
    final_img = km.visualise(img)
    final_img.save("result.png")
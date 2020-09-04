def compound_interest(principle:float, rate:float, peroids:int): 
    # Calculates compound interest  
    total_return = principle * (pow((1 + rate / 100), peroids)) 
    print("Total Interest $:", round(total_return, 2))
    print("Anualized Peroid %", round(total_return / principle, 1) * 100)

# compount daily for 1 year (market days)
compound_interest(principle=100000, rate=.5, peroids=250)


from scipy.cluster.hierarchy import linkage, is_valid_linkage, fcluster
from scipy.spatial.distance import pdist

## Load dataset
X = np.load("dataset.npy")

## Hierarchical clustering
dists = pdist(X)
Z = linkage(dists, method='centroid', metric='euclidean')

print(is_valid_linkage(Z))

## Now let's say we want the flat cluster assignement with 10 clusters.
#  If cut_tree() was working we would do
from scipy.cluster.hierarchy import cut_tree
cut = cut_tree(Z, 10)

clust = fcluster(Z, k, criterion='maxclust')


### from scipy.cluster.hierarchy import cut_tree
from scipy import cluster
np.random.seed(23)
X = np.random.randn(50, 4)
Z = cluster.hierarchy.ward(X, )
cutree = cluster.hierarchy.cut_tree(Z, n_clusters=[5, 10])
cutree[:10]


# good = [0, 1, 3, 4, 8, 9, 11, 14, 23, 25, 27, 28, 30, 34, 36]
# listed_bad = [2, 5, 7, 10, 13, 15, 16, 17, 18, 19, 20, 21, 22, 29, 33, 37, 52, 53]
# confirmed_bad = [2, 7, 10, 12, 13, 15, 16, 17, 20, 22, 38, 52, 53]
# neverseen_bad = [5, 18, 19, 21, 29, 33]
# listed_blank = [6, 17, 18, 19, 24, 26, 32, 35, 39-51, 54, 55, 56, 59]
# after_hours = [12]
# odd_lot = [37]
# neutral = [41]


def apply_fft(series, components=[3, 6, 9, 100]):
    
    close_fft = np.fft.fft(series)
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    plt.figure(figsize=(14, 7), dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())

    for num_ in components:
        fft_list_m10 = np.copy(fft_list) 
        fft_list_m10[num_:-num_] = 0
        plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))

    plt.plot(series, label='Real')
    plt.xlabel('Time')
    plt.ylabel('USD')
    plt.title('Stock trades & Fourier transforms')
    plt.legend()
    plt.show()

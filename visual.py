import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE


def pca(states, labels, n_components):
    assert n_components==2 or n_components==3, 'Wrong number of components'

    print('PCA')
    pca = PCA(n_components=n_components)

    print('Fitting & transforming')
    transformed_states = pca.fit_transform(states)

    print('Visual')
    plt.clf()
    plt.cla()
    colors = np.choose(labels, ['blue', 'red', 'black', 'green', 'pink', 'yellow', 'brown', 'magenta', 'cyan', 'orange'])
    if n_components == 2:
        plt.scatter(transformed_states[:, 0], transformed_states[:, 1], c=colors)
    elif n_components == 3:
        fig = plt.figure(1, figsize=(4, 3))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        ax.scatter(transformed_states[:, 0], transformed_states[:, 1], transformed_states[:, 2], c=colors)

    plt.show()


def tsne(states, labels, n_components):
    assert n_components==2 or n_components==3, 'Wrong number of components'

    print('T-SNE')
    tsne = TSNE(n_components=n_components)

    print('Fitting & transforming')
    transformed_states = tsne.fit_transform(states)

    print('Visual')
    plt.clf()
    plt.cla()
    colors = np.choose(labels, ['blue', 'red', 'black', 'green', 'pink', 'yellow', 'brown', 'magenta', 'cyan', 'orange'])
    if n_components == 2:
        plt.scatter(transformed_states[:, 0], transformed_states[:, 1], c=colors)
    elif n_components == 3:
        fig = plt.figure(1, figsize=(4, 3))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        ax.scatter(transformed_states[:, 0], transformed_states[:, 1], transformed_states[:, 2], c=colors)

    plt.show()


if __name__ == '__main__':
    print('Loading')
    df = pickle.load(open('/home/petrbel/Desktop/states.pkl', 'rb'))
    pca(states=list(df['states']), labels=list(df['labels']), n_components=2)
    pca(states=list(df['states']), labels=list(df['labels']), n_components=3)
    tsne(states=list(df['states']), labels=list(df['labels']), n_components=2)
    tsne(states=list(df['states']), labels=list(df['labels']), n_components=3)
    print('Finished')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import joblib


class FindSimilarSong:

    def __init__(self,kmean_model="kmean_model.pkl", data="data/data_appended.csv", ids_clusters="ids.txt", taille_playlist=60):
        try:
            self.kmean_model = joblib.load(kmean_model)
        except:
            raise ValueError("The model is not found")

        try:
            self.data = pd.read_csv(data)
        except:
            raise ValueError("The reference data is not found")

        columns = ['valence', 'year', 'acousticness', 'artists', 'danceability', 'duration_ms', 'energy', 'explicit',
                   'id', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'name', 'popularity',
                   'release_date', 'speechiness', 'tempo','genre']
        self.data = shuffle(self.data, random_state=42)
        self.df = pd.DataFrame(self.data, columns=columns)

        drop_columns = ['name', 'artists', 'release_date', 'year','genre']

        scaler = MinMaxScaler()
        self.df['year'] = scaler.fit_transform(self.df[['year']])
        self.df['duration_ms'] = scaler.fit_transform(self.df[['duration_ms']])
        self.df['key'] = scaler.fit_transform(self.df[['key']])
        self.df['loudness'] = scaler.fit_transform(self.df[['loudness']])
        self.df['popularity'] = scaler.fit_transform(self.df[['popularity']])
        self.df['tempo'] = scaler.fit_transform(self.df[['tempo']])

        self.df = self.df.drop(columns=drop_columns, errors='ignore')


        self.clusters = []
        self.ids = []
        with open(ids_clusters, 'r') as file:
            clusters = file.readlines()
            for line in clusters:
                tab = line.split(" ")
                self.ids.append(tab[0])
                self.clusters.append(int(tab[1]))

        self.df["cluster"] = self.clusters

        self.taille_playlist = taille_playlist



    def _get_similar_song(self, song_id, nb_voisin):
        try:
            song = self.df[self.data["id"] == song_id]
            song = song.drop(columns=["cluster","id"],errors='ignore')
        except:
            raise ValueError("The song is not found")

        elements_in_cluster = []

        ids_in_cluster = []

        song_cluster = self.kmean_model.predict(song)[0]
        for index in range(len(self.clusters)):
            if self.clusters[index] == song_cluster:
                elements_in_cluster.append(self.ids[index])
                ids_in_cluster.append(self.ids[index])


        cluster_songs = self.df[self.df["cluster"] == song_cluster]


        # Calculate the distances between the song and all other songs in the same cluster
        distances = np.linalg.norm(cluster_songs.drop(columns=["cluster","id"],errors="ignore").values - song.values, axis=1)
        new_tab_dist = []
        for i in range(len(distances)):
            if distances[i] != 0:
                new_tab_dist.append((ids_in_cluster[i],distances[i]))

        sorted_distance = sorted(new_tab_dist, key=lambda x: x[1])
        # Get the indices of the n_neighbors closest songs
        if len(sorted_distance) < nb_voisin:
            closest_indices = [x[0] for x in sorted_distance]
        else:
            closest_indices = [x[0] for x in sorted_distance[:nb_voisin]]

        return closest_indices


    def find_songs_for_playlist(self, list_song_id):
        clusters_song = []
        nb_song_by_cluster = self.taille_playlist // len(list_song_id)
        for song_id in list_song_id:
            songs = self._get_similar_song(song_id, nb_song_by_cluster)
            clusters_song.append(songs)

        ids_for_new_playlist = []
        for songs in clusters_song:
            ids_for_new_playlist.extend(songs)
        return ids_for_new_playlist

f = FindSimilarSong()
f.find_songs_for_playlist(["0XGcXc6VkB5dx6RNWxV0rF","2kPA4clZYYrB9Cb7Uzh5YG"])
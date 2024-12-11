import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from requests import delete
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import false_discovery_control
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
            self.data = shuffle(self.data, random_state=42)
        except:
            raise ValueError("The reference data is not found")

        columns = ['valence', 'year', 'acousticness', 'artists', 'danceability', 'duration_ms', 'energy', 'explicit',
                   'id', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'name', 'popularity',
                   'release_date', 'speechiness', 'tempo','genre']
        drop_columns = ['name', 'artists', 'release_date', 'year','genre']

        self.df = pd.DataFrame(self.data, columns=columns)
        scaler = MinMaxScaler()
        self.df['year'] = scaler.fit_transform(self.df[['year']])
        self.df['duration_ms'] = scaler.fit_transform(self.df[['duration_ms']])
        self.df['key'] = scaler.fit_transform(self.df[['key']])
        self.df['loudness'] = scaler.fit_transform(self.df[['loudness']])
        self.df['popularity'] = scaler.fit_transform(self.df[['popularity']])
        self.df['tempo'] = scaler.fit_transform(self.df[['tempo']])
        self.copied_df = self.df.copy()

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


    def _check_unicity(self, ids_for_playlist):
        state_duplicated = True
        # for i in range(len(ids_for_playlist)):
        #     for j in range(i+1,len(ids_for_playlist)):
        #         # if ids_for_playlist[i] == ids_for_playlist[j]:
        #         #     state_duplicated = False
        #         if self.copied_df[self.copied_df["id"] == ids_for_playlist[i]]["name"].values[0] == self.copied_df[self.copied_df["id"] == ids_for_playlist[j]]["name"].values[0] and self.copied_df[self.copied_df["id"] == ids_for_playlist[i]]["artists"].values[0] == self.copied_df[self.copied_df["id"] == ids_for_playlist[j]]["artists"].values[0]:
        #             if  ids_for_playlist[j] not in duplicated_ids:
        #                 duplicated_ids.append(ids_for_playlist[j])
        #                 state_duplicated = False

        filtered_df = self.copied_df[self.copied_df["id"].isin(ids_for_playlist)]
        duplicates = filtered_df.duplicated(subset=["name", "artists"], keep='first')
        duplicated_ids = filtered_df[duplicates]["id"].tolist()
        if len(duplicated_ids) > 0:
            state_duplicated = False
        return state_duplicated, duplicated_ids

    def _delete_and_replace(self,ids_for_playlist, reference_ids, duplicated_ids):
        nb_delete = 0
        ids_to_delete = []
        # for i in range(len(ids_for_playlist)):
        #     for j in range(i+1,len(ids_for_playlist)):
        #         # if j-nb_delete <len(ids_for_playlist) and ids_for_playlist[i] == ids_for_playlist[j]:
        #         #     ids_to_delete.append(j)
        #         if self.copied_df[self.copied_df["id"] == ids_for_playlist[i]]["name"].values[0] == self.copied_df[self.copied_df["id"] == ids_for_playlist[j]]["name"].values[0] and self.copied_df[self.copied_df["id"] == ids_for_playlist[i]]["artists"].values[0] == self.copied_df[self.copied_df["id"] == ids_for_playlist[j]]["artists"].values[0]:
        #             if  ids_for_playlist[j] in duplicated_ids:
        #                 ids_to_delete.append(j)

        filtered_df = self.copied_df[self.copied_df["id"].isin(ids_for_playlist)]
        duplicates = filtered_df.duplicated(subset=["name", "artists"], keep='first')
        duplicated_ids = filtered_df[duplicates]["id"].tolist()

        # for i in range(len(ids_to_delete)):
        #     if ids_to_delete.count(ids_to_delete[i]) > 1:
        #         ids_to_delete.remove(ids_to_delete[i])
        #
        # for i in ids_to_delete:
        #     if i in ids_for_playlist:
        #         ids_for_playlist.remove(i)
        #         nb_delete += 1

        for id in duplicated_ids:
            if id in ids_for_playlist:
                ids_for_playlist.remove(id)
                nb_delete += 1


        ignore_ids = reference_ids + duplicated_ids
        for i in range(nb_delete):
            id_playslist , ignore_ids= self._get_similar_song(reference_ids[i],1,ignore_ids, ids_for_playlist)
            ids_for_playlist.append(id_playslist[0])

        return ids_for_playlist



    def _get_similar_song(self, song_id, nb_voisin,ignore_ids, song_already_selected = []):
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
                if ids_in_cluster[i] != song_id:
                    new_tab_dist.append((ids_in_cluster[i],distances[i]))

        sorted_distance = sorted(new_tab_dist, key=lambda x: x[1])
        # Get the indices of the n_neighbors closest songs
        if len(song_already_selected) == 0:
            if len(sorted_distance) < nb_voisin:
                closest_indices = [x[0] for x in sorted_distance if x[0] not in ignore_ids]
            else:
                closest_indices = [x[0] for x in sorted_distance[:nb_voisin] if x[0] not in ignore_ids]
                if len(closest_indices) < nb_voisin:
                    for i in range(len(sorted_distance)):
                        if sorted_distance[i][0] not in ignore_ids and sorted_distance[i][0] not in closest_indices:
                            closest_indices.append(sorted_distance[i][0])
                        if len(closest_indices) == nb_voisin:
                            break
        else:
            closest_indices = []
            for i in range(len(sorted_distance)):
                if sorted_distance[i][0] not in song_already_selected and sorted_distance[i][0] not in ignore_ids:
                    closest_indices.append(sorted_distance[i][0])
                if len(closest_indices) == nb_voisin:
                    break

        ignore_ids += closest_indices

        return closest_indices, ignore_ids


    def find_songs_for_playlist(self, list_song_id):
        clusters_song = []
        nb_song_by_cluster = self.taille_playlist // len(list_song_id)
        nb_song = [nb_song_by_cluster for _ in range(len(list_song_id))]
        index = 0
        while sum(nb_song) < self.taille_playlist and index < len(nb_song):
            nb_song[index] += 1
            index += 1

        ignore_ids = list_song_id
        for i in range(len(list_song_id)):
            songs, ignore_ids = self._get_similar_song(list_song_id[i], nb_song[i], ignore_ids)
            clusters_song.append(songs)

        ids_for_new_playlist = []
        for songs in clusters_song:
            ids_for_new_playlist.extend(songs)

        is_unique, duplicated_ids = self._check_unicity(ids_for_new_playlist)
        if not is_unique:
            ids_for_new_playlist = self._delete_and_replace(ids_for_new_playlist,list_song_id,duplicated_ids)


        return ids_for_new_playlist

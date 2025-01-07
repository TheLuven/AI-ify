import numpy as np
import pandas as pd
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


    def _check_unicity(self, ids_for_playlist):
        for i in range(len(ids_for_playlist)):
            for j in range(i+1,len(ids_for_playlist)):
                if ids_for_playlist[i] == ids_for_playlist[j]:
                    return False
        return True

    def _delete_and_replace(self,ids_for_playlist, reference_ids):
        nb_delete = 0
        for i in range(len(ids_for_playlist)):
            for j in range(i+1,len(ids_for_playlist)):
                if ids_for_playlist[i] == ids_for_playlist[j]:
                    ids_for_playlist.pop(i)
                    nb_delete += 1

        for i in range(nb_delete):
            ids_for_playlist += self._get_similar_song(reference_ids[i],1,reference_ids, ids_for_playlist)

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

        return closest_indices


    def find_songs_for_playlist(self, list_song_id):
        clusters_song = []
        nb_song_by_cluster = self.taille_playlist // len(list_song_id)
        nb_song = [nb_song_by_cluster for _ in range(len(list_song_id))]
        index = 0
        while sum(nb_song) < self.taille_playlist and index < len(nb_song):
            nb_song[index] += 1
            index += 1

        for i in range(len(list_song_id)):
            songs = self._get_similar_song(list_song_id[i], nb_song[i], list_song_id)
            clusters_song.append(songs)

        ids_for_new_playlist = []
        for songs in clusters_song:
            ids_for_new_playlist.extend(songs)

        if not self._check_unicity(ids_for_new_playlist):
            ids_for_new_playlist = self._delete_and_replace(ids_for_new_playlist,list_song_id)


        return ids_for_new_playlist

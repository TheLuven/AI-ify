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
from collections import Counter


class FindSimilarSong:

    def __init__(self,kmean_model="kmean_model.pkl", data="data/data_appended.csv", taille_playlist=60):
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

        dataframe = self.df.copy()
        dataframe = dataframe.drop(columns=["id"], errors='ignore')

        self.clusters = self.kmean_model.predict(dataframe)
        self.ids = self.df["id"].tolist()

        self.df["cluster"] = self.clusters

        self.taille_playlist = taille_playlist


    def _check_unicity(self, ids_for_playlist, top50):
        state_duplicated = True

        filtered_top_df = self.copied_df[self.copied_df["id"].isin(top50)]
        filtered_playlist = self.copied_df[self.copied_df["id"].isin(ids_for_playlist)]
        filtered_df = pd.concat([filtered_top_df, filtered_playlist])
        duplicates = filtered_df.duplicated(subset=["name", "artists"], keep=False)
        duplicated_ids_top = filtered_df[duplicates]["id"].tolist()

        filtered_df = self.copied_df[self.copied_df["id"].isin(ids_for_playlist)]
        duplicates = filtered_df.duplicated(subset=["name", "artists"], keep='first')
        duplicated_ids_name_artist = filtered_df[duplicates]["id"].tolist()

        duplicated_ids = duplicated_ids_top + duplicated_ids_name_artist
        if len(duplicated_ids) > 0:
            state_duplicated = False
        return state_duplicated, duplicated_ids


    def _getStats(self,songs,song_reference):
        with open("stats.txt", "a") as stats_file:
            stats_file.write(self.copied_df[self.copied_df["id"] == song_reference]["name"].values[0] + " " +
                             self.copied_df[self.copied_df["id"] == song_reference]["artists"].values[0] + "\n")

            for song in songs:
                stats_file.write("\t" + self.copied_df[self.copied_df["id"] == song]["name"].values[0] + " " + self.copied_df[self.copied_df["id"] == song]["artists"].values[0] + "\n")

            stats_file.write("\n")
            stats_file.close()


    def _delete_and_replace(self,ids_for_playlist, reference_ids, duplicated_ids,ignore_ids):
        nb_delete = 0

        for id in duplicated_ids:
            if id in ids_for_playlist:
                ids_for_playlist.remove(id)
                nb_delete += 1


        ignore_ids = ignore_ids + reference_ids + duplicated_ids
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

        filtered_top_df = self.copied_df[self.copied_df["id"].isin(list_song_id)]
        filtered_playlist = self.copied_df[self.copied_df["id"].isin(self.ids)]
        filtered_df = pd.concat([filtered_top_df, filtered_playlist])
        duplicates = filtered_df.duplicated(subset=["name", "artists"], keep=False)
        duplicated_ids_top = filtered_df[duplicates]["id"].tolist()

        ignore_ids = list_song_id + duplicated_ids_top
        for i in range(len(list_song_id)):
            songs, ignore_ids = self._get_similar_song(list_song_id[i], nb_song[i], ignore_ids)
            self._getStats(songs,list_song_id[i])
            clusters_song.append(songs)

        ids_for_new_playlist = []
        for songs in clusters_song:
            ids_for_new_playlist.extend(songs)

        is_unique, duplicated_ids = self._check_unicity(ids_for_new_playlist, list_song_id)
        if not is_unique:
            ids_for_new_playlist = self._delete_and_replace(ids_for_new_playlist,list_song_id,duplicated_ids,ignore_ids)


        return ids_for_new_playlist


    def find_songs_for_playlist_version2(self,top50):
        clusters_song = []

        for song_id in top50:
            try:
                song = self.df[self.data["id"] == song_id]
                song = song.drop(columns=["cluster", "id"], errors='ignore')
            except:
                raise ValueError("The song is not found")
            clusters_song.append(self.kmean_model.predict(song))

        clusters_frequences = {}
        for element, count in Counter(map(tuple, clusters_song)).items():
            clusters_frequences[str(element[0])] = count

        nb_song_by_cluster = {}
        for key in clusters_frequences.keys():
            nb_song_by_cluster[key] = round(self.taille_playlist * (clusters_frequences[key] / len(top50)))

        print(sum(nb_song_by_cluster.values()))
        sorted_clusters_frequences = dict(sorted(nb_song_by_cluster.items(), key=lambda item: item[1], reverse=True))
        print(sorted_clusters_frequences)

        filtered_top_df = self.copied_df[self.copied_df["id"].isin(top50)]
        filtered_playlist = self.copied_df[self.copied_df["id"].isin(self.ids)]
        filtered_df = pd.concat([filtered_top_df, filtered_playlist])
        duplicates = filtered_df.duplicated(subset=["name", "artists"], keep=False)
        duplicated_ids_top = filtered_df[duplicates]["id"].tolist()

        ignore_ids = top50 + duplicated_ids_top

        return self._get_song_according_to_cluster(sorted_clusters_frequences,ignore_ids,top50)




    def _get_song_according_to_cluster(self,sorted_clusters_frequences,ignore_ids,top50):
        ids_for_new_playlist = []
        for key in sorted_clusters_frequences.keys():
            nb_song = sorted_clusters_frequences[key]
            cluster_songs = self.df[self.df["cluster"] == int(key)]

            centroid = cluster_songs.drop(columns=["cluster", "id"], errors='ignore').mean()

            distances = np.linalg.norm(cluster_songs.drop(columns=["cluster", "id"], errors='ignore').values - centroid.values, axis=1)
            closest_songs = cluster_songs.iloc[np.argsort(distances)[:nb_song]]["id"].tolist()

            ids_for_new_playlist.extend(closest_songs)

        is_unique, duplicated_ids = self._check_unicity(ids_for_new_playlist, top50)
        ignore_ids = ignore_ids + duplicated_ids
        if not is_unique:
            print("n est pas unique")
             #ids_for_new_playlist = self._delete_and_replace(ids_for_new_playlist,top50,duplicated_ids,ignore_ids)
        return ids_for_new_playlist





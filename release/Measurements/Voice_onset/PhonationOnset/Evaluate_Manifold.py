from os import walk
from os.path import isfile, join
import numpy as np
import pandas as pd
import flammkuchen as fl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import librosa
from sklearn.preprocessing import MinMaxScaler
import os
import librosa.display
import umap
import cv2

"""
Code to make the t-SNE and UMAP plots
"""
# based on: https://medium.com/@LeonFedden/comparative-audio-analysis-with-wavenet-mfccs-umap-t-sne-and-pca-cb8237bfce2f

data_path =  r'D:\Uni\Masterarbeit\Daten\Annotated_subjects'

save_path = r'D:\Uni\Masterarbeit\Final_Results\Clustering'
def load_all(participant_selection):


    """
    loads the datafiles of each participant and saves every single event into a list
    Args:
        participant_selection None or list of Strings: If None, all subjects at "data_path" location are considered
            otherwise only participants which are contained in the list of stringt are considered

    Returns: List of Dicts

    """
    parent_folder = data_path
    participant_selection = participant_selection
    f = []
    for (dirpath, dirnames, filenames) in walk(parent_folder):
        f.extend(filenames)
        break

    split_f = []
    for file in f:
        split = file.split('_')
        split_f.append(split[0])

    new_f = []
    if participant_selection is not None:
        for participant in participant_selection:
            for file in f:
                if participant in file:
                    new_f.append(file)

        f = new_f
    c=0
    file_list = []
    for i, file_name in enumerate(f):
        list = (fl.load(parent_folder+'/'+file_name))
        for file in list:
            file_list.append(file)


    #make_manifold(np.asarray(features_list),df)
    return file_list


def obtain_features(files, resampling = None, n_mfcc=13, image=False):
    """
    calculates the features for each events acoustic signal and adds them to that events dict

    Args:
        files: list of dicts containing the data of each subject
        resampling None or int: if not none, the acoustic signal gets resampled by the frequency given in resampling

    Returns: list of dicts

    """

    for dict in files:
        if resampling == None:
            mfccs = librosa.feature.mfcc(dict['acoustic'], 40000, n_mfcc=n_mfcc)
        else:
            mfccs = librosa.feature.mfcc(librosa.resample(dict['acoustic'], 40000, resampling), resampling, n_mfcc=n_mfcc)


        if image == False:
            features = get_features(mfccs.T,n_mfcc)
        else:
            features = cv2.resize(mfccs, dsize=(64, 128), interpolation=cv2.INTER_CUBIC).flatten(order='C')

        dict['mfccs'] = mfccs
        dict['mfccs_features'] = features

    return files


def get_features(mfccs,n_mfcc):

    """
    calculates features accross the melspectogramm coefficients
    Args:
        mfccs: array containg the mel melspectogramm coefficients

    Returns: ndarray

    """
    stddev_features = np.std(mfccs, axis=0)

    # Get the mean
    mean_features = np.mean(mfccs, axis=0)

    # Get the average difference of the features
    average_difference_features = np.zeros((n_mfcc,))
    for i in range(0, len(mfccs) - 2, 2):
        average_difference_features += mfccs[i] - mfccs[i + 1]
    average_difference_features /= (len(mfccs) // 2)
    average_difference_features = np.array(average_difference_features)

    # Concatenate the features to a single feature vector
    concat_features_features = np.hstack((stddev_features, mean_features))
    concat_features_features = np.hstack((concat_features_features, average_difference_features))

    return concat_features_features


def get_scaled_tsne_embeddings(features, perplexity, iteration):

    """
    calculates the 2-D t-sne manifold
    Args:
        features: ndarray
        perplexity: int
        iteration: int

    Returns: ndarray

    """
    embedding = TSNE(n_components=2,
                     perplexity=perplexity,
                     n_iter=iteration).fit_transform(features)
    scaler = MinMaxScaler()
    scaler.fit(embedding)
    return scaler.transform(embedding)


def get_scaled_umap_embeddings(features, neighbour, distance):

    """
    calculates the umap mandifold
    Args:
        features: ndarray
        neighbour: int
        distance: int

    Returns: ndarray

    """
    embedding = umap.UMAP(n_neighbors=neighbour,
                          min_dist=distance,
                          metric='correlation').fit_transform(features)
    scaler = MinMaxScaler()
    scaler.fit(embedding)
    return scaler.transform(embedding)


def make_manifold(files, coloring_by, save_name, t_sne= True, cluster_by_parameter=False):
    """
    Make the plot of either the umap or t-sne manifold and colorizes the datapoints according to the vowel or onset types
    Args:
        files: list of dicts containing the data of each subject
        coloring_by, String: color the manifold datapoints by means of metadatainfo. E.g. vowels, onsettype
        save_name, String: name at which the plot data should be saved
        t_sne, Boolean: use T-SNE clustering when true, otherwise use UMAP

    Returns: None

    """

    #perplexities = [2, 5, 10, 20, 30, 50, 100]
    perplexities = [20, 30, 50]
    iterations = [250, 500, 1000, 2000, 5000]

    neighbours = [5, 10, 15, 30, 50]
    distances = [0.000, 0.001, 0.01, 0.1, 0.5]

    #neighbours = [5, 15]
    #distances = [0.01,0.1, 0.5]

    extracted_list = []

    for dict in files:
        if cluster_by_parameter == False:
            extracted_info = {'acoustic': dict['acoustic'], 'vowel': dict['meta_info']['vowel'],
                              'onset_type': dict['meta_info']['onset_type'], 'name': dict['file_name'],
                              'features': dict['mfccs_features']}
        else:
            feature_list = []
            for parameter in dict['parameters']:
                if dict['parameters'][parameter] != None and parameter != 'voc':
                #if dict['parameters'][parameter] != None:
                    feature_list.append(dict['parameters'][parameter]['time'])
                    feature_list.append(dict['parameters'][parameter]['relative_time'])
            extracted_info = {'acoustic': dict['acoustic'], 'vowel': dict['meta_info']['vowel'],
                              'onset_type': dict['meta_info']['onset_type'], 'name': dict['file_name'],
                              'features': feature_list}
        extracted_list.append(extracted_info)
    df = pd.DataFrame.from_dict(extracted_list)
        #features_list.append(dict['mfccs_features'])
    #df = df[df['vowel'].str.contains("a")].reset_index()
    if t_sne:
        variables1 = perplexities
        variables2 = iterations

    else:

        variables1 = neighbours
        variables2 = distances

    for variable1 in variables1:
        for varibale2 in variables2:
            if t_sne:
                manifold = get_scaled_tsne_embeddings(np.asarray([row[0] for row in df.filter(like='features').values]), variable1, varibale2)

            else:
                manifold = get_scaled_umap_embeddings(np.asarray([row[0]for row in df.filter(like='features').values]), variable1, varibale2)
            #tnse_embeddings_mfccs.append(manifold)
            print('debug')

            if coloring_by == 'onset types':

                nd_count = df[df['onset_type'] == 'Not defined'].index.values
                if len(nd_count) > 0:
                    plt.scatter(manifold[nd_count, 0], manifold[nd_count, 1], c='red', label='Not defined')

                gs_count = df[df['onset_type'] == 'Glottal Stroke'].index.values
                if len(gs_count) > 0:
                    plt.scatter(manifold[gs_count, 0], manifold[gs_count, 1], c='green', label='Glottal Stroke')

                ga_count = df[df['onset_type'] == ' Glottal Attack'].index.values
                if len(ga_count) > 0:
                    plt.scatter(manifold[ga_count, 0], manifold[ga_count, 1], c='purple', label='Glottal Attack')

                breathy_count = df[df['onset_type'] == 'Breathy'].index.values
                if len(breathy_count) > 0:
                    plt.scatter(manifold[breathy_count, 0], manifold[breathy_count, 1], c='orange', label='Breathy')

                Simultaneous_count = df[df['onset_type'] == 'Simultaneous'].index.values
                if len(Simultaneous_count) > 0:
                    plt.scatter(manifold[Simultaneous_count, 0], manifold[Simultaneous_count, 1], c='blue', label='Simultaneous')

                un_count = df[df['onset_type'] == 'Undecided'].index.values
                if len(un_count) > 0:
                    plt.scatter(manifold[un_count, 0], manifold[un_count, 1], c='black', label='Undecided')

            if coloring_by == 'vowel':
                a_count = df[df['vowel'] == 'a'].index.values
                plt.scatter(manifold[a_count, 0], manifold[a_count, 1], c='orange', label='a')

                e_count = df[df['vowel'] == 'e'].index.values
                plt.scatter(manifold[e_count, 0], manifold[e_count, 1], c='blue', label='e')

                o_count = df[df['vowel'] == 'o'].index.values
                plt.scatter(manifold[o_count, 0], manifold[o_count, 1], c='green', label='o')
            plt.legend()


            if t_sne:
                plt.title('t-SNE colored by '+ coloring_by + ' perplexity: '+ str(variable1) +', iterations: '+ str(varibale2))
            else:
                plt.title('UMAP colored by '+ coloring_by + ' neighbour: '+ str(variable1) +', distance: '+ str(varibale2))

            if save_name == None:
                plt.show()
            else:
                if t_sne:
                    plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results\Clustering',
                                             't_sne_'+save_name+'_'+coloring_by+
                                             '_perplexity_'+str(variable1)+'_iterations_'+str(varibale2)+
                                             '.svg'),format='svg', dpi=300)
                    plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results\Clustering',
                                             't_sne_'+save_name+'_'+coloring_by+
                                             '_perplexity_'+str(variable1)+'_iterations_'+str(varibale2)+
                                             '.png'),format='png', dpi=300)
                else:
                    plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results\Clustering',
                                             'umap_'+save_name+'_'+coloring_by+
                                             '_neigbour_'+str(variable1)+'_distance_'+str(varibale2)+
                                             '.svg'),format='svg', dpi=300)
                    plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results\Clustering',
                                             'umap_'+save_name+'_'+coloring_by+
                                             '_neigbour_'+str(variable1)+'_distance_'+str(varibale2)+
                                             '.png'),format='png', dpi=300)
                plt.close()



def make_manifold_modified(files, coloring_by, save_name, t_sne= True, cluster_by_parameter=False):
    """
    Make the plot of either the umap or t-sne manifold and colorizes the datapoints according to the vowel or onset types
    Args:
        files: list of dicts containing the data of each subject
        coloring_by, String: color the manifold datapoints by means of metadatainfo. E.g. vowels, onsettype
        save_name, String: name at which the plot data should be saved
        t_sne, Boolean: use T-SNE clustering when true, otherwise use UMAP

    Returns: None

    """

    #perplexities = [2, 5, 10, 20, 30, 50, 100]
    perplexities = [20, 30, 50]
    iterations = [250, 500, 1000, 2000, 5000]

    neighbours =[15]
    distances = [0.5]

    #neighbours = [5, 15]
    #distances = [0.01,0.1, 0.5]

    extracted_list = []

    for dict in files:
        if cluster_by_parameter == False:
            extracted_info = {'acoustic': dict['acoustic'], 'vowel': dict['meta_info']['vowel'],
                              'onset_type': dict['meta_info']['onset_type'], 'name': dict['file_name'],
                              'features': dict['mfccs_features']}
        else:
            feature_list = []
            for parameter in dict['parameters']:
                if dict['parameters'][parameter] != None and parameter != 'voc':
                #if dict['parameters'][parameter] != None:
                    feature_list.append(dict['parameters'][parameter]['time'])
                    feature_list.append(dict['parameters'][parameter]['relative_time'])
            extracted_info = {'acoustic': dict['acoustic'], 'vowel': dict['meta_info']['vowel'],
                              'onset_type': dict['meta_info']['onset_type'], 'name': dict['file_name'],
                              'features': feature_list}
        extracted_list.append(extracted_info)
    df = pd.DataFrame.from_dict(extracted_list)
        #features_list.append(dict['mfccs_features'])
    #df = df[df['vowel'].str.contains("a")].reset_index()
    if t_sne:
        variables1 = perplexities
        variables2 = iterations

    else:

        variables1 = neighbours
        variables2 = distances

    for variable1 in variables1:
        for varibale2 in variables2:
            if t_sne:
                manifold = get_scaled_tsne_embeddings(np.asarray([row[0] for row in df.filter(like='features').values]), variable1, varibale2)

            else:
                manifold = get_scaled_umap_embeddings(np.asarray([row[0]for row in df.filter(like='features').values]), variable1, varibale2)
            #tnse_embeddings_mfccs.append(manifold)
            print('debug')



            nd_count = df[df['onset_type'] == 'Not defined'].index.values
            if len(nd_count) > 0:
                plt.scatter(manifold[nd_count, 0], manifold[nd_count, 1], c='red', label='Not defined')

            gs_count = df[df['onset_type'] == 'Glottal Stroke'].index.values
            if len(gs_count) > 0:
                plt.scatter(manifold[gs_count, 0], manifold[gs_count, 1], c='green', label='Glottal Stroke')

            ga_count = df[df['onset_type'] == ' Glottal Attack'].index.values
            if len(ga_count) > 0:
                plt.scatter(manifold[ga_count, 0], manifold[ga_count, 1], c='purple', label='Glottal Attack')

            breathy_count = df[df['onset_type'] == 'Breathy'].index.values
            if len(breathy_count) > 0:
                plt.scatter(manifold[breathy_count, 0], manifold[breathy_count, 1], c='orange', label='Breathy')

            Simultaneous_count = df[df['onset_type'] == 'Simultaneous'].index.values
            if len(Simultaneous_count) > 0:
                plt.scatter(manifold[Simultaneous_count, 0], manifold[Simultaneous_count, 1], c='blue', label='Simultaneous')


            un_count = df[df['onset_type'] == 'Undecided'].index.values
            if len(un_count) > 0:
                plt.scatter(manifold[un_count, 0], manifold[un_count, 1], c='black', label='Undecided')

            plt.legend()
            plt.title(
                'UMAP colored by ' + coloring_by + ' neighbour: ' + str(variable1) + ', distance: ' + str(varibale2))
            plt.show()

            a_count = df[df['vowel'] == 'a'].index.values
            plt.scatter(manifold[a_count, 0], manifold[a_count, 1], c='orange', label='a')

            e_count = df[df['vowel'] == 'e'].index.values
            plt.scatter(manifold[e_count, 0], manifold[e_count, 1], c='blue', label='e')

            o_count = df[df['vowel'] == 'o'].index.values
            plt.scatter(manifold[o_count, 0], manifold[o_count, 1], c='green', label='o')

            u_count = df[df['vowel'] == 'u'].index.values
            plt.scatter(manifold[u_count, 0], manifold[u_count, 1], c='purple', label='u')
            plt.legend()
            plt.title(
                'UMAP colored by ' + coloring_by + ' neighbour: ' + str(variable1) + ', distance: ' + str(varibale2))
            plt.show()


            if t_sne:
                plt.title('t-SNE colored by '+ coloring_by + ' perplexity: '+ str(variable1) +', iterations: '+ str(varibale2))
            else:
                plt.title('UMAP colored by '+ coloring_by + ' neighbour: '+ str(variable1) +', distance: '+ str(varibale2))

            # if save_name == None:
            #     plt.show()
            # else:
            #     if t_sne:
            #         plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results\Clustering',
            #                                  't_sne_'+save_name+'_'+coloring_by+
            #                                  '_perplexity_'+str(variable1)+'_iterations_'+str(varibale2)+
            #                                  '.svg'),format='svg', dpi=300)
            #         plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results\Clustering',
            #                                  't_sne_'+save_name+'_'+coloring_by+
            #                                  '_perplexity_'+str(variable1)+'_iterations_'+str(varibale2)+
            #                                  '.png'),format='png', dpi=300)
            #     else:
            #         plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results\Clustering',
            #                                  'umap_'+save_name+'_'+coloring_by+
            #                                  '_neigbour_'+str(variable1)+'_distance_'+str(varibale2)+
            #                                  '.svg'),format='svg', dpi=300)
            #         plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results\Clustering',
            #                                  'umap_'+save_name+'_'+coloring_by+
            #                                  '_neigbour_'+str(variable1)+'_distance_'+str(varibale2)+
            #                                  '.png'),format='png', dpi=300)
            #     plt.close()
#files = load_all(['Cate'])
#files = obtain_features(files)
#files = obtain_features(files,8000,128)

#load_all(['Evan','Cate'])
#load_all(['Cate'])
#files = load_all(None)
#files = load_all(['Cate','Evan','S01_','S02_','S003','S004','S005','S006','S007','S008','S009','S010'])

#______ clustering by parameter values

# files = load_all(['S01_','S02_','S003','S004','S005','S006','S007','S008','S009','S010'])
# make_manifold(files,'vowel',t_sne=True, save_name = 's1-10_parameters',cluster_by_parameter=True)
# make_manifold(files,'vowel',t_sne=False, save_name = 's1-10_parameters',cluster_by_parameter=True)
# make_manifold(files,'onset types',t_sne=True, save_name = 's1-10_parameters',cluster_by_parameter=True)
# make_manifold(files,'onset types',t_sne=False, save_name = 's1-10_parameters',cluster_by_parameter=True)
# make_manifold(files,'onset types',t_sne=True, save_name = 's1-10_parameters',cluster_by_parameter=True)

#files = load_all(['Cate','Evan'])
files = load_all(['S01_','S02_','S003','S004','S005','S006','S007','S008','S009','S010'])

#files = obtain_features(files,8000,128)
#make_manifold(files,'vowel',t_sne=True, save_name = 'cate-evan_parameters',cluster_by_parameter=True)
#make_manifold(files,'vowel',t_sne=False, save_name = 'cate-evan_parameters',cluster_by_parameter=True)
make_manifold_modified(files,'onset types',t_sne=False, save_name = 'cate-evan_parameters',cluster_by_parameter=True)
#make_manifold(files,'onset types',t_sne=False, save_name = 'cate-evan_parameters',cluster_by_parameter=True)
#
# files = load_all(['Cate'])
#
# make_manifold(files,'vowel',t_sne=True, save_name = 'cate_parameters',cluster_by_parameter=True)
# make_manifold(files,'vowel',t_sne=False, save_name = 'cate_parameters',cluster_by_parameter=True)
# make_manifold(files,'onset types',t_sne=True, save_name = 'cate_parameters',cluster_by_parameter=True)
# make_manifold(files,'onset types',t_sne=False, save_name = 'cate_parameters',cluster_by_parameter=True)
#
# files = load_all(['Evan'])
#
# make_manifold(files,'vowel',t_sne=True, save_name = 'evan_parameters',cluster_by_parameter=True)
# make_manifold(files,'vowel',t_sne=False, save_name = 'evan_parameters',cluster_by_parameter=True)
# make_manifold(files,'onset types',t_sne=True, save_name = 'evan_parameters',cluster_by_parameter=True)
# make_manifold(files,'onset types',t_sne=False, save_name = 'evan_parameters',cluster_by_parameter=True)

#files = load_all(None)
#files = obtain_features(files)
#files = obtain_features(files,8000,128)
# make_manifold(files,'onset types',t_sne=True, save_name = 'all_parameter',cluster_by_parameter=True)
# make_manifold(files,'onset types',t_sne=False, save_name = 'all_parameter',cluster_by_parameter=True)
# make_manifold(files,'vowel',t_sne=True, save_name = 'all_parameter',cluster_by_parameter=True)
# make_manifold(files,'vowel',t_sne=False, save_name = 'all_parameter',cluster_by_parameter=True)


#______ clustering by full acoustic signal

# files = load_all(['S01_','S02_','S003','S004','S005','S006','S007','S008','S009','S010'])
# #files = load_all(['S73'])
#
# # files = obtain_features(files)
# files = obtain_features(files,8000,128)
# make_manifold(files,'vowel',t_sne=True, save_name = 's1-10')
# make_manifold(files,'vowel',t_sne=False, save_name = 's1-10')
# make_manifold(files,'onset types',t_sne=True, save_name = 's1-10')
# make_manifold(files,'onset types',t_sne=False, save_name = 's1-10')
#
# files = load_all(['Cate','Evan','S01_','S02_','S003','S004','S005','S006','S007','S008','S009','S010'])
#
# #files = obtain_features(files)
# files = obtain_features(files,8000,128)
# make_manifold(files,'vowel',t_sne=True, save_name = 's1-10-cate-evan')
# make_manifold(files,'vowel',t_sne=False, save_name = 's1-10-cate-evan')
# make_manifold(files,'onset types',t_sne=True, save_name = 's1-10-cate-evan')
# make_manifold(files,'onset types',t_sne=False, save_name = 's1-10-cate-evan')
#
# files = load_all(['Cate','Evan'])
#
# #files = obtain_features(files)
# files = obtain_features(files,8000,128)
# make_manifold(files,'vowel',t_sne=True, save_name = 'cate-evan')
# make_manifold(files,'vowel',t_sne=False, save_name = 'cate-evan')
# make_manifold(files,'onset types',t_sne=True, save_name = 'cate-evan')
# make_manifold(files,'onset types',t_sne=False, save_name = 'cate-evan')
# #
#
# #
# files = load_all(['Evan'])
#
# #files = obtain_features(files)
# files = obtain_features(files,8000,128)
# make_manifold(files,'vowel',t_sne=True, save_name = files = load_all(['Cate'])
#
# #files = obtain_features(files)
# files = obtain_features(files,8000,128)
# make_manifold(files,'vowel',t_sne=True, save_name = 'cate')
# make_manifold(files,'vowel',t_sne=False, save_name = 'cate')
# make_manifold(files,'onset types',t_sne=True, save_name = 'cate')
# make_manifold(files,'onset types',t_sne=False, save_name = 'cate')'evan')
# make_manifold(files,'vowel',t_sne=False, save_name = 'evan')
# make_manifold(files,'onset types',t_sne=True, save_name = 'evan')
# make_manifold(files,'onset types',t_sne=False, save_name = 'evan')
#

#
# files = load_all(None)
# #files = obtain_features(files)
# files = obtain_features(files,8000,128)
# make_manifold(files,'onset types',t_sne=True, save_name = 'all')
# make_manifold(files,'onset types',t_sne=False, save_name = 'all')
# make_manifold(files,'vowel',t_sne=True, save_name = 'all')
# make_manifold(files,'vowel',t_sne=False, save_name = 'all')
#
#
# files = obtain_features(files,8000,128)
# make_manifold(files,'vowel',t_sne=True, save_name = 's1-10')
# make_manifold(files,'vowel',t_sne=False, save_name = 's1-10')
# make_manifold(files,'onset types',t_sne=True, save_name = 's1-10')
# make_manifold(files,'onset types',t_sne=False, save_name = 's1-10')

# files = load_all(['S01_','S02_','S003','S004','S005','S006','S007','S008','S009','S010'])
#
# files = obtain_features(files,8000,128, image= True)
# make_manifold(files,'vowel',t_sne=True, save_name = 's1-10')
# make_manifold(files,'vowel',t_sne=False, save_name = 's1-10')
# make_manifold(files,'onset types',t_sne=True, save_name = 's1-10')
# make_manifold(files,'onset types',t_sne=False, save_name = 's1-10')
#
# files = load_all(['Cate','Evan','S01_','S02_','S003','S004','S005','S006','S007','S008','S009','S010'])
#
# #files = obtain_features(files)
# files = obtain_features(files,8000,128, image= True)
# make_manifold(files,'vowel',t_sne=True, save_name = 's1-10-cate-evan')
# make_manifold(files,'vowel',t_sne=False, save_name = 's1-10-cate-evan')
# make_manifold(files,'onset types',t_sne=True, save_name = 's1-10-cate-evan')
# make_manifold(files,'onset types',t_sne=False, save_name = 's1-10-cate-evan')
#
# files = load_all(['Cate','Evan'])
#
# #files = obtain_features(files)
# files = obtain_features(files,8000,128, image= True)
# make_manifold(files,'vowel',t_sne=True, save_name = 'cate-evan')
# make_manifold(files,'vowel',t_sne=False, save_name = 'cate-evan')
# make_manifold(files,'onset types',t_sne=True, save_name = 'cate-evan')
# make_manifold(files,'onset types',t_sne=False, save_name = 'cate-evan')
# #
#
# #
# files = load_all(['Evan'])
#
# #files = obtain_features(files)
# files = obtain_features(files,8000,128, image= True)
# make_manifold(files,'onset types',t_sne=True, save_name = 'evan')
#
# make_manifold(files,'vowel',t_sne=True, save_name = 'evan')
# make_manifold(files,'vowel',t_sne=False, save_name = 'evan')
# make_manifold(files,'onset types',t_sne=True, save_name = 'evan')
# make_manifold(files,'onset types',t_sne=False, save_name = 'evan')
#
# files = load_all(['Cate'])
#
# #files = obtain_features(files)
# files = obtain_features(files,8000,128, image= True)
# #make_manifold(files,'vowel',t_sne=True, save_name = 'cate')
# #make_manifold(files,'vowel',t_sne=False, save_name = 'cate')
# #make_manifold(files,'onset types',t_sne=True, save_name = 'cate')
# make_manifold(files,'onset types',t_sne=False, save_name = 'cate')
#
# files = load_all(None)
# #files = obtain_features(files)
# files = obtain_features(files,8000,128, image= True)
# make_manifold(files,'onset types',t_sne=True, save_name = 'all')
# make_manifold(files,'onset types',t_sne=False, save_name = 'all')
# make_manifold(files,'vowel',t_sne=True, save_name = 'all')
# make_manifold(files,'vowel',t_sne=False, save_name = 'all')

#
# files = load_all(['S01_','S02_','S003','S004','S005','S006','S007','S008','S009','S010'])
#
# #files = obtain_features(files)
# files = obtain_features(files, 8000, 128, image= True)
# #make_manifold(files,'vowel',t_sne=True, save_name = 'cate')
# #make_manifold(files,'vowel',t_sne=False, save_name = 'cate')
# #make_manifold(files,'onset types',t_sne=True, save_name = 'cate')
# make_manifold(files,'onset types',t_sne=False, save_name = None)



print('debug')
from os import walk
from os.path import isfile, join
import numpy as np
import pandas as pd
import flammkuchen as fl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from PhonationOnset.VoiceOnsetTime import VoiceOnsetTime
from PhonationOnset.VocalAttackTime import VocalAttackTime
from PhonationOnset.VocalRiseTime import VocalRiseTime
from PhonationOnset.VoiceOnsetCoordination import VoiceOnsetCoordination
import seaborn
import os
import flammkuchen

import matplotlib.pylab as pylab

"""
Code to make Boxplots and table of paramter results
"""

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update({'font.size': 10})
pylab.rcParams.update(params)

data_path = r'D:\Uni\Masterarbeit\Daten\Annotated_subjects'

save_path = r'D:\Uni\Masterarbeit\Final_Results'
import copy


def load_all(participant_selection=None, exclude_selection=None):

    """
    loads the files
    Args:
        participant_selection, List of Strings: only considers files which have substrings in common with participant_selection
        exclude_selection, List of Strings: discards files which have substrings in common with participant_selection:

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
    if exclude_selection == None:
        if participant_selection is not None:
            for participant in participant_selection:
                for file in f:
                    if participant in file:
                        new_f.append(file)

            f = new_f
    else:
        for participant in exclude_selection:
            for file in f:
                if participant in file:
                    new_f.append(file)
        f = [x for x in f if x not in new_f]

    file_list = []
    for i, file_name in enumerate(f):
            file = fl.load(parent_folder+'/'+file_name)
            file_list.append(file)

    return file_list

def modify_parameters(file_list, parameter=None, custom_hyperparameters=None):

    """
    recalculated the parameters
    Either with their default hyperparameters or the hyperparameters given

    Args:
        file_list, List of dicts: containing the data
        parameter, None or String: if None all parameter will be calculated, if String
                    only the given parameter is calculated
        custom_hyperparameters, dict: if parameter is not none, the given parameter parameter will be calcuted with the
                                    values given

    Returns: List of dicts containg the calculated values

    """

    default_kunduk = fl.load(r'../GUI/Settings/vot_kunduk_default_keys')
    default_mergel = fl.load(r'../GUI/Settings/vot_mergel_default_keys')
    default_vrt = fl.load(r'../GUI/Settings/vrt_default_keys')
    default_vat = fl.load(r'../GUI/Settings/vat_default_keys')
    default_voc = fl.load(r'../GUI/Settings/voc_default_keys')

    default_kunduk['first_significant_peak_crit'] = 0.5
    default_mergel['first_significant_peak_crit'] = 0.5

    default_vrt['segment_window'] = 0.8


    file_list_copy = copy.deepcopy(file_list)
    except_counter = 0
    for file in file_list_copy:
        for j, event_dict in enumerate(file):
            print('calculating: ' + event_dict['file_name'])
            #HAD initial some problems with the onset type labeling differences between the real and acted dataset
            if event_dict['meta_info']['onset_type'] == ' Glottal attack':
                event_dict['meta_info']['onset_type'] = ' Glottal Attack'
            if event_dict['meta_info']['onset_type'] == ' Glottal stroke':
                event_dict['meta_info']['onset_type'] = 'Glottal Stroke'
            if event_dict['meta_info']['onset_type'] == ' Undecided':
                event_dict['meta_info']['onset_type'] = 'Undecided'
            if event_dict['meta_info']['onset_type'] == ' Simultaneous':
                event_dict['meta_info']['onset_type'] = 'Simultaneous'

            if custom_hyperparameters == None:
                event_dict['meta_info']['onset_type'] = event_dict['meta_info']['onset_type'] + '_default'
            else:
                event_dict['meta_info']['onset_type'] = event_dict['meta_info']['onset_type'] + '_adjusted'

            if parameter == None or 'vot_mergel' in parameter:
                if custom_hyperparameters != None:
                    hp = event_dict['parameters']['vot_mergel']['hyperparameters']
                    for key in custom_hyperparameters:
                        hp[key] = custom_hyperparameters[key]
                    vot = VoiceOnsetTime(**hp)
                else:
                    vot = VoiceOnsetTime(**default_mergel)
                try:
                    vot.compute(time = event_dict['time'], acoustic= event_dict['acoustic'], framerate=event_dict['meta_info']['framerate'], method = 'mergel')
                    event_dict['parameters']['vot_mergel']['time'] = vot.time
                    event_dict['parameters']['vot_mergel']['relative_time'] = vot.norm_time
                except Exception:
                    except_counter += 1

            #only implemented the hyperparameter adjustments for kunduk, mergel & vrt
            if parameter == None or 'vot_kunduk' in parameter:
                if custom_hyperparameters != None:
                    hp = event_dict['parameters']['vot_kunduk']['hyperparameters']
                    for key in custom_hyperparameters:
                        hp[key] = custom_hyperparameters[key]
                    vot = VoiceOnsetTime(**hp)

                else:
                    vot = VoiceOnsetTime(**default_kunduk)
                try:
                    vot.compute(time = event_dict['time'], acoustic= event_dict['acoustic'], framerate=event_dict['meta_info']['framerate'])
                    event_dict['parameters']['vot_kunduk']['time'] = vot.time
                    event_dict['parameters']['vot_kunduk']['relative_time'] = vot.norm_time
                except Exception:
                    except_counter += 1

            if parameter == None or parameter == 'vrt':
                if custom_hyperparameters != None:
                    hp = event_dict['parameters']['vrt']['hyperparameters']
                    for key in custom_hyperparameters:
                        hp[key] = custom_hyperparameters[key]
                    vrt = VocalRiseTime(**hp)

                else:
                    vrt = VocalRiseTime(**default_vrt)
                vrt.compute(time = event_dict['time'], acoustic= event_dict['acoustic'], framerate=event_dict['meta_info']['framerate'])
                event_dict['parameters']['vrt']['time'] = vrt.time
                event_dict['parameters']['vrt']['relative_time'] = vrt.norm_time

            if parameter == None or parameter == 'vat':
                vat = VocalAttackTime(**default_vat)
                vat.compute(time = event_dict['time'], acoustic= event_dict['acoustic'], egg= event_dict['egg'] ,framerate=event_dict['meta_info']['framerate'])
                event_dict['parameters']['vat']['time'] = vat.time
                event_dict['parameters']['vat']['relative_time'] = vat.norm_time

            if parameter == None or parameter == 'voc':

                try:
                    voc = VoiceOnsetCoordination(**default_voc)
                    voc.compute(time = event_dict['time'], acoustic= event_dict['airflow'], egg= event_dict['egg'] ,framerate=event_dict['meta_info']['framerate'])
                    event_dict['parameters']['voc']['time'] = voc.time
                    event_dict['parameters']['voc']['relative_time'] = voc.norm_time
                except Exception:
                    None
    #print out how any parameter could not be calculated
    print(except_counter)
    return file_list_copy




def extract_data(file_list,time = True):

    """
    extracts the time or relative time data out of the dicts into a dataframe
    Args:
        file_list: which files to process
        time, default = True: process time or relative time

    Returns: dict

    """

    for i,file in enumerate(file_list):
        for j,event_dict in enumerate(file):
            dict={}
            #event_dict=event_dict[0]
            #if event_dict['meta_info']['vowel'] == 'a':# or event_dict['meta_info']['vowel'] == 'o':
            for key in (event_dict['parameters']):
                if event_dict['parameters'][key] is not None and isinstance(event_dict['parameters'][key]['time'],float):
                        if time:
                            dict[key] = [event_dict['parameters'][key]['time']]

                        else:
                            dict[key] = [event_dict['parameters'][key]['relative_time']]
                elif event_dict['parameters'][key] is None:
                    dict[key] = [None]
                else:
                    print('debug')
            dict['class'] = event_dict['meta_info']['onset_type']
            if dict['class'] == ' Glottal attack':
                dict['class'] = ' Glottal Attack'
            if dict['class'] == ' Glottal stroke':
                dict['class'] = 'Glottal Stroke'
            if dict['class'] == ' Undecided':
                dict['class'] = 'Undecided'
            if dict['class'] == ' Simultaneous':
                dict['class'] = 'Simultaneous'
            if i == 0 and j == 0:
                #df = pd.DataFrame.from_dict(dict,index = event_dict['file_name'])
                df = pd.DataFrame.from_dict(dict)
                df = df.rename(index={0:event_dict['file_name']})

            else:
                #concat = [df,pd.DataFrame.from_dict(dict, index = event_dict['file_name'])]
                concat = [df,pd.DataFrame.from_dict(dict).rename(index={0:event_dict['file_name']})]

                df = pd.concat(concat)

    return df



def make_average_comp_per_onset_type(df,keys, time=True , add_to_save_name='', add_to_title=''):

    """
    Makes a comparision plot for the acted and real datas mean values for each given parameter
    Args:
        df, dataframe: containing the extracted parameter times/relative times and the corresponding class
        keys, List of strings: for which parameters a boxplot should be made
        time, bool: switch to adjust the title name from time to relative tome
        add_to_save_name, String: substring which is added to the saved boxplots name
        add_to_title, String: substring which is added to the saved boxplots title

    Returns: None

    """
    df1 = df[~df.index.str.contains('Cate') & ~df.index.str.contains('Evan')]
    df2 = df[df.index.str.contains('Cate', case=True) | df.index.str.contains('Evan', case=True)]
    print('debug')
    classes = ['Breathy', 'Simultaneous', 'Glottal Stroke', ' Glottal Attack', 'Undecided']
    for key in keys:
        ax = plt.subplot(111)
        x_values = [1,2]
        y_values = [df2[key].mean(),df1[key].mean()]
        #fig, ax = plt.subplots(1, 1)
        plt.plot(x_values,y_values)
        plt.scatter(x_values,y_values,label='All onset types')
        for cl in classes:
            x_values = [1,2]
            y_values = [df2[df2['class'].str.match(cl)][key].mean(),df1[df1['class'].str.match(cl)][key].mean()]
            #fig, ax = plt.subplots(1, 1)
            plt.plot(x_values,y_values)
            plt.scatter(x_values,y_values,label=cl)
            plt.xticks(x_values,['acted','real'])
    #plt.title('Mean comparision between the real and acted data ')
        plt.title(add_to_title+'\n '+key+' mean comparision between \n the real (S01-S10) and acted data ')
        if time == True:
            plt.ylabel('mean times')

        else:
            plt.ylabel('relative mean times')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.15, box.height])
        plt.legend(bbox_to_anchor=(1, 1))

        #plt.show()
    #TODO: check title!
    #plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results' ,'mean_comp_all.svg'), format='svg',dpi=300)
    #plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results','mean_comp_all.png'), format='png')
        plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results', key+ '_mean_comp'+add_to_save_name+'.svg'), format='svg', dpi=300,
                    bbox_inches = 'tight', pad_inches = 0)
        plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results', key+'_mean_comp'+add_to_save_name+'.png'), format='png',
                    bbox_inches = 'tight', pad_inches = 0)
        plt.close()
    #plt.show()


    #TODO: check title!
    #
    # plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results' ,'rel_mean_comp_all.svg'), format='svg',dpi=300)
    # plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results','rel_mean_comp_all.png'), format='png')
    # #plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results', 'rel_mean_comp.svg'), format='svg', dpi=300)
    # #plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results', 'rel_mean_comp.png'), format='png')
    # plt.close()
    # #plt.show()


def make_box_plot_comp(df, parameters,time = True ,add_to_save_name='', add_to_title=''):

    """
    Makes boxplots for both the acted and real data and
    for each given parameter, to compare the distributions of the different classes
    Args:
        df, dataframe: containing the extracted parameter times/relative times and the corresponding class
        parameters, List of strings: for which parameters a boxplot should be made
        time, bool: switch to adjust the title name from time to relative tome
        add_to_save_name, String: substring which is added to the saved boxplots name
        add_to_title, String: substring which is added to the saved boxplots title

    Returns: None

    """

    #without cate & evan
    df1 = df[~df.index.str.contains('Cate') & ~df.index.str.contains('Evan')]
    df1.groupby(by='class').describe()
    #with cate & evan
    df2 = df[df.index.str.contains('Cate', case=True) | df.index.str.contains('Evan', case=True)]

    if len(df['class'].unique()) == 5:

        order=['Breathy', 'Simultaneous', 'Glottal Stroke', ' Glottal Attack', 'Undecided']
        palette = 'colorblind'

    elif df['class'].str.contains('Simultaneous_adjusted').sum() > 0:
        palette = ['silver', 'grey', 'saddlebrown', 'sandybrown', 'darkgreen', 'forestgreen', 'darkorange', 'orange', 'purple',
                   'darkorchid']
        order=['Breathy', 'Breathy_adjusted', 'Simultaneous', 'Simultaneous_adjusted', 'Glottal Stroke',
               'Glottal Stroke_adjusted', ' Glottal Attack', ' Glottal Attack_adjusted', 'Undecided',
               'Undecided_adjusted']

    elif df['class'].str.contains('Simultaneous_default').sum() > 0:
        palette = ['silver', 'grey', 'saddlebrown', 'sandybrown', 'darkgreen', 'forestgreen', 'darkorange', 'orange', 'purple',
                   'darkorchid']
        order=['Breathy', 'Breathy_default', 'Simultaneous', 'Simultaneous_default', 'Glottal Stroke',
               'Glottal Stroke_default', ' Glottal Attack',' Glottal Attack_default', 'Undecided', 'Undecided_default']



    for key in parameters:


        boxplot1 = seaborn.boxplot(y='class', x=key,
                    data=df1, order = order,
                    palette="colorblind",showfliers = False)
        x_min1, x_max1 = boxplot1.get_xlim()
        boxplot2 = seaborn.boxplot(y='class', x=key,
                                  data=df2,
                                  order=order,
                                  palette="colorblind", showfliers=False)
        x_min2, x_max2 = boxplot2.get_xlim()

        if x_min1<x_min2:
            min = x_min1
        else:
            min = x_min2

        if x_max1>x_max2:
            max = x_max1
        else:
            max = x_max2
        plt.close()
        ax = plt.subplot(111)

        boxplot1 = seaborn.boxplot(y='class', x=key,
                    data=df1, order =order,
                    palette=palette,showfliers = False)
        if time == True:
            plt.title(add_to_title+ 'real data (S01-S10) ' + key + ' time distributions')
        else:
            plt.title(add_to_title+ 'real data (S01-S10) ' + key + ' relative time distributions')

        plt.xlim(min,max)
        box = ax.get_position()
        ax.set_position([box.x0+0.1, box.y0, box.width * 0.8, box.height])


        plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results',key+'_just_S1-10'+add_to_save_name+'.svg'), format = 'svg', dpi=300)
        plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results',key+'_just_S1-10'+add_to_save_name+'.png'), format = 'png')

        plt.close()
        ax = plt.subplot(111)

        boxplot2 = seaborn.boxplot(y='class', x=key,
                                   data=df2,
                                   order=order,
                                   palette=palette, showfliers=False)
        #plt.title('acted data ' + key + ' time distributions, hyperparameter comparison between GUI annotated and automatic calculation')
        if time == True:
            plt.title(add_to_title+ 'acted data ' + key + ' time distributions')
        else:
            plt.title(add_to_title+ 'acted data ' + key + ' relative time distributions')

        plt.xlim(min, max)
        box = ax.get_position()
        ax.set_position([box.x0+0.1, box.y0, box.width * 0.8, box.height])


        plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results',key+'_just_cate_and_evan'+add_to_save_name+'.svg'), format = 'svg', dpi=300)
        plt.savefig(os.path.join(r'D:\Uni\Masterarbeit\Final_Results', key + '_just_cate_and_evan'+add_to_save_name+'.png'), format='png')

        #plt.show()
        plt.close()



def recalculate_vrt():

    """
    method to recalculate specifically the vrt param values, because the hyperparameter of 0.2 second was way to small
    Returns:

    """
    files = load_all(None)
    param = {'segment_window': 0.8}
    for file in files:
        for j, event_dict in enumerate(file):
            print('calculating: ' + event_dict['file_name'])
            if event_dict['parameters']['vrt'] != None:
                hp = event_dict['parameters']['vrt']['hyperparameters']
                for key in param:
                    hp[key] = param[key]
                vrt = VocalRiseTime(**hp)

                vrt.compute(time = event_dict['time'], acoustic= event_dict['acoustic'], framerate=event_dict['meta_info']['framerate'])
                event_dict['parameters']['vrt']['time'] = vrt.time
                event_dict['parameters']['vrt']['relative_time'] = vrt.norm_time
                event_dict['parameters']['vrt']['onset_label'] = vrt.onset_label
        flammkuchen.save(data_path + '/' + event_dict['file_name'].split('-')[0], file)
        print('debug')

#extract_data(f_list_changed)

# all normal
f_list = load_all(['Cate','Evan','S01_','S02_','S003','S004','S005','S006','S007','S008','S009','S010'])#,'S04_','S05_','S06',S])
#time
df = extract_data(f_list)
df['voc'] = df['voc'].astype(np.float64)

df1 = df[~df.index.str.contains('Cate') & ~df.index.str.contains('Evan')]
df2 = df[df.index.str.contains('Cate', case=True) | df.index.str.contains('Evan', case=True)]
print('debug')
# df1.corr(method='pearson').to_excel(os.path.join(save_path, "corr_S1-10.xlsx"))
# df2.corr(method='pearson').to_excel(os.path.join(save_path, "corr_cate-evan.xlsx"))
#
# df1.groupby(by='class').describe().to_excel(os.path.join(save_path, "params_S1-10.xlsx"))
# df1.groupby(by='class').describe().T.to_excel(os.path.join(save_path, "params_S1-10_T.xlsx"))
#
# df2.groupby(by='class').describe().to_excel(os.path.join(save_path, "params_cate-evan.xlsx"))
# df2.groupby(by='class').describe().T.to_excel(os.path.join(save_path, "params_cate-evan_T.xlsx"))
#
# make_box_plot_comp(df,list(df.columns.values[:-1]))
# make_average_comp_per_onset_type(df,list(df.columns.values[:-1]))
# #relative time
#
# df = extract_data(f_list, time=False)
# df['voc'] = df['voc'].astype(np.float64)
#
# df1 = df[~df.index.str.contains('Cate') & ~df.index.str.contains('Evan')]
# df2 = df[df.index.str.contains('Cate', case=True) | df.index.str.contains('Evan', case=True)]
#
# df1.corr(method='pearson').to_excel(os.path.join(save_path, "corr_S1-10_relative_time.xlsx"))
# df2.corr(method='pearson').to_excel(os.path.join(save_path, "corr_cate-evan_relative_time.xlsx"))
#
# df1.groupby(by='class').describe().to_excel(os.path.join(save_path, "params_S1-10_relative_time.xlsx"))
# df1.groupby(by='class').describe().T.to_excel(os.path.join(save_path, "params_S1-10_T_relative_time.xlsx"))
#
# df2.groupby(by='class').describe().to_excel(os.path.join(save_path, "params_cate-evan_relative_time.xlsx"))
# df2.groupby(by='class').describe().T.to_excel(os.path.join(save_path, "params_cate-evan_T_relative_time.xlsx"))
#
# make_box_plot_comp(df,list(df.columns.values[:-1]), time=False, add_to_save_name='_relative_time')
# make_average_comp_per_onset_type(df,list(df.columns.values[:-1]), time=False, add_to_save_name='_relative_time')

#____________________________________________________________________________________________________________________________
#compare with default parameters
# f_list = load_all(['Cate','Evan','S01_','S02_','S003','S004','S005','S006','S007','S008','S009','S010'])#,'S04_','S05_','S06',S])
# #f_list = load_all(['Cate','S010'])#,'S04_','S05_','S06',S])
#
# f_list_changed = modify_parameters(f_list)
# f = f_list + f_list_changed
#
# #Time
# df_modified = extract_data(f_list_changed)
#
# df_modified['voc'] = df_modified['voc'].astype(np.float64)
#
# make_average_comp_per_onset_type(df_modified,list(df_modified.columns.values[:-1]), add_to_save_name='_default', add_to_title='default hyperparameters ')
#
# df1_modified = df_modified[~df_modified.index.str.contains('Cate') & ~df_modified.index.str.contains('Evan')]
# df2_modified = df_modified[df_modified.index.str.contains('Cate', case=True) | df_modified.index.str.contains('Evan', case=True)]
#
# df1_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_S1-10_default.xlsx"))
# df2_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_cate-evan_default.xlsx"))
#
# df1_modified.groupby(by='class').describe().to_excel(os.path.join(save_path, "params_default_S1-10.xlsx"))
# df1_modified.groupby(by='class').describe().T.to_excel(os.path.join(save_path, "params_default_S1-10_T.xlsx"))
#
# df2_modified.groupby(by='class').describe().to_excel(os.path.join(save_path, "params_default_cate-evan.xlsx"))
# df2_modified.groupby(by='class').describe().T.to_excel(os.path.join(save_path, "params_default_cate-evan_T.xlsx"))
#
# df = extract_data(f)
#
# make_box_plot_comp(df,list(df.columns.values[:-1]), add_to_save_name='_default', add_to_title='default hyperparameters ')
#
# # relative Times
# df_modified = extract_data(f_list_changed, time=False)
# df_modified['voc'] = df_modified['voc'].astype(np.float64)
#
# make_average_comp_per_onset_type(df_modified,list(df_modified.columns.values[:-1]), time = False, add_to_save_name='_default_relative_time', add_to_title='default hyperparameters ')
#
# df1_modified = df_modified[~df_modified.index.str.contains('Cate') & ~df_modified.index.str.contains('Evan')]
# df2_modified = df_modified[df_modified.index.str.contains('Cate', case=True) | df_modified.index.str.contains('Evan', case=True)]
#
# df1_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_S1-10_default_relative_time.xlsx"))
# df2_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_cate-evan_default_relative_time.xlsx"))
#
# df1_modified.groupby(by='class').describe().to_excel(os.path.join(save_path, "params_default_S1-10_relative_time.xlsx"))
# df1_modified.groupby(by='class').describe().T.to_excel(os.path.join(save_path, "params_default_S1-10_T_relative_time.xlsx"))
#
# df2_modified.groupby(by='class').describe().to_excel(os.path.join(save_path, "params_default_cate-evan_relative_time.xlsx"))
# df2_modified.groupby(by='class').describe().T.to_excel(os.path.join(save_path, "params_default_cate-evan_T_relative_time.xlsx"))
#
# df = extract_data(f,time=False)
# # df['voc'] = df['voc'].astype(np.float64)
# make_box_plot_comp(df,list(df.columns.values[:-1]), time = False ,add_to_save_name='_default_relative_time', add_to_title='default hyperparameters ')
#
# print('debug')
# #MERGEL adujustment
#
#
# print('debug')

# f_list = load_all(['Cate','Evan','S01_','S02_','S003','S004','S005','S006','S007','S008','S009','S010'])#,'S04_','S05_','S06',S])
# #
# #____first sig peak offs
# f_list_changed = modify_parameters(f_list,parameter='vot_mergel', custom_hyperparameters={'first_significant_peak_option': "off"})
# f = f_list + f_list_changed
#
# df_modified = extract_data(f_list_changed)
#
# make_average_comp_per_onset_type(df_modified,['vot_mergel'], add_to_save_name='_adjusted_mergel_fspo-off',add_to_title='modified hyperparameters ')
#
# df1_modified = df_modified[~df_modified.index.str.contains('Cate') & ~df_modified.index.str.contains('Evan')]
# df2_modified = df_modified[df_modified.index.str.contains('Cate', case=True) | df_modified.index.str.contains('Evan', case=True)]
#
# df1_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_MERGEL_params_adjusted_mergel_fspo-off_S1-10.xlsx"))
# df2_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_MERGEL_params_adjusted_mergel_fspo-off_cate-evan.xlsx"))
#
# df1_modified.groupby(by='class')['vot_mergel'].describe().to_excel(os.path.join(save_path, "MERGEL_params_adjusted_mergel_fspo-off_S1-10.xlsx"))
# df1_modified.groupby(by='class')['vot_mergel'].describe().T.to_excel(os.path.join(save_path, "MERGEL_params_adjusted_mergel_fspo-off_S1-10_T.xlsx"))
#
# df2_modified.groupby(by='class')['vot_mergel'].describe().to_excel(os.path.join(save_path, "MERGEL_params_adjusted_mergel_fspo-off_cate-evan.xlsx"))
# df2_modified.groupby(by='class')['vot_mergel'].describe().T.to_excel(os.path.join(save_path, "MERGEL_params_adjusted_mergel_fspo-off_cate-evan_T.xlsx"))
#
# df = extract_data(f)
#
# make_box_plot_comp(df,['vot_mergel'], add_to_save_name='_adjusted_mergel_fspo-off', add_to_title='modified hyperparameters ')
#
# #first sig peak percentag 80
# f_list_changed = modify_parameters(f_list,parameter='vot_mergel', custom_hyperparameters={'first_significant_peak_crit':0.8})
# f = f_list + f_list_changed
#
# df_modified = extract_data(f_list_changed)
#
# make_average_comp_per_onset_type(df_modified,['vot_mergel'], add_to_save_name='_adjusted_mergel_fspc-80',add_to_title='modified hyperparameters ')
#
# df1_modified = df_modified[~df_modified.index.str.contains('Cate') & ~df_modified.index.str.contains('Evan')]
# df2_modified = df_modified[df_modified.index.str.contains('Cate', case=True) | df_modified.index.str.contains('Evan', case=True)]
#
# df1_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_MERGEL_params_adjusted_mergel_fspc-80_S1-10.xlsx"))
# df2_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_MERGEL_params_adjusted_mergel_fspc-80_cate-evan.xlsx"))
#
#
# df1_modified.groupby(by='class')['vot_mergel'].describe().to_excel(os.path.join(save_path, "MERGEL_params_adjusted_mergel_fspc-80_S1-10.xlsx"))
# df1_modified.groupby(by='class')['vot_mergel'].describe().T.to_excel(os.path.join(save_path, "MERGEL_params_adjusted_mergel_fspc-80_S1-10_T.xlsx"))
#
# df2_modified.groupby(by='class')['vot_mergel'].describe().to_excel(os.path.join(save_path, "MERGEL_params_adjusted_mergel_fspc-80_cate-evan.xlsx"))
# df2_modified.groupby(by='class')['vot_mergel'].describe().T.to_excel(os.path.join(save_path, "MERGEL_params_adjusted_mergel_fspc-80_cate-evan_T.xlsx"))
#
# df = extract_data(f)
#
# make_box_plot_comp(df,['vot_mergel'], add_to_save_name='_adjusted_mergel_fspc-80', add_to_title='modified hyperparameters ')
#
# custom_hyperparameters={'F_0_bandpass_filter_range': 0.4,'first_significant_peak_crit':0.8,'freq_crit':0.15}
#
# f_list_changed = modify_parameters(f_list,parameter='vot_mergel',
#         custom_hyperparameters={'F_0_bandpass_filter_range': 0.4,'first_significant_peak_crit':0.8,'freq_crit':0.15})
# f = f_list + f_list_changed
#
# df_modified = extract_data(f_list_changed)
#
# make_average_comp_per_onset_type(df_modified,['vot_mergel'], add_to_save_name='_adjusted_mergel_fspc-80_fr-40_fc_15',add_to_title='modified hyperparameters ')
#
# df1_modified = df_modified[~df_modified.index.str.contains('Cate') & ~df_modified.index.str.contains('Evan')]
# df2_modified = df_modified[df_modified.index.str.contains('Cate', case=True) | df_modified.index.str.contains('Evan', case=True)]
#
# df1_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_MERGEL_params_adjusted_mergel_fspc-80_fr-40_fc_15_S1-10.xlsx"))
# df2_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_MERGEL_params_adjusted_mergel_fspc-80_fr-40_fc_15_cate-evan.xlsx"))
#
# df1_modified.groupby(by='class')['vot_mergel'].describe().to_excel(os.path.join(save_path, "MERGEL_params_adjusted_mergel_fspc-80_fr-40_fc_15_S1-10.xlsx"))
# df1_modified.groupby(by='class')['vot_mergel'].describe().T.to_excel(os.path.join(save_path, "MERGEL_params_adjusted_mergel_fspc-80_fr-40_fc_15_S1-10_T.xlsx"))
#
# df2_modified.groupby(by='class')['vot_mergel'].describe().to_excel(os.path.join(save_path, "MERGEL_params_adjusted_mergel_fspc-80_fr-40_fc_15_cate-evan.xlsx"))
# df2_modified.groupby(by='class')['vot_mergel'].describe().T.to_excel(os.path.join(save_path, "MERGEL_params_adjusted_mergel_fspc-80_fr-40_fc_15_cate-evan_T.xlsx"))
#
# df = extract_data(f)
#
# make_box_plot_comp(df,['vot_mergel'], add_to_save_name='_adjusted_mergel_fspc-80_fr-40_fc_15', add_to_title='modified hyperparameters ')
#
# #_________________________________________________________________
# f_list_changed = modify_parameters(f_list,parameter='vot_mergel',
#         custom_hyperparameters={'F_0_bandpass_filter_range': 0.4,'first_significant_peak_crit':0.8,'freq_crit':0.5})
# f = f_list + f_list_changed
#
# df_modified = extract_data(f_list_changed)
#
# make_average_comp_per_onset_type(df_modified,['vot_mergel'], add_to_save_name='_adjusted_mergel_fspc-80_fr-40_fc_50',add_to_title='modified hyperparameters ')
#
# df1_modified = df_modified[~df_modified.index.str.contains('Cate') & ~df_modified.index.str.contains('Evan')]
# df2_modified = df_modified[df_modified.index.str.contains('Cate', case=True) | df_modified.index.str.contains('Evan', case=True)]
#
# df1_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_MERGEL_params_adjusted_mergel_fspc-80_fr-40_fc_50_S1-10.xlsx"))
# df2_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_MERGEL_params_adjusted_mergel_fspc-80_fr-40_fc_50_cate-evan.xlsx"))
#
# df1_modified.groupby(by='class')['vot_mergel'].describe().to_excel(os.path.join(save_path, "MERGEL_params_adjusted_mergel_fspc-80_fr-40_fc_50_S1-10.xlsx"))
# df1_modified.groupby(by='class')['vot_mergel'].describe().T.to_excel(os.path.join(save_path, "MERGEL_params_adjusted_mergel_fspc-80_fr-40_fc_50_S1-10_T.xlsx"))
#
# df2_modified.groupby(by='class')['vot_mergel'].describe().to_excel(os.path.join(save_path, "MERGEL_params_adjusted_mergel_fspc-80_fr-40_fc_50_cate-evan.xlsx"))
# df2_modified.groupby(by='class')['vot_mergel'].describe().T.to_excel(os.path.join(save_path, "MERGEL_params_adjusted_mergel_fspc-80_fr-40_fc_50_cate-evan_T.xlsx"))
#
# df = extract_data(f)
#
# make_box_plot_comp(df,['vot_mergel'], add_to_save_name='_adjusted_mergel_fspc-80_fr-40_fc_50', add_to_title='modified hyperparameters ')
#
# #KUNDUK Adjustment
# #____first sig peak off
#
# f_list_changed = modify_parameters(f_list,parameter='vot_kunduk', custom_hyperparameters={'first_significant_peak_option': "off"})
# f = f_list + f_list_changed
#
# df_modified = extract_data(f_list_changed)
#
# make_average_comp_per_onset_type(df_modified,['vot_kunduk'], add_to_save_name='_adjusted_kunduk_fspo-off',add_to_title='modified hyperparameters ')
#
# df1_modified = df_modified[~df_modified.index.str.contains('Cate') & ~df_modified.index.str.contains('Evan')]
# df2_modified = df_modified[df_modified.index.str.contains('Cate', case=True) | df_modified.index.str.contains('Evan', case=True)]
#
# df1_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_KUNDUK_params_adjusted_kunduk_fspo-off_S1-10.xlsx"))
# df2_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_KUNDUK_params_adjusted_kunduk_fspo-off_cate-evan.xlsx"))
#
# df1_modified.groupby(by='class')['vot_kunduk'].describe().to_excel(os.path.join(save_path, "KUNDUK_params_adjusted_kunduk_fspo-off_S1-10.xlsx"))
# df1_modified.groupby(by='class')['vot_kunduk'].describe().T.to_excel(os.path.join(save_path, "KUNDUK_params_adjusted_kunduk_fspo-off_S1-10_T.xlsx"))
#
# df2_modified.groupby(by='class')['vot_kunduk'].describe().to_excel(os.path.join(save_path, "KUNDUK_params_adjusted_kunduk_fspo-off_cate-evan.xlsx"))
# df2_modified.groupby(by='class')['vot_kunduk'].describe().T.to_excel(os.path.join(save_path, "KUNDUK_params_adjusted_kunduk_fspo-off_cate-evan_T.xlsx"))
#
# df = extract_data(f)
#
# make_box_plot_comp(df,['vot_kunduk'], add_to_save_name='_adjusted_kunduk_fspo-off', add_to_title='modified hyperparameters ')
#
# #first sig peak percentag 80
# f_list_changed = modify_parameters(f_list,parameter='vot_kunduk', custom_hyperparameters={'first_significant_peak_crit':0.8})
# f = f_list + f_list_changed
#
# df_modified = extract_data(f_list_changed)
#
# make_average_comp_per_onset_type(df_modified,['vot_kunduk'], add_to_save_name='_adjusted_kunduk_fspc-80',add_to_title='modified hyperparameters ')
#
# df1_modified = df_modified[~df_modified.index.str.contains('Cate') & ~df_modified.index.str.contains('Evan')]
# df2_modified = df_modified[df_modified.index.str.contains('Cate', case=True) | df_modified.index.str.contains('Evan', case=True)]
#
# df1_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_KUNDUK_params_adjusted_kunduk_fspc-80_S1-10.xlsx"))
# df2_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_KUNDUK_params_adjusted_kunduk_fspc-80_cate-evan.xlsx"))
#
# df1_modified.groupby(by='class')['vot_kunduk'].describe().to_excel(os.path.join(save_path, "KUNDUK_params_adjusted_kunduk_fspc-80_S1-10.xlsx"))
# df1_modified.groupby(by='class')['vot_kunduk'].describe().T.to_excel(os.path.join(save_path, "KUNDUK_params_adjusted_kunduk_fspc-80_S1-10_T.xlsx"))
#
# df2_modified.groupby(by='class')['vot_kunduk'].describe().to_excel(os.path.join(save_path, "KUNDUK_params_adjusted_kunduk_fspc-80_cate-evan.xlsx"))
# df2_modified.groupby(by='class')['vot_kunduk'].describe().T.to_excel(os.path.join(save_path, "KUNDUK_params_adjusted_kunduk_fspc-80_cate-evan_T.xlsx"))
#
# df = extract_data(f)
#
# make_box_plot_comp(df,['vot_kunduk'], add_to_save_name='_adjusted_kunduk_fspc-80', add_to_title='modified hyperparameters ')
#
# custom_hyperparameters={'F_0_bandpass_filter_range': 0.4,'first_significant_peak_crit':0.8,'freq_crit':0.15}
#
# f_list_changed = modify_parameters(f_list,parameter='vot_kunduk',
#         custom_hyperparameters={'F_0_bandpass_filter_range': 0.4,'first_significant_peak_crit':0.8,'freq_crit':0.15})
# f = f_list + f_list_changed
#
# df_modified = extract_data(f_list_changed)
#
# make_average_comp_per_onset_type(df_modified,['vot_kunduk'], add_to_save_name='_adjusted_kunduk_fspc-80_fr-40_fc_15',add_to_title='modified hyperparameters ')
#
# df1_modified = df_modified[~df_modified.index.str.contains('Cate') & ~df_modified.index.str.contains('Evan')]
# df2_modified = df_modified[df_modified.index.str.contains('Cate', case=True) | df_modified.index.str.contains('Evan', case=True)]
#
# df1_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_KUNDUK_params_adjusted_kunduk_fspc-80_fr-40_fc_15_S1-10.xlsx"))
# df2_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_KUNDUK_params_adjusted_kunduk_fspc-80_fr-40_fc_15_cate-evan.xlsx"))
#
# df1_modified.groupby(by='class')['vot_kunduk'].describe().to_excel(os.path.join(save_path, "KUNDUK_params_adjusted_kunduk_fspc-80_fr-40_fc_15_S1-10.xlsx"))
# df1_modified.groupby(by='class')['vot_kunduk'].describe().T.to_excel(os.path.join(save_path, "KUNDUK_params_adjusted_kunduk_fspc-80_fr-40_fc_15_S1-10_T.xlsx"))
#
# df2_modified.groupby(by='class')['vot_kunduk'].describe().to_excel(os.path.join(save_path, "KUNDUK_params_adjusted_kunduk_fspc-80_fr-40_fc_15_cate-evan.xlsx"))
# df2_modified.groupby(by='class')['vot_kunduk'].describe().T.to_excel(os.path.join(save_path, "KUNDUK_params_adjusted_kunduk_fspc-80_fr-40_fc_15_cate-evan_T.xlsx"))
#
# df = extract_data(f)
#
# make_box_plot_comp(df,['vot_kunduk'], add_to_save_name='_adjusted_kunduk_fspc-80_fr-40_fc_15', add_to_title='modified hyperparameters ')
#
# #_________________________________________________________________
# f_list_changed = modify_parameters(f_list,parameter='vot_kunduk',
#         custom_hyperparameters={'F_0_bandpass_filter_range': 0.4,'first_significant_peak_crit':0.8,'freq_crit':0.5})
# f = f_list + f_list_changed
#
# df_modified = extract_data(f_list_changed)
#
# make_average_comp_per_onset_type(df_modified,['vot_kunduk'], add_to_save_name='_adjusted_kunduk_fspc-80_fr-40_fc_50',add_to_title='modified hyperparameters ')
#
# df1_modified = df_modified[~df_modified.index.str.contains('Cate') & ~df_modified.index.str.contains('Evan')]
# df2_modified = df_modified[df_modified.index.str.contains('Cate', case=True) | df_modified.index.str.contains('Evan', case=True)]
#
# df1_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_KUNDUK_params_adjusted_kunduk_fspc-80_fr-40_fc_50_S1-10.xlsx"))
# df2_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_KUNDUK_params_adjusted_kunduk_fspc-80_fr-40_fc_50_cate-evan.xlsx"))
#
# df1_modified.groupby(by='class')['vot_kunduk'].describe().to_excel(os.path.join(save_path, "KUNDUK_params_adjusted_kunduk_fspc-80_fr-40_fc_50_S1-10.xlsx"))
# df1_modified.groupby(by='class')['vot_kunduk'].describe().T.to_excel(os.path.join(save_path, "KUNDUK_params_adjusted_kunduk_fspc-80_fr-40_fc_50_S1-10_T.xlsx"))
#
# df2_modified.groupby(by='class')['vot_kunduk'].describe().to_excel(os.path.join(save_path, "KUNDUK_params_adjusted_kunduk_fspc-80_fr-40_fc_50_cate-evan.xlsx"))
# df2_modified.groupby(by='class')['vot_kunduk'].describe().T.to_excel(os.path.join(save_path, "KUNDUK_params_adjusted_kunduk_fspc-80_fr-40_fc_50_cate-evan_T.xlsx"))
#
# df = extract_data(f)
#
# make_box_plot_comp(df,['vot_kunduk'], add_to_save_name='_adjusted_kunduk_fspc-80_fr-40_fc_50', add_to_title='modified hyperparameters ')

#________________joined corr

# f_list = load_all(['Cate','Evan','S01_','S02_','S003','S004','S005','S006','S007','S008','S009','S010'])#,'S04_','S05_','S06',S])
# f_list_changed = modify_parameters(f_list,parameter=['vot_mergel','vot_kunduk'], custom_hyperparameters={'first_significant_peak_option': "off"})
# df_modified = extract_data(f_list_changed)
# df_modified['voc'] = df_modified['voc'].astype(np.float64)
#
# df1_modified = df_modified[~df_modified.index.str.contains('Cate') & ~df_modified.index.str.contains('Evan')]
# df2_modified = df_modified[df_modified.index.str.contains('Cate', case=True) | df_modified.index.str.contains('Evan', case=True)]
# df1_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_JOINT_fspo-off_S1-10.xlsx"))
# df2_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_JOINT_fspo-off_cate-evan.xlsx"))
#
# f_list_changed = modify_parameters(f_list,parameter=['vot_mergel','vot_kunduk'], custom_hyperparameters={'first_significant_peak_crit':0.8})
# df_modified = extract_data(f_list_changed)
# df_modified['voc'] = df_modified['voc'].astype(np.float64)
#
# df1_modified = df_modified[~df_modified.index.str.contains('Cate') & ~df_modified.index.str.contains('Evan')]
# df2_modified = df_modified[df_modified.index.str.contains('Cate', case=True) | df_modified.index.str.contains('Evan', case=True)]
# df1_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_JOINT_fspc-80_S1-10.xlsx"))
# df2_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_JOINT_fspc-80_cate-evan.xlsx"))
#
# f_list_changed = modify_parameters(f_list,parameter=['vot_mergel','vot_kunduk'],
#         custom_hyperparameters={'F_0_bandpass_filter_range': 0.4,'first_significant_peak_crit':0.8,'freq_crit':0.15})
# df_modified = extract_data(f_list_changed)
# df_modified['voc'] = df_modified['voc'].astype(np.float64)
#
# df1_modified = df_modified[~df_modified.index.str.contains('Cate') & ~df_modified.index.str.contains('Evan')]
# df2_modified = df_modified[df_modified.index.str.contains('Cate', case=True) | df_modified.index.str.contains('Evan', case=True)]
# df1_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_MJOINT_fspc-80_fr-40_fc_15_S1-10.xlsx"))
# df2_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_JOINT_fspc-80_fr-40_fc_15_cate-evan.xlsx"))
#
# f_list_changed = modify_parameters(f_list,parameter=['vot_mergel','vot_kunduk'],
#         custom_hyperparameters={'F_0_bandpass_filter_range': 0.4,'first_significant_peak_crit':0.8,'freq_crit':0.5})
# df_modified = extract_data(f_list_changed)
# df_modified['voc'] = df_modified['voc'].astype(np.float64)
#
# df1_modified = df_modified[~df_modified.index.str.contains('Cate') & ~df_modified.index.str.contains('Evan')]
# df2_modified = df_modified[df_modified.index.str.contains('Cate', case=True) | df_modified.index.str.contains('Evan', case=True)]
# df1_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_JOINT_fspc-80_fr-40_fc_50_S1-10.xlsx"))
# df2_modified.corr(method='pearson').to_excel(os.path.join(save_path, "corr_JOINT_fspc-80_fr-40_fc_50_cate-evan.xlsx"))


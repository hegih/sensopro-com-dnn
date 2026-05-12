"""
Read csvs that have been exported from vicon. Expects 1 table per file, e.g., 01_B._traj.csv
"""

import math
import numpy as np
import pandas as pd
import os
import csv
import re
from typing import *
import xml.etree.ElementTree as ET


def parse_csv(path_to_csv):
    """
    :param path_to_csv:
    :return: a dataframe trajectory_df_with_frame with column names combined from the super- and subheader.
    column names: 'Frame', 'Sub Frame', ..., 'Rails4_Z', 'Trajectory Count_Count',
    alignment_rows: [0, ..., N-1] corresponding to frame numbers [1, ..., N]
    trajectory_df_with_frame attributes:
        trajectory_df_with_frame.attrs['file_name'] = 'WKM_01_Sp._traj.csv
        trajectory_df_with_frame.attrs['trial_name'] = 'WKM_01_Sp'
        trajectory_df_with_frame.attrs['export_type'] = '_traj', '_seg', ..., or ''
        trajectory_df_with_frame.attrs['units'] = ['', '', 'mm', ...]
    """
    # get headers and reader object from csv
    if not path_to_csv.endswith('.csv'):
        path_to_csv = f"{path_to_csv}.csv"
    with open(path_to_csv) as fp:
        reader = csv.reader(fp, delimiter=';')
        header_rows = [row for idx, row in enumerate(reader) if idx in range(2, 5)]  # read alignment_rows 2,3,4
        # header_row[0] is superheader (Rails1); [1] subheader (X,Y,Z); [2] unit (mm)

    # parse header alignment_rows to get column names and units
    superheader = ''
    column_names = []
    n = min([len(row) for row in header_rows])
    for idx in range(len(header_rows[1])):
        if header_rows[0][idx] != '':
            superheader = header_rows[0][idx]
        if superheader == '':
            seperator = ''
        else:
            seperator = '_'
        column_names.append(f"{superheader.split(':')[-1]}{seperator}{header_rows[1][idx]}")
    units = header_rows[2]

    # read table into dataframe
    df = pd.read_csv(path_to_csv, sep=';', skiprows=5, header=None)
    df.columns = column_names

    # Get metadata from file name and add it to the dataframe attributes
    file_name = os.path.split(path_to_csv)[-1]
    split_object = file_name.split('.')
    trial_name = split_object[0]
    if len(split_object) > 1:
        export_type = split_object[1]  # e.g., '_traj'
    else:
        export_type = ''
    df.attrs['file_name'] = file_name
    df.attrs['trial_name'] = trial_name
    df.attrs['export_type'] = export_type
    df.attrs['units'] = units

    # remove sub frame column
    df = df.drop(columns=['Sub Frame'])

    return df


# other helper functions:
# removed obsolete function: def add_trial_timestamps(df, sampling_rate=500, start_cirterion='FIN')

def recursive_path_callback(directory_prefix: str, callback, include_suffixes: Tuple = (), exclude_suffixes: Tuple = (),
                            skip_folders: Tuple = ()):
    """
    :param directory_prefix: the directory to search, usually the database root
    :param callback: a function to call with each file path
    :param include_suffixes: a tuple of suffixes, all files with different suffixes are skipped. ["_traj.csv"]
    :param exclude_suffixes: a tuple of suffixes, files and folders with one of these suffixes are ignored
    :param skip_folders: a tuple of prefixes, folders with these prefixes are skipped
    :return: None
    """

    filenames = os.listdir(directory_prefix)
    for filename in filenames:
        if filename.startswith('.'):  # ignore files and folders starting with a dot
            continue
        if filename.endswith(exclude_suffixes):
            continue
        filepath = os.path.join(directory_prefix, filename)
        if os.path.isdir(filepath):
            if filename.endswith(skip_folders):
                continue
            recursive_path_callback(filepath, callback, include_suffixes=include_suffixes,
                                    exclude_suffixes=exclude_suffixes,
                                    skip_folders=skip_folders)
            continue
        if not filename.endswith(include_suffixes):
            continue

        if os.path.isfile(filepath):
            callback(filepath)


def import_vsk(session_path):
    """
    Reads a vsk file and returns the anthropometric parameters that have been saved in Nexus.
    :param session_path: path to the mocap session folder or directly to the vsk file
    :return pcp_parameters: dictionary mapping attribute names to values
    """
    # get vsk file path
    if session_path.endswith('.vsk'):
        vsk_path = session_path
    else:
        filenames = os.listdir(session_path)
        vsk_candidates = [x for x in filenames if x.endswith('.vsk')]
        if len(vsk_candidates) < 1:
            raise AssertionError(f"no vsk file in {session_path}")
        elif len(vsk_candidates) > 1:
            print(f'two vsk files in {session_path}')
        vsk_path = os.path.join(session_path, vsk_candidates[0])
    xml_tree = ET.parse(vsk_path)
    root = xml_tree.getroot()
    parameters = root[0]
    symmetric_attributes = ('LegLength', 'AsisTrocanterDistance', 'KneeWidth', 'AnkleWidth',
                            'ShoulderOffset', 'ElbowWidth', 'WristWidth', 'HandThickness')
    set_attributes = tuple(attribute for suffix in symmetric_attributes
                           for attribute in ('Left' + suffix, 'Right' + suffix))

    set_attributes = ('Bodymass', 'Height', 'InterAsisDistance', 'LegLength') + set_attributes
    n_missing = len([(child.attrib['NAME'],) for child in parameters if
                               child.tag == 'StaticParameter' and child.attrib[
                                   'NAME'] in set_attributes and 'VALUE' not in child.attrib.keys()])
    if n_missing > 0:
        print("issue reading vsk in ", session_path)
    pcp_parameters = dict((child.attrib['NAME'], child.attrib['VALUE']) if 'VALUE' in child.attrib.keys()
                          else (child.attrib['NAME'], "Error") for child in parameters if
                          child.tag == 'StaticParameter' and child.attrib['NAME'] in set_attributes)
    return pcp_parameters


import os
from os.path import isfile, join
import numpy as np
import pandas as pd
import vicon_csv_import
from scipy.spatial.transform import Rotation
from collections import defaultdict
import random

###################################################################################################
# global variables
###################################################################################################
BASEPATH = "PATH/TO/DATASET"

STATIC_PATH = BASEPATH + "export_ai_static/"
# input and output columns
ALL_OUTPUT_COLUMNS = ['com_x', 'com_y', 'com_z']
OUTPUT_COLUMN = 'com_y'  # architecture only supports one output at a time
INPUT_COLUMNS = [  # comment columns to train without them
    'left_back_roll', 'left_back_pitch', 'left_back_yaw',
    'left_mid_roll', 'left_mid_pitch', 'left_mid_yaw',  # new
    'left_front_roll', 'left_front_pitch', 'left_front_yaw',
    'left_imu_roll', 'left_imu_pitch', 'left_imu_yaw',
    'left_imu_accel_x', 'left_imu_accel_y', 'left_imu_accel_z',
    'left_imu_gyro_x', 'left_imu_gyro_y', 'left_imu_gyro_z',
    'right_back_roll', 'right_back_pitch', 'right_back_yaw',
    'right_mid_roll', 'right_mid_pitch', 'right_mid_yaw',  # new
    'right_front_roll', 'right_front_pitch', 'right_front_yaw',
    'right_imu_roll', 'right_imu_pitch', 'right_imu_yaw',
    'right_imu_accel_x', 'right_imu_accel_y', 'right_imu_accel_z',
    'right_imu_gyro_x', 'right_imu_gyro_y', 'right_imu_gyro_z',
    'left_x', 'left_y', 'left_z',
    'right_x', 'right_y', 'right_z',
    'z_difference'
]
# print('number of inputs:', len(INPUT_COLUMNS))
# constants
REST_LENGTH = 1726.
MIN_XDISTANCE = 350.  # the minimum distance between foot center and anchor point


###################################################################################################
# loader functions to load csvs into dataframes, rename columns, and tare based on static data
###################################################################################################
def load_static_reference(static_directory, pcp_name, mid_segment=False, use_static_end=False):
    """
    Get reference data (i.e., offsets) for a specific participant.
    Usage: tared_left_accel_z = left_accel_z - offsets['left_imu_accel_z']
    use_static_end is intended for some XDN trials where IMUs have been incorrectly (re-)attached

    :param static_directory: directory containing all static trials
    :param pcp_name: first 6 characters of trial_name, e.g., 'ABD097'
    :param mid_segment: if true, use midsegment angular data in addition to imu angles
    :param use_static_end: if True use static_step_end instead of static_step (_start)
    :return: a pandas Series containing offsets. (can be used like a dictionary)
    """
    # Note: With f"{pcp_name}_static_tape", it would be possible to obtain a com_z offset too
    if use_static_end:
        static_trial_name = f"{pcp_name}_static_step_end"
    else:
        static_trial_name = f"{pcp_name}_static_step"
    static_imu_df = load_imu_df(static_directory, static_trial_name)
    static_seg_df = load_seg_df(static_directory, static_trial_name, mid_segment=mid_segment)
    offsets = pd.concat([static_seg_df.mean(axis=0, skipna=True),
                         static_imu_df.mean(axis=0, skipna=True)])
    return offsets  # type: pd.Series


# prepare mappers that change the names of segment dataframe columns
segment_map = {'LeftTapeBack': 'left_back', 'LeftTapeFront': 'left_front',
               'RightTapeBack': 'right_back', 'RightTapeFront': 'right_front',
               'LeftTapeMid': 'left_mid', 'RightTapeMid': 'right_mid'}
full_segment_map = {}
for x in segment_map.keys():
    for xyz in 'XYZ':
        # full_segment_map[f"{x}_R{'XYZ'[idx]}"] = f"{segment_map[x]}_{('roll', 'pitch', 'yaw')[idx]}"
        full_segment_map[f"{x}_R{xyz}"] = f"{segment_map[x]}_R{xyz}"


def load_seg_df(trial_directory, trial_name, mid_segment=False):
    seg_df = vicon_csv_import.parse_csv(os.path.join(trial_directory, trial_name + '._seg.csv'))
    seg_df = seg_df[list(full_segment_map.keys())]  # remove unwanted columns
    if not mid_segment:  # remove columns
        seg_df.drop(labels=[x for x in seg_df.columns if 'Mid' in x], axis=1, inplace=True)
    seg_df.rename(full_segment_map, axis=1, inplace=True)  # rename remaining columns
    return seg_df


def load_mod_df(trial_directory, trial_name):
    mod_df = vicon_csv_import.parse_csv(os.path.join(trial_directory, trial_name + '._mod.csv'))
    # replace column names by replacing entire strings
    mod_df.rename({'CentreOfMass_X': 'com_x', 'CentreOfMass_Y': 'com_y', 'CentreOfMass_Z': 'com_z'}, axis='columns',
                  inplace=True)
    return mod_df[['com_x', 'com_y', 'com_z']]


def load_imu_df(trial_directory, trial_name):
    imu_df = vicon_csv_import.parse_csv(os.path.join(trial_directory, trial_name + '._dev.csv'))
    # replace column names by replacing substrings
    imu_renaming = \
        {'TS-00814 - ': 'left_imu_', 'TS-00827 - ': 'right_imu_', 'TS-00827R - ': 'right_imu_',
         'Global Angle_x': 'RX', 'Global Angle_y': 'RY', 'Global Angle_z': 'RZ'}
    old_names = list(imu_renaming.keys())
    column_names = list(imu_df.columns)
    for idx in range(len(column_names)):
        for old in old_names:  # replace substrings
            column_names[idx] = column_names[idx].replace(old, imu_renaming[old])
    imu_df.columns = column_names

    # remove unnecessary columns
    imu_df.drop('Frame', axis=1, inplace=True)  # remove frame number
    for x in imu_df.columns:
        if 'High' in x or 'mag' in x:  # remove high acceleration marker and magnetometer data
            imu_df.drop(x, axis=1, inplace=True)

    return imu_df


def load_tared_df(trial_directory, trial_name, mid_segment=False, offsets=None, offsets_xdn_end=None):
    """
    Load a full trial and tare it using the provided offsets obtained from load_static_reference()
    offsets_xdn_end is only needed for the XDN participant

    :param trial_directory:
    :param trial_name:
    :param mid_segment: if true, use midsegment angular data in addition to imu angles
    :param offsets: preload offsets from load_static_reference to improve performance
    :param offsets_xdn_end: preload end offsets for xdn trials to improve performance
    :return: a dataframe containing tared tape data (without derived inputs)
    """

    trial_number = int(trial_name.split('_')[1])
    is_xdn = trial_name.startswith('XDN')
    is_sld = trial_name.startswith('SLD')
    if offsets is None:
        offsets = load_static_reference(STATIC_PATH, trial_name[:6], mid_segment=mid_segment)
    if is_xdn and offsets_xdn_end is None:
        offsets_xdn_end = load_static_reference(STATIC_PATH, trial_name[:6],
                                                mid_segment=mid_segment, use_static_end=True)

    seg_df = load_seg_df(trial_directory, trial_name, mid_segment=mid_segment)
    mod_df = load_mod_df(trial_directory, trial_name)
    imu_df = load_imu_df(trial_directory, trial_name)

    # fix problematic trials by mirroring raw imu data: Y=-Y; Z= -Z; X remains the same
    # for XDN:
    #   trials 1-12 (and static_start) have both IMU attached upside-down (affects warm-up and one-leg-stand trials)
    #   trials 13-34 have the left IMU correctly attached, the right IMU is still upside-down
    #   trials 35-40 (and static_end): both IMU are correctly attached (affects wave and squat trials)
    # for SLD:
    #   In all trials, the left IMU is attached correctly and the right IMU is upside-down
    both_upside_down = is_xdn and trial_number <= 12
    right_upside_down = is_sld or (is_xdn and trial_number <= 34)
    if both_upside_down:  # both IMUs are upside down
        prefix_subset = ['right_imu_', 'left_imu_']
    elif right_upside_down:  # right IMU is upside down
        prefix_subset = ['right_imu_']
    else:
        prefix_subset = []
    for prefix in prefix_subset:
        for suffix in ['accel_y', 'accel_z', 'gyro_y', 'gyro_z']:
            offsets[prefix + suffix] = -offsets[prefix + suffix]
            imu_df[prefix + suffix] = -imu_df[prefix + suffix]

    # acceleration and gyro offsets are simple subtractions
    cols = [f"{s}_imu_{ag}_{d}" for s in ('left', 'right') for ag in ('accel', 'gyro') for d in 'xyz']
    for column in cols:
        imu_df[column] = imu_df[column] - offsets[column]

    # combine df here, so that both imu and seg data are tared in one go
    full_df = pd.concat([seg_df, imu_df, mod_df], axis=1)

    # tare angluar data for front and back tape segments and BlueTrident IMUs
    if mid_segment:
        prefixes = [f"{s}_{p}" for s in ('left', 'right') for p in ('front', 'back', 'mid', 'imu')]
    else:
        prefixes = [f"{s}_{p}" for s in ('left', 'right') for p in ('front', 'back', 'imu')]
    suffix_map = {'RX': 'roll', 'RY': 'pitch', 'RZ': 'yaw'}
    for prefix in prefixes:
        segment_rxyz = [f'{prefix}_{d}' for d in ('RX', 'RY', 'RZ')]
        segment = full_df[segment_rxyz].to_numpy()
        seg_rot = Rotation.from_rotvec(segment, degrees=True)
        # if is_xdn, use offsets_xdn_end for left (>=13) or both left and right IMUs (>= 35)
        use_offset_end = is_xdn and ((prefix == 'right_imu' and not right_upside_down) or
                                     (prefix == 'left_imu' and not both_upside_down))
        if use_offset_end:
            offset = offsets_xdn_end[segment_rxyz].to_numpy()
        else:
            offset = offsets[segment_rxyz].to_numpy()
        reference_r = Rotation.from_rotvec(offset, degrees=True)
        tared_rotation = seg_rot * reference_r.inv()
        tared_euler = tared_rotation.as_euler('XYZ', degrees=True)
        full_df[segment_rxyz] = tared_euler
        # rename column to reflect that it now represents euler angles
        name_map = {name: name[:-2] + suffix_map[name[-2:]] for name in segment_rxyz}
        full_df.rename(name_map, axis=1, inplace=True)
    full_df.sort_index(axis=1, inplace=True)
    return full_df  # shape: Nx33 (3 output columns, 30 input columns, missing 7 input columns)


# Note: angular data in XDN and SLD trials is still wonky, mid_segment data is preferable

###################################################################################################
# data extraction: displacement_trigo (get xyz_displacement), assemble_data_dictionaries
###################################################################################################
# function to get z_displacement estimate from tape pitch
def displacement_trigo(pitch_front, pitch_back, exclude_negative_indices=False, degrees=True, ymode=False):
    """
    use model to get z_displacement (down is positive) and x_distance_from_front (further back is bigger)

    :param pitch_front: numpy array of pitch angles (can also accept positive or negative yaw angles)
    :param pitch_back: numpy array of pitch angles (can also accept positive or negative yaw angles)
    :param exclude_negative_indices: set negative angles to 0.
    :param degrees: expect pitch_front and pitch_back to be in degrees, default True. If False: expect radians
    :param ymode: Support for yaw input. If True, assume that all small angles (<0.2°) are zero
    :return: z_displacement, x_distance_from_front
    """
    rest_length = REST_LENGTH  # in mm, from measurements in lab
    if degrees:
        pitch_front = np.radians(pitch_front)
        pitch_back = np.radians(pitch_back)
    else:
        pitch_front = np.array(pitch_front)
        pitch_back = np.array(pitch_back)
    if exclude_negative_indices:
        invalid_indices = np.logical_or(pitch_front < 0, pitch_back < 0)
        pitch_front[invalid_indices], pitch_back[invalid_indices] = 1.e-5, 1.e-5  # will be treated as 0. by isclose
    else:  # exclude indices where signs don't match
        invalid_indices = np.sign(pitch_front) != np.sign(pitch_back)
        pitch_front[invalid_indices], pitch_back[invalid_indices] = 1.e-5, 1.e-5  # will be treated as 0. by isclose
    tan_front = np.tan(pitch_front)
    tan_back = np.tan(pitch_back)

    front_zero = np.isclose(tan_front, 0., atol=1.e-4)  # 1/x is not stable near x=0
    back_zero = np.isclose(tan_back, 0., atol=1.e-4)
    if ymode:  # ignore small angles: 0.4° corresponds to ~1cm y-displacement at x=1200mm
        front_zero = np.logical_or(front_zero, np.abs(pitch_front) < np.radians(0.1))
        back_zero = np.logical_or(back_zero, np.abs(pitch_back) < np.radians(0.1))

    is_zero = np.logical_or(front_zero, back_zero)
    z_displacement, distance_from_front = np.zeros(len(pitch_front)), np.zeros(len(pitch_front))
    z_displacement[is_zero] = 0.  # avoid division by zero
    z_displacement[~is_zero] = rest_length / (1 / tan_front[~is_zero] + 1 / tan_back[~is_zero])
    # get distance from front by rearranging tan(pitch_front) = z_pos / x_pos, use abs to avoid negative x
    distance_from_front[is_zero] = rest_length / 2  # returns tape-mid when angles close to 0.
    distance_from_front[~is_zero] = np.abs(z_displacement[~is_zero] / tan_front[~is_zero])
    distance_from_front = distance_from_front.clip(MIN_XDISTANCE, rest_length - MIN_XDISTANCE)
    return z_displacement, distance_from_front  # , x_position_check


def crop_trial(data_df, trial_name, batch_size=2048, where='both'):
    # for normal length trials (42s-48s when cropped 5000:-2500), go toward 45.056s (22’528 frames)
    # i.e., if longer than 45056ms, crop to 22528 frames. else: crop to multiple of 2048
    nan_indices = data_df.index[~np.all(~np.isnan(data_df), axis=1)]
    data_df_cropped = data_df.drop(nan_indices, axis=0)
    target_start, target_length = 5000, 11 * batch_size  # 22528
    # trials > 48s after fixed cropping (all three are 50.sth, all have no nans)
    # OUM242_13_W (exercise 7400:30000), LAM235_24_Ste (6700:30700), WKM258_25_Sta (6300:30800)
    if trial_name == 'OUM242_13_W':
        target_start = 7400
    if trial_name == 'LAM235_24_Ste':
        target_start = 6700
    if trial_name == 'WKM258_25_Sta':
        target_start = 6300
    target_end = target_start + target_length - 1
    if target_start in data_df_cropped.index and target_end in data_df_cropped.index:
        data_df_cropped = data_df_cropped.loc[target_start:target_end + 1]  # take index 5000:27528
    elif len(data_df_cropped) >= target_length:
        data_df_cropped = data_df_cropped.iloc[-target_length:]  # take the last 22528 frames
    elif (trial_name == 'UGD024_19_W' or
          (len(data_df_cropped) >= target_length - batch_size and not trial_name == 'FAD101_28_Ste')):
        # UGD_19_W is only about 44.4 seconds before cropping
        if not trial_name == 'UGD024_19_W': # known issue for UGD_19, shouldn't happen elsewhere
            print('\nPOTENTIAL ISSUE:', trial_name, "has length", len(data_df_cropped))
        data_df_cropped = data_df_cropped.iloc[-(target_length - batch_size):]  # take the last 20480 frames
    else:
        # trials < 40s (there are only 5)
        if trial_name == 'FAD101_28_Ste':  # in FAD101_28_Ste the index starts at 88577
            # FAD101_28_Ste had NaNs during the exercise. Crop to 6000:29200 then remove NaNs to be sure
            data_df_cropped = data_df.iloc[6000:29200]
            nan_indices = data_df_cropped.index[~np.all(~np.isnan(data_df_cropped), axis=1)]
            if len(nan_indices) > 0:
                print(f"\nfound {len(nan_indices)} NaNs in FAD_28_Ste\n")
            data_df_cropped = data_df_cropped.drop(nan_indices, axis=0)
        elif trial_name == 'CLN305_25_W' or trial_name == 'CLN305_26_W':
            # CLN_25_W and CLN_26_W have nans at start and end, too short for cropping
            pass
        elif trial_name == 'UGD024_18_W':
            # UGD_18_W and UGD_19_W end-nans (exercise start 4100: and 4000: respectively)
            data_df_cropped = data_df_cropped.iloc[4100:]
        elif trial_name == 'UGD024_19_W':
            data_df_cropped = data_df_cropped.iloc[4000:]
        else:
            raise AssertionError(f"trial {trial_name} slipped through the cracks")

    batch_offset = len(data_df_cropped) % batch_size
    if where == 'start':
        data_df_cropped = data_df_cropped.iloc[batch_offset:]
    elif where == 'end':
        data_df_cropped = data_df_cropped.iloc[:-batch_offset]
    else:  # where == 'both'
        data_df_cropped = data_df_cropped.iloc[batch_offset // 2:-batch_offset // 2]
    return data_df_cropped


def assemble_data_dictionaries(trial_names=None, file_path=BASEPATH):
    if trial_names is None:
        trial_names = [f for f in os.listdir(file_path) if isfile(join(file_path, f))]
        trial_names = [f.split('.')[0] for f in trial_names if f.endswith('seg.csv')]
    trial_names.sort()  # ensure that trials within participant are in order
    current_exercise = trial_names[-1].split('_')[-1]  # W, Ste, or Sta
    # participant_inputs = defaultdict(list)  # contains (pcp, [trial0, ..., trial3]) items
    # participant_outputs = defaultdict(list)  # items contain list of dicts with COM_XYZ]
    participant_dfs = defaultdict(list)  # inputs and outputs, items: (pcp, [trial0, ..., trial3]
    trial_stats = []

    mid_segment = any(['mid' in x for x in INPUT_COLUMNS])  # true if midsegments are included
    print("Assembling Trial Dictionaries:")
    for trial_name in trial_names:
        print(trial_name, end=', ')
        pcp_name = trial_name[:3]
        data_df = load_tared_df(file_path, trial_name, mid_segment=mid_segment)  # load tared data

        # apply trigo functions on left and right tape
        z_left, x_left = displacement_trigo(
            -data_df["left_front_pitch"], data_df["left_back_pitch"], ymode=False)
        y_left, _ = displacement_trigo(
            -data_df["left_front_yaw"], data_df["left_back_yaw"], ymode=True)
        z_right, x_right = displacement_trigo(
            -data_df["right_front_pitch"], data_df["right_back_pitch"], ymode=False)
        y_right, _ = displacement_trigo(
            -data_df["right_front_yaw"], data_df["right_back_yaw"], ymode=True)

        # add height estimation to dataset
        data_df["left_x"], data_df["left_y"], data_df["left_z"] = x_left, y_left, z_left
        data_df["right_x"], data_df["right_y"], data_df["right_z"] = x_right, y_right, z_right
        data_df["z_difference"] = z_left - z_right  # add tape height difference to dataset

        # cut trial and remove rows with nans
        # data_df_cut = data_df.iloc[5000:-2500, :]
        # nan_indices = data_df_cut.index[~np.all(~np.isnan(data_df_cut), axis=1)]
        # data_df_cut = data_df_cut.drop(nan_indices, axis=0)
        data_df_cropped = crop_trial(data_df, trial_name, batch_size=2048, where='start')

        # determine trial length
        clean_data_duration = data_df_cropped.shape[0] / 500
        trial_stats.append([trial_name, clean_data_duration])

        # save full df
        data_df_cropped.name = trial_name
        participant_dfs[pcp_name].append(data_df_cropped)

        # save trial durations as csv
        trial_stats_df = pd.DataFrame(trial_stats, columns=['trial', 'duration[s]'])
        trial_stats_df.to_csv(f"{BASEPATH}/stats_{current_exercise}.csv", sep=';')

    # split into training and test sets and save inputs and outputs in a single dataframe parquet
    training_path = f"{BASEPATH}/parquet/training_{current_exercise}/"
    test_path = f"{BASEPATH}/parquet/test_{current_exercise}/"
    os.makedirs(training_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    for (pcp, pcp_trials) in participant_dfs.items():
        trial_indices = list(range(0, len(pcp_trials)))
        test_index = random.randint(0, len(pcp_trials) - 1)
        pd.DataFrame.to_parquet(pcp_trials[test_index], f"{test_path}/{pcp}_test.pkl")
        trial_indices.remove(test_index)
        for (idx, num) in zip(trial_indices, range(len(pcp_trials) - 1)):
            pd.DataFrame.to_parquet(pcp_trials[idx], f"{training_path}/{pcp}_{num}.pkl")
    print('DONE')


def static_midpoint_stats(exports_directory="D:\\Dataset_export_gapfilled\\"):
    file_name = "static_step._traj.csv"
    x_mids = []
    y_mids = []
    for pcp in os.listdir(exports_directory):
        pcp_static_path = os.path.join(exports_directory, pcp, file_name)
        static_df = vicon_csv_import.parse_csv(pcp_static_path)
        x_mid = static_df[['LeftTapeMid3_X','RightTapeMid3_X']].median(axis=0).mean()
        y_mid = static_df[['LeftTapeMid3_Y','RightTapeMid3_Y']].median(axis=0).mean()
        print(pcp, np.round(x_mid, decimals=1), np.round(y_mid, decimals=1))
        if np.isnan(x_mid) or np.isnan(y_mid):
            print(pcp, "has nans")
        else:
            x_mids.append(x_mid)
            y_mids.append(y_mid)
    print(pd.DataFrame(x_mids).describe()) # x_mean, x_std =  np.nanmean(x_mids), np.nanstd(x_mids)
    print(pd.DataFrame(y_mids).describe()) # y_mean, y_std = np.nanmean(y_mids), np.nanstd(y_mids)





if __name__ == '__main__':
    if not os.path.exists(f"{BASEPATH}/parquet/"):
        os.makedirs(f"{BASEPATH}/parquet/")
    for exercise in ('Sta', 'Ste', 'W'):  # choose exercise resp. input set
        file_path = f"{BASEPATH}/export_ai_{exercise}/"
        assemble_data_dictionaries(trial_names=None, file_path=file_path)

# VOAT: Voice Onset Analysis Tool

VOAT is a Python software developed for semi-automatic segmentation of the vowel phonation onset from different modalities such as acoustic, EGG, or airflow signals. VOAT can detect the vowel onset on recordings with multiple phonations. Furthermore, the user can manually select the segments from the recording where the onset is present or apply a voice activity detection for automatic segmentation. VOAT provides filtering options to facilitate the detection and allows the saving of results in an Excel file for further processing. VOAT also allows batch processing to detect vowel onset on several files automatically.

## Requirements
The current version of VOAT is only supported for Windows operating systems.

## Installation
All necessary files to run VOAT are inside the "release" folder. To run the software, you must download the folder "release" and open the file "VOAT.exe" located in "release/VOAT.exe". 

## Supported file formats
-	WAV: It can be an audio file containing one or more channels. It can also be a WAV file containing audio and EGG in two channels.
-	ADICHT: An ADICHT file is a data file used by LabChart for Windows (version 7 and later).
-	CSV: VOAT can read the GAW signals exported from the GAT software (https://www.hno-klinik.uk-erlangen.de/phoniatrie/forschung/computational-medicine/gat-software/). The columns in the CSV file should be separated by “;” (semicolon).

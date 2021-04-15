# FinalYearProject-sinhui
This is a guide on how to build the system step by step

For those who not intend to rebuild the system from scratch, kindly skip to instruction 6.
For those who intend to only test the model with testing set, kindly skip to instrcution .
1. Download the youtube-dl package. For Windows user, please go to folder ForWindowsUser. For Mac user, please go to folder ForMacUser.

2. The package includes the videos download script, named download.bat (Windows) or download.sh (Mac).

3. For Windows user, kindly download the videos to local by clicking the download.bat. For Mac user, kindly open up a terminal and type the following:

		bash download.sh
	If the permission is denied, then do the following:
	
		chmod 777 download.sh

4. Split the videos into training and testing set.

5. Download trainlist01.txt, testlist01.txt, textfiles and testfiles folder.

6. Install the libraries required by executing the following commands:
		
		conda install python=3.8.5
		pip install opencv-python==4.5.1.48
		pip install ax-platform==0.1.20
		pip install tqdm==4.50.2
		conda install -c anaconda numpy
		conda install -c anaconda tensorflow-gpu
		conda install -c anaconda cudatoolkit
		conda install -c anaconda h5py
		conda install pillow

7. To rebuild the system, run the following files:
	
	data_preprocessing.ipynb or data_proprocessing.py
	feature_extraction.ipynb or feature_extraction.py

    This may take few hours to process, if you wish to save the time, then skip running the files and 	  do the following:

	Download the pickle folder containing:

		each_video_extra_frame.pickle
		each_video_frame.pickle
		n.pickle
		video_rnn.pickle
		X.pickle
		y.pickle
		
8. Next, run 
	

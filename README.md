# FinalYearProject-sinhui
This is a guide on how to build the system step by step

For those who intend to only test the model with testing set and saved model, kindly skip to instruction 11.
1. Download the youtube-dl package. For Windows user, please go to folder ForWindowsUser. For Mac user, please go to folder ForMacUser.

2. The package includes the videos download script, named download.bat (Windows) or download.sh (Mac).

3. For Windows user, kindly download the videos to local by clicking the download.bat. For Mac user, kindly open up a terminal and type the following:

		bash download.sh
	If the permission is denied, then do the following:
	
		chmod 777 download.sh

4. Split the videos into training and testing set.

5. Download the following:

		trainlist01.txt
		testlist01.txt
		items in textfiles folder
		items in testfiles folder

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

7. To rebuild the system, run the following files: (please get data_preprocessing file from the folder ForMacUser or ForWindowsUser)

		data_preprocessing.ipynb or data_proprocessing.py
		feature_extraction.ipynb or feature_extraction.py

    This may take few hours to process, if you wish to save the time, then skip running the files and do the following:

	Download the pickle folder containing:

		each_video_extra_frame.pickle
		each_video_frame.pickle
		n.pickle
		video_rnn.pickle
		X.pickle
		y.pickle
	
	Note: for X.pickle, the file size has exceeded 25 MB, kindly go and get at Google Drive. (Drive link is included at the bottom)
		
8. Next, to transform feature into RNN shape, run
	
		transformation_to_rnn.ipynb or transformation_to_rnn.py
		
9. Run the Ax experiment.

		ax_experiment.ipynb or ax_experiment.py
		
10. Train the model by running:

		training.ipynb or training.py
	
11. Run the following to preprocess the testing data:

		test_preprocessing.ipynb or test_preprocessing.py
	
	If you wish to skip processing the testing data, you can download the pickle files.
	
		X_testing.pickle
		y_testing.pickle
		
13. Download the saved model from GitHub.
		
14. Load and test the model by running:

		testing.ipynb or testing.py

Note: ax_client_snapshot contains details on the 250 trials using Ax experiment

For any further inquiries or issue faced while executing the files, kindly contact the following email:
sinhui1999@1utar.my

Drive link: https://drive.google.com/drive/folders/1LyCji2UXEcTU0RtSFMSlhBtlfIXtMnzJ?usp=sharing

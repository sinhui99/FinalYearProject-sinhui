# FinalYearProject-sinhui
This is a guide on how to build the system step by step. You can either use ipynb file or py file provided here.

For those who intend to only test the model with testing set and saved model, kindly skip to instruction 11.

If you have downloaded any pickle from GitHub or Drive, kindly put them in a folder named 'pickle' before running the program.

If you have downloaded any saved model, either hdf5 or the folder, kindly put them in the root you currently running your program.

1. Download the youtube-dl package. For Windows user, please go to folder ForWindowsUser. For Ubuntu user, please go to folder ForUbuntuUser.

2. The package includes the videos download script, named download.bat (Windows) or download.sh (Ubuntu).

3. For Windows user, kindly download the videos to local by clicking the download.bat. For Ubuntu user, kindly open up a terminal and type the following:

		bash download.sh
	If the permission is denied, then do the following:
	
		chmod 777 download.sh

4. Split the videos into training and testing set.
	
		Name training videos folder to videos				Name testing videos folder to testvideos 
		videos/jdezKogGKW0.mp4						testvideos/Xhu5Bz1xDf0.mp4
		videos/JjVO--nexcA.mkv						testvideos/XLWx0_I1qLQ.mp4
		videos/LbO-qvy9JlY.mp4						testvideos/_y3rFsvz8qQ.mkv
		videos/lpS9mYfvKIY.mp4						testvideos/3eDBuzDgLP0.mp4
		videos/p8fqqMi26EI.mp4						testvideos/46xKDm6OFHw.mp4
		videos/smBYcpMVeuo.mp4						testvideos/CFWSgo-ftrQ.mp4
		videos/WbN8QWa8fWA.mp4						testvideos/HktqGcJQA4w.mp4
		videos/weUenGaF8cs.mp4						testvideos/ms4hO0h1dOg.mp4
		videos/YXUmL753tYo.mp4						testvideos/xBUu1q-1KSI.mkv
		videos/Z5i6jdYkHcA.mp4
		videos/3fYpcapas0k.mp4
		videos/5dCpEzu1ka0.mkv
		videos/5OJfbYQtKtk.mp4
		videos/5V0LsqPzDTU.mp4
		videos/7Fau-IwbuJc.mp4
		videos/7jpruuvWK7E.mp4
		videos/9ifbXe4TsSc.mp4
		videos/eFDPJlvsrU8.mp4
		videos/ep0FVRUzweg.mp4
		videos/HfUwxxpzHrs.mp4
		
	Place the videos to their corresponding folder.
		
5. Download the following:

		trainlist01.txt
		testlist01.txt
		items in textfiles folder
		items in testfiles folder
	
	Note that if the videos extension have changed upon downloading, you have to modify the trainlist01.txt or testlist01.txt accordingly.

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

7. To rebuild the system, run the following files: (please get data_preprocessing file from the folder ForUbuntuUser or ForWindowsUser, move the file to root if you want to run it)

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
	
	Note: For X.pickle, the file size has exceeded 25 MB, kindly get at Drive at models/pickle/ folder. Drive link: https://drive.google.com/drive/folders/1LyCji2UXEcTU0RtSFMSlhBtlfIXtMnzJ?usp=sharing
		
8. Next, to transform feature into RNN shape, run
	
		transformation_to_rnn.ipynb or transformation_to_rnn.py
		
9. Run the Ax experiment.

		ax_experiment.ipynb or ax_experiment.py
		
	You can skip this step if you want to use the optimal hyperparameters available in the pickle folder to train the model. Go pickle folder to download the pickle file. 
		
		best_hyperparameters.pickle
		
10. Train the model by running:

		training.ipynb or training.py
	Upon complete running the file, the trained model will be saved. You may need to modify the name of the model and the name of the hdf5 file if needed.
	
11. Run the following to preprocess the testing data: (please get test_preprocessing file from the folder ForUbuntuUser or ForWindowsUser, move to file to root if you want to run it)

		test_preprocessing.ipynb or test_preprocessing.py
	
	If you wish to skip processing the testing data, you can download the pickle files from drive at models/pickle/ folder. Drive link: https://drive.google.com/drive/folders/1LyCji2UXEcTU0RtSFMSlhBtlfIXtMnzJ?usp=sharing
	
		X_testing.pickle
		y_testing.pickle
		
12. Download the saved model from Drive at models/ folder because the saved model has exceeded 25 MB. You need not to do this step if you are building from scratch. Drive link: https://drive.google.com/drive/folders/1LyCji2UXEcTU0RtSFMSlhBtlfIXtMnzJ?usp=sharing
	
	You can choose to use the saved model in folder or the hdf5 file to predict the class on testing set. By default, it is choosing the hdf5 file, kindly change it if needed.
		
13. Load and test the model by running:

		testing.ipynb or testing.py
	The default model to be loaded is model_1607, if you wish to test on different models, kindly change to the corresponding model name.

Note 1: ax_client_snapshot contains details on the 250 trials using Ax experiment

Note 2: train_new.csv and test_new.csv are just for your reference, it will be generated if you run the system step by step and if you are not, you do not need the files.

For any further inquiries or issue faced while executing the files, kindly contact the following email:
sinhui1999@1utar.my

Drive link: https://drive.google.com/drive/folders/1LyCji2UXEcTU0RtSFMSlhBtlfIXtMnzJ?usp=sharing

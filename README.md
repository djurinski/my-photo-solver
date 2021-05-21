# my-photo-solver
simple photo "equation" solver using OpenCV adn Keras/TesnorFlow
IMPORTANT: TO BE FINISHED

How to use?

	-import project in your favourite IDE (PyCharm in my case)
	-to se example calculation execute main
	-to calculate mathematical term in given photo, call evaluate(image) from main, where image is in form of a numpay array
	-if photo is stored in image format in memory, call function evaluate(cv.imread(path))
	-IMPORTANT: expression must be "pretty", writen on BLANK piece of paper, marker should be fine, NO SMUDGES, NO PAPER EGDES - should be cropped
	-when you execute main for the first time, 4 example photos from memory will be evaluated and result shown

-when evaluating, characterDetector displays every character it detected in the photo, to disable that comment (#) line 56 and 57 in characterDetector

	-model is stored in memory and loaded every time it makes prediction
	-model is trained in GoogleColab with GPU accelerator, with same parameters given in trainModel
	-model accuracy on test data is about 84% but dataset is rather bad, 3 sources (one source for brackets, myself as source for divided and to few of samples, and third for all others)
	-to train model again uncomment line 41-57 in prepareTrainig and execute, then execute trainModel, it will automaticly save
	-about accuracy, since dataset is bad (look nines or brackets) it doesn't predict good on my handwriting
	-final part is not finished, it's still implemented with eval() method

TO BE DONE:

	-some photos have problems with resolution, perhaps if minus sign is to narrow or number 1 also, it doesn't detect it, but if filter for contours is set too small it detects tiny smudges and noise, to work on that
	-find or make new dataset
	-impelent final method for evaluation, eval() == evil
	-refactor whole code

-in future, try to improve characterDetector to read on gridded paper etc.


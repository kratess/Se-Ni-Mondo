# Se Ni Mondo

"Se ni mondo" is a predictive AI model designed for sentiment analysis based on Italian tweets.

![640x360_1524829883906 LP_7772730](https://github.com/user-attachments/assets/1ab47899-06dc-4bc4-a21a-e7478c535251)

_“Se ni’ mondo esistesse un po’ di bene
e ognun si honsiderasse suo fratello
ci sarebbe meno pensieri e meno pene
e il mondo ne sarebbe assai più bello”_ - Pietro Pacciani

It was built for a competition of Artificial Intelligence during the last year of university.

This model uses a **Bag of words** strategy implemented through *Mean Squared Error* and *Gradient Descent* methods to modify the weights.
## How to

Firstly create a directory called ` epochs`

You will need to preprocess each file, both train and test csv files.

`python main.py --preprocess --csv_file train` - to preprocess the train.csv file

`python main.py --preprocess --csv_file test` - to preprocess the test.csv file

Now you can start training your model.

`python main.py --train --csv_file train --epochs <number>` - to train the model based on the train__preprocess.csv file output from the preprocessing step.

Each epoch will show you the **accuracy**, **f1-score** and **mse**. When you consider the values are fair enough you can evaluate your model on the test dataset.

`python main.py --test --csv_file test` - to test the model based on the test__preprocess.csv file output from the preprocessing step.

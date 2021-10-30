At first, I used one convolutional layer, one maxpooling layer, one hidden layer with 64 neurons and a dropout rate of 20%. And I found that the accuracy was like 5%. So I added one more convolutional layer, one more pooling layer, and more neurons to the hidden layer to 256, which increased the accuracy to about 85%.

However, even though the accuracy was decent, when I tried using the model to predict some images downloaded online, the accuracy of the prediction was far from satisfactory, which I thought was due to overfitting. So I increased the dropout rate to solve this problem. Finally, I got a pretty cool model.

In conclusion, processing the image twice using convolutional layer and pooling layer would be better than once; the dropout rate would better be around 40%; the neurons of the hidden layers would better be way larger than the neurons in the output layer.

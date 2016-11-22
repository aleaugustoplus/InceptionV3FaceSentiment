'''Code for fine-tuning Inception V3 for a new task.
   Start with Inception V3 network, not including last fully connected layers.
   Train a simple fully connected layer on top of these.
'''

import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout
import keras
import inception_v3 as inception
import sys,os

N_CLASSES = 3
IMSIZE = (299, 299)
train_dir = 'dataset/train'
test_dir = 'dataset/validation'


def generate_html_test(model):

	output=""" <!DOCTYPE html>
		<html>
			<head>
			<style>
			table, th, td {
				border: 1px solid black;
				border-collapse: collapse;
			}
			th, td {
				padding: 5px;
				text-align: left;
			}
			</style>
			</head>
			<body>

				<h2>Table of predictions on the test set:</h2>
				<table style="width:100%">
				  <tr>
					<th>Figure</th>
					<th>Class</th>
					<th>Classification Scores</th>
					<th>Class Predicted</th>
				  </tr>"""


	for d in os.listdir(test_dir):

			subdir=test_dir + "/" + d
			print "Directory:", subdir
			x=0
			for f in os.listdir(subdir):				

				output+='<tr><td><img src="%s" border="0"></td> \n' % (subdir + "/" + f)
				output+='<td>%s</td>\n' % d
				print "File:", subdir + "/" + f
			
				p=predict(model, subdir + "/" + f)
				output+='<td>%s</td>\n' % p
				output+='<td>%s</td><tr>\n' % np.argmax(p)


				x+=1
				if x>100:
					break

	output+=""" </table>
				</body>
				</html>
				"""
	file = open('test_results.html', 'w')
	file.write(output)
	file.close()

	return

def train():
	
	print "Loading model"
	model = generate_model()
	print "Model loaded"
	
	print 'Trainable weights'
	print model.trainable_weights


	# Data generators for feeding training/testing images to the model.
	print "Train ImageData"
	train_datagen = ImageDataGenerator(rescale=1./255)
	print "Train generator"
	train_generator = train_datagen.flow_from_directory(
		    train_dir,  # this is the target directory
		    target_size=IMSIZE,  # all images will be resized to 299x299 Inception V3 input
		    batch_size=20,
		    class_mode='categorical')

	print "Test image data"
	test_datagen = ImageDataGenerator(rescale=1./255)
	print "Teste generator"
	test_generator = test_datagen.flow_from_directory(
		    test_dir,  # this is the target directory
		    target_size=IMSIZE,  # all images will be resized to 299x299 Inception V3 input
		    batch_size=10,
		    class_mode='categorical')

	print "Model fit"
	model.fit_generator(
		    train_generator,
		    samples_per_epoch=308,
		    nb_epoch=15,
		    validation_data=test_generator,
		    verbose=2,
		    nb_val_samples=11)

	print "Saving the weights"
	model.save_weights('face_pretrain.h5')  # always save your weights after training or during training
	print "Weights succesfuly saved!"
        model.save('model.ker')
	generate_html_test(model)
	return model
def generate_model():
	# Start with an Inception V3 model, not including the final softmax layer.	
	base_model = inception.InceptionV3(weights='imagenet')
	print 'Loaded Inception model'


        for layer in base_model.layers:
                layer.trainable = True


	# Add on new fully connected layers for the output classes.
	x = Dense(32, activation='relu')(base_model.get_layer('flatten').output)
	x = Dropout(0.5)(x)
	predictions = Dense(N_CLASSES, activation='softmax', name='predictions')(x)

	model = Model(input=base_model.input, output=predictions)

	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


	# Show some debug output
	print (model.summary())

	return model

def load_weights(model):
#	model = keras.models.load_model('model.ker')
	model.load_weights('face_pretrain.h5')

	return model


def predict(model, img_path):

	#img_path = img_path = 'sport3/validation/hockey/img_2997.jpg'
	img = image.load_img(img_path, target_size=IMSIZE)
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = inception.preprocess_input(x)
	preds = model.predict(x)
	print('Predicted:', preds)

	return preds

def predict_class(model, img_path):

	pred=predict(model, img_path)
	
	return np.argmax(pred)

def main():	
	m=generate_model()	
	m=train()	
	#m=load_weights(m)
	#generate_html_test(m)

	return


if __name__ == "__main__":
    main()



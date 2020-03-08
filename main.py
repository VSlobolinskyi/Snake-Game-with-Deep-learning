from game import *
from nn_behavior import generate_training_data

from keras.models import Sequential
from keras.layers import Dense

pygame.init()
display=pygame.display.set_mode((display_width,display_height))
clock=pygame.time.Clock()

'''
LEFT -> button_direction = 0
RIGHT -> button_direction = 1
DOWN ->button_direction = 2
UP -> button_direction = 3
'''
'''
LEFT -> direction = -1
FORWARD -> direction = 0
RIGHT ->button_direction = 1
'''

model = Sequential()
model.add(Dense(250, input_dim=display_width*display_height, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(3,  activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

training_data_x, training_data_y = generate_training_data(model, display,clock)


# model.fit((np.array(training_data_x).reshape(-1,7)),( np.array(training_data_y).reshape(-1,3)), batch_size = 256,epochs= 3)
#
# model.save_weights('model.h5')
# model_json = model.to_json()
# with open('model.json', 'w') as json_file:
#     json_file.write(model_json)
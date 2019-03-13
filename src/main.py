import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print len(training_data)
print len(validation_data)
print len(test_data)

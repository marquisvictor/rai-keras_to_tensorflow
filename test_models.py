# Default graph is initialized when the library is imported
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import tensorflow.compat.v1 as tf
import numpy as np

# Load the image and convert it to array
def image_to_array(image_path):
    image = img_to_array(load_img(image_path, target_size=(700, 700))) / 255.
    image = np.expand_dims(image, axis=0)
    return image

# Load keras model (.hdf5 or .h5 model)
def predict_using_keras(image, keras_model_path):
    model = load_model(keras_model_path)
    prediction_result = model.predict(image)
    return prediction_result

def predict_using_tf(image, model_path, input_tensor_layer_name, output_tensor_layer_name):
    with tf.Graph().as_default() as graph: # Set default graph as graph

        with tf.Session() as sess:
            # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
            with tf.io.gfile.GFile(model_path,'rb') as f:

                # Set FCN graph to the default graph
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()

                # Import a graph_def into the current default Graph (In this case, the weights are (typically) embedded in the graph)
                tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    return_elements=None,
                    name="",
                    op_dict=None,
                    producer_op_list=None
                )

                # Print the name of operations in the session
                # for op in graph.get_operations():
                #     print("Operation Name :",op.name)         # Operation name
                #     print("Tensor Stats :",str(op.values()))     # Tensor name

                # INFERENCE Here
                l_input = graph.get_tensor_by_name(input_tensor_layer_name) # Input Tensor
                l_output = graph.get_tensor_by_name(output_tensor_layer_name) # Output Tensor

                #initialize_all_variables
                tf.global_variables_initializer()

                # Run Kitty model on single image
                Session_out = sess.run( l_output, feed_dict = {l_input : image})
                return Session_out

def main():
    image_path = 'sample_images/lippincott_brao_001.jpg'
    input_tensor_layer_name = 'input_1:0'
    output_tensor_layer_name = 'dense_2/Softmax:0'
    tf_model_path = 'tf_models/five_classes_tf_model.pb'
    keras_model_path = 'keras_models/olamide-model_all_classes_mobile_net_2_retrained.hdf5'

    image_array = image_to_array(image_path)

    keras_pred_result = predict_using_keras(image_array, keras_model_path)
    print('Keras Model output {}'.format(keras_pred_result[0]))

    tf_pred_result = predict_using_tf(image_array, tf_model_path, input_tensor_layer_name, output_tensor_layer_name)
    # output = np.argmax(tf_pred_result[0], axis=-1)
    print('TF Model output {}'.format(tf_pred_result))

if __name__== "__main__":
    main()
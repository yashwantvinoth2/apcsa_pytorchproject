import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ImageClassifier {

    public static void main(String[] args) throws Exception {

        int height = 224;
        int width = 224;
        int channels = 3;
        int numClasses = 10;
        int batchSize = 16;
        int epochs = 5;

        DataSetIterator trainLoader = new Cifar10DataSetIterator(batchSize, true);
        DataSetIterator valLoader = new Cifar10DataSetIterator(batchSize, false);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.001))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()

                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nIn(channels)
                        .nOut(32)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())

                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())

                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(64)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())

                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())

                .layer(new DenseLayer.Builder()
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())

                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())

                .setInputType(org.deeplearning4j.nn.conf.inputs.InputType.convolutional(
                        height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        model.setListeners(new ScoreIterationListener(10));

        for (int epoch = 0; epoch < epochs; epoch++) {

            System.out.println("Epoch " + (epoch + 1) + "/" + epochs);

            model.fit(trainLoader);

            var trainEval = model.evaluate(trainLoader);
            var valEval = model.evaluate(valLoader);

            System.out.println("Train Accuracy: " + trainEval.accuracy());
            System.out.println("Validation Accuracy: " + valEval.accuracy());

            trainLoader.reset();
            valLoader.reset();
        }

        System.out.println("Training Complete!");
    }
}

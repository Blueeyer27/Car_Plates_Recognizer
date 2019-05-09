using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bachelor_Evgeniy
{
    class NeuralNetwork
    {
        public void Teach(double[] input, double[] desiredOutput)
        {
            GetNextInput(input);
            CountFuncs();
            CountErrors(desiredOutput);
            CountFullError(desiredOutput);
            UpdateWeights();


        }
        public void Test(double[] input)
        {
            GetNextInput(input);
            CountFuncs();
        }
        double alpha;
        public double fullError;
        public void CountFullError(double[] desiredOutput)
        {
            double error = 0;
            for (int i = 0; i < errors.Length; i++)
            {
                error += Math.Pow((desiredOutput[i] - neuronActivationFuncs[i]), 2);
            }
            this.fullError = error;

        }
        public NeuralNetwork(int inputCount, int neuronCount, double alpha)
        {
            this.alpha = alpha;
            fullError = 1;
            Random rand = new Random();
            inputs = new double[inputCount];
            neuronActivationFuncs = new double[neuronCount];
            weights = new double[inputCount, neuronCount];
            errors = new double[neuronCount];
            for (int i = 0; i < inputCount; i++)
                for (int j = 0; j < neuronCount; j++)
                {
                    inputs[i] = 0;
                    neuronActivationFuncs[j] = 0;
                    weights[i, j] = rand.NextDouble() * 2 - 1;
                    errors[j] = 1;
                }
        }

        public void UpdateWeights()
        {
            for (int i = 0; i < inputs.Length; i++)
                for (int j = 0; j < neuronActivationFuncs.Length; j++)
                {
                    weights[i, j] += CountDerivative(neuronActivationFuncs[j]) * errors[j] * inputs[i];
                }
        }

        public double CountDerivative(double activationFunc)
        {
            return -alpha * activationFunc * (1 - activationFunc);
        }
        public double CountNeuronOutput(int neuronIndex)
        {
            double output = 0;
            for (int i = 1; i < inputs.Length; i++)
            {
                output += inputs[i] * weights[i, neuronIndex];
            }
            output += inputs[0] * weights[0, neuronIndex];//В нулевом элемента всегда единица.
            return output;
        }

        public double CountActivationFunc(int neuronIndex)
        {
            return 1f / (1 + Math.Exp(-alpha * CountNeuronOutput(neuronIndex)));
        }

        public void CountFuncs()
        {
            for (int i = 0; i < neuronActivationFuncs.Length; i++)
                neuronActivationFuncs[i] = CountActivationFunc(i);
        }

        public void GetNextInput(double[] inputs)
        {
            Array.Copy(inputs, this.inputs, inputs.Length);
        }

        public void CountErrors(double[] desiredResult)
        {
            for (int i = 0; i < neuronActivationFuncs.Length; i++)
                errors[i] = neuronActivationFuncs[i] - desiredResult[i];
        }

        double[] inputs;
        double[,] weights;
        public double[] neuronActivationFuncs;
        public double[] errors;
    }
}

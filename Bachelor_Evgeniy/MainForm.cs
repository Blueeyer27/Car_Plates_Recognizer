using OpenCvSharp;
using System;
using System.Windows.Forms;
using System.Windows;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Drawing;
using System.Linq;
using System.Threading;
using System.Xml.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace Bachelor_Evgeniy
{
    public partial class MainForm : Form
    {
        private const int RECOGNIZE_DIGITS = 0;
        private const int RECOGNIZE_LETTERS = 1;
        private const int RECOGNIZE_ALL = 2;

        private const String AVAILABLE_SYMBOLS = "ABCEHKMOPTXY0123456789";
        private List<int> letterPos = new List<int>();
        private List<PictureBox> symbolPicBoxes = new List<PictureBox>();
        private String trainingDataPath = "../../TrainingData/";
        private Dictionary<Char, double[,]> trainedModels = new Dictionary<Char, double[,]>();

        private NeuralNetwork neuralNetwork;

        public MainForm()
        {
            InitializeComponent();
            neuralNetwork = new NeuralNetwork(3251, 22, 0.05);
        }

        private void MainForm_Load(object sender, EventArgs e)
        {
            letterPos.Add(0);
            letterPos.Add(4);
            letterPos.Add(5);

            symbolPicBoxes.Add(symbolPictureBox1);
            symbolPicBoxes.Add(symbolPictureBox2);
            symbolPicBoxes.Add(symbolPictureBox3);
            symbolPicBoxes.Add(symbolPictureBox4);
            symbolPicBoxes.Add(symbolPictureBox5);
            symbolPicBoxes.Add(symbolPictureBox6);
            symbolPicBoxes.Add(symbolPictureBox7);
            symbolPicBoxes.Add(symbolPictureBox8);
            symbolPicBoxes.Add(symbolPictureBox9);
        }

        private void TrainNeuralNetwork()
        {
            int correctCount = 0;
            Random rand = new Random();
            int i = 0;
            DateTime started = DateTime.Now;

            while (correctCount < 100)
            {
                if (neuralNetwork.fullError < 0.1)
                    correctCount++;
                else correctCount = 0;
                i++;

                var symbol = rand.Next() % 22;

                var fileName = String.Format("{0}_{1}.jpg",
                    AVAILABLE_SYMBOLS[symbol],
                    String.Format("{0:D3}", rand.Next()%10 + 1));

                var image = Image.FromFile(trainingDataPath + fileName);
                double[] nextInput = GetInputFromImage((Bitmap)image);
                double[] desiredOutput = new double[22];
                desiredOutput[symbol] = 1;

                neuralNetwork.Teach(nextInput, desiredOutput);
                //printErrors(net.errors);
            }

            DateTime ended = DateTime.Now;
            TimeSpan dateDiff = ended.Subtract(started);
            //difference = "Teaching time in seconds: " + Math.Round(dateDiff.TotalSeconds, 3);
            //gTextBoxString = printErrors(net.errors);

            trainLabel.Text = "Нейронная сеть успешно обучена. Директория обучающей выборки: " + trainingDataPath + " Файл обученных эталонов: trained.mdlo";
            trainLabel.ForeColor = Color.Green;
        }

        private void TrainModels()
        {
            DirectoryInfo trainingDir = new DirectoryInfo(trainingDataPath);

            foreach (char symbol in AVAILABLE_SYMBOLS)
            {
                FileInfo[] trainImgFiles = trainingDir.GetFiles(symbol + "*.jpg");
                foreach (var file in trainImgFiles)
                {
                    using (var img = Image.FromFile(trainingDataPath + file.Name))
                    {
                        if (!trainedModels.ContainsKey(symbol))
                            trainedModels.Add(symbol, CreateZeroMatrix(img.Width, img.Height));

                        double[,] currMat;
                        trainedModels.TryGetValue(symbol, out currMat);
                        UpdateWeights(symbol, (Bitmap)img, trainImgFiles.Length);
                    }

                    //MessageBox.Show(MatrixToString(symbol, 50, 65));
                }
            }

            try
            {
                using (FileStream str = new FileStream("trained.mdlo", FileMode.Create))
                {
                    BinaryFormatter binaryFormatter = new BinaryFormatter();
                    binaryFormatter.Serialize(str, trainedModels);
                }
            }
            catch (Exception ex)
            {
                trainLabel.Text = "Ошибка при сохранении эталонов: " + ex.Message;
                trainLabel.ForeColor = Color.Red;
                return;
            }


            trainLabel.Text = "Все эталоны успешно обучены. Директория обучающей выборки: " + trainingDataPath + " Файл обученных эталонов: trained.mdlo";
            trainLabel.ForeColor = Color.Green;
        }

        private double[] GetInputFromImage(Bitmap img)
        {
            double[] input = new double[img.Height * img.Width + 1];
            input[0] = 1; //BIAS

            for (var i = 0; i < img.Height; i++)
                for (var j = 0; j < img.Width; j++)
                    input[i * img.Width + j + 1] = img.GetPixel(j, i).R > 0 ? 0d : 1d;

            return input;
        }

        private char RecognizeSymbol(Bitmap img, int state = RECOGNIZE_ALL)
        {
            var max = -99999999d;
            char answer = (char)0;
            foreach (KeyValuePair<char, double[,]> model in trainedModels)
            {
                if (state == RECOGNIZE_DIGITS && !Char.IsDigit(model.Key)) continue;
                if (state == RECOGNIZE_LETTERS && !Char.IsLetter(model.Key)) continue;

                var currValue = 0d;

                for (var i = 0; i < img.Width; i++)
                    for (var j = 0; j < img.Height; j++)
                        currValue += model.Value[i, j] * (img.GetPixel(i, j).R > 0 ? 0d : 1d);

                if (currValue > max)
                {
                    max = currValue;
                    answer = model.Key;
                }
            }

            if (answer == 0)
                return '_';

            return answer;
        }

        private void UpdateWeights(char matName, Bitmap img, int imgCount)
        {
            for (var i = 0; i < img.Height; i++)
            {
                for (var j = 0; j < img.Width; j++)
                    trainedModels[matName][j, i] += (img.GetPixel(j, i).R > 0 ? -1d : 1d) / imgCount; //TODO: do we need -1? Or 0 is better?
            }
        }

        private String MatrixToString(Char matName, int rows, int cols)
        {
            String matrixStr = "";

            for (var i = 0; i < cols; i++)
            {
                for (var j = 0; j < rows; j++)
                    matrixStr += trainedModels[matName][j, i] + " ";
                matrixStr += "\n";
            }

            return matrixStr;
        }

        private double[,] CreateZeroMatrix(int rows, int cols)
        {
            var matrix = new double[rows, cols];
            for (var i = 0; i < cols; i++)
                for (var j = 0; j < rows; j++)
                    matrix[j, i] = 0;

            return matrix;
        }

        private Image CvMatToImg(Mat img)
        {
            using (var ms = new MemoryStream(img.ToBytes()))
                return Image.FromStream(ms);
        }

        private String GenerateRandomName(int size)
        {
            var chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
            var stringChars = new char[size];
            var random = new Random();

            for (int i = 0; i < stringChars.Length; i++)
            {
                stringChars[i] = chars[random.Next(chars.Length)];
            }

            return new String(stringChars);
        }

        private Mat MakePictureBlurred(Mat picture)
        {
            Mat gray = new Mat();
            Mat blurred = new Mat();

            Cv2.CvtColor(picture, gray, ColorConversionCodes.BGR2GRAY);
            Cv2.GaussianBlur(gray, blurred, new OpenCvSharp.Size(5, 5), 0);

            return blurred;
        }

        private List<Rect> SortSymbolsByLocation(List<Rect> symbols)
        {
            Rect temp;

            for (int write = 0; write < symbols.Count; write++)
            {
                for (int sort = 0; sort < symbols.Count - 1; sort++)
                {
                    if (symbols[sort].Location.X > symbols[sort + 1].Location.X)
                    {
                        temp = symbols[sort + 1];
                        symbols[sort + 1] = symbols[sort];
                        symbols[sort] = temp;
                    }
                }
            }

            return symbols;
        }

        private List<Rect> RemoveInnerRects(List<Rect> rects)
        {
            List<int> indToDelete = new List<int>();

            for (var i = 0; i < rects.Count - 1; i++)
                for (var j = i + 1; j < rects.Count; j++)
                {
                    var currRectLoc = rects[i].Location;
                    var nextRectLoc = rects[j].Location;

                    // on X                    
                    var nearestInd = (rects[i].Location.X < rects[j].Location.X ? i : j);
                    var nearestRect = rects[nearestInd];

                    var toDelete = nearestInd == i ? j : i;
                    if (Math.Abs(currRectLoc.X - nextRectLoc.X) < nearestRect.Width &&
                        Math.Abs(currRectLoc.Y - nextRectLoc.Y) < nearestRect.Height && !indToDelete.Contains(toDelete))
                    {
                        indToDelete.Add(toDelete);
                        continue;
                    }
                }

            indToDelete.Sort();
            var deleted = 0;
            foreach (var i in indToDelete)
            {
                rects.RemoveAt(i - deleted);
                deleted++;
            }

            return rects;
        }

        private Mat CreateMatrix(OpenCvSharp.Size size, MatType matType, int mult = 1)
        {
            Mat mtx = new Mat(size, matType);
            for (int w = 0; w < mtx.Rows; ++w)
            {
                for (int h = 0; h < mtx.Cols; ++h)
                    mtx.Set<int>(w, h, 1 * mult);
            }

            return mtx;
        }

        private void recognizeButton_Click(object sender, EventArgs e)
        {
            if (openFileToRecDialog.ShowDialog() != DialogResult.OK) return;

            ClearOldData();
            //RecognizeSymbol((Bitmap)Image.FromFile(openFileToRecDialog.FileName));

            //var img = Cv2.ImRead(@"../../TestData/car_1.jpg");
            var img = Cv2.ImRead(openFileToRecDialog.FileName);
            if (img.Empty())
            {
                resultLabel.Text = "Image not found.";
                resultLabel.ForeColor = Color.Red;
                return;
            }

            origImgPictureBox.Image = CvMatToImg(img);
            //Cv2.ImShow("original", img);
            //CreateMatrix(img.Size(), 5);

            // Getting plate number
            var digitCascade = new CascadeClassifier(@"../../Cascades/haarcascade_russian_plate_number.xml");
            var plates = digitCascade.DetectMultiScale(img);

            Mat plateNum = GetPlate(plates, img);
            if (plateNum == null) return;

            List<Rect> symbols = GetSymbolRects(plateNum);

            symbols = RemoveInnerRects(symbols);
            symbols = SortSymbolsByLocation(symbols);

            var symbolNum = 0;
            var answer = "";
            foreach (var symbol in symbols)
            {
                //Cv2.Rectangle(plateNum, symbol, Scalar.Blue, 2);
                //Cv2.ImShow("img", plateNum);

                OpenCvSharp.Size symbolSize = new OpenCvSharp.Size(50, 65);
                var symbolImg = plateNum.Clone(symbol).Resize(symbolSize);
                var blSymbol = MakePictureBlurred(symbolImg);

                var thresh = new Mat();

                Cv2.Threshold(blSymbol, thresh, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
                var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(1, 5));
                Cv2.MorphologyEx(thresh, thresh, MorphTypes.Open, kernel);

                //if (!thresh.SaveImage(trainingDataPath + GenerateRandomName(7) + ".jpg"))
                //{
                //    MessageBox.Show("Something gone wrong!");
                //}

                //Cv2.ImShow("symbol" + symbolNum, thresh);
                try
                {
                    symbolPicBoxes[symbolNum].Image = (Bitmap)CvMatToImg(thresh);
                }
                catch (Exception)
                {
                    break;
                }

                var state = letterPos.Contains(symbolNum) ? RECOGNIZE_LETTERS : RECOGNIZE_DIGITS;
                answer += RecognizeSymbol((Bitmap)CvMatToImg(thresh), state);
                //MessageBox.Show(rect.Location + "  " + rect.Size + "  " + rect.Width + ":" + rect.Height);

                symbolNum++;
            }

            resultLabel.Text = "Номер машины: " + answer;
            resultLabel.ForeColor = Color.Green;
            //Cv2.ImShow("img", plateNum);
            //Cv2.ImShow("edged", blurred);
        }

        private void ClearOldData()
        {
            foreach (var pictureBox in symbolPicBoxes)
                pictureBox.Image = null;
        }

        private List<Rect> GetSymbolRects(Mat plateNum)
        {
            Mat edgedPlate = new Mat();
            var blPlate = MakePictureBlurred(plateNum);
            Cv2.Canny(blPlate, edgedPlate, 50, 200);

            var contours = new Mat[50];
            Mat hierarchy = new Mat();

            Cv2.FindContours(edgedPlate, out contours, hierarchy, RetrievalModes.List, ContourApproximationModes.ApproxSimple);


            List<Rect> symbols = new List<Rect>();
            foreach (var cnt in contours)
            {
                var rect = Cv2.BoundingRect(cnt);
                if (rect.Width < 15 || rect.Width > 40 || rect.Height < 30 || rect.Height > 60)
                    continue;

                symbols.Add(rect);
            }

            return symbols;
        }

        private Mat GetPlate(Rect[] plates, Mat img)
        {
            Mat plateNum = null;
            foreach (var plate in plates)
            {
                //Cv2.Rectangle(img, plate, Scalar.Red, 2);
                OpenCvSharp.Size plateSize = new OpenCvSharp.Size(400, 120);
                plateNum = img.Clone(plate).Resize(plateSize);
                //Cv2.ImShow("Plate", plateNum);

                platePictureBox.Image = (Bitmap)CvMatToImg(plateNum);
                break; //TODO: choose the best one
            }

            return plateNum;
        }

        private void trainButton_Click(object sender, EventArgs e)
        {
            TrainNeuralNetwork();
            //TrainModels();
        }

        private void loadEthalonsButton_Click(object sender, EventArgs e)
        {
            try
            {
                using (FileStream str = new FileStream("trained.mdlo", FileMode.Open))
                {
                    BinaryFormatter binaryFormatter = new BinaryFormatter();
                    trainedModels = (Dictionary<Char, double[,]>)binaryFormatter.Deserialize(str);
                }
            }
            catch (Exception ex)
            {
                trainLabel.Text = "Ошибка при считывании эталонов: " + ex.Message;
                trainLabel.ForeColor = Color.Red;
                return;
            }

            trainLabel.Text = "Эталоны успешно считаны с файла: trained.mdlo";
            trainLabel.ForeColor = Color.Green;
        }
    }
}

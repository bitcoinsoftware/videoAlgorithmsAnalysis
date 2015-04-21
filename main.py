from gui import *
from Display import *
from functools import partial 
import os, cv2, sys
import pickle
from scipy import stats

algorithmsFolder = 'Algorithms'
classifiersFolder= 'Classifiers'
output = []


def connect(ui):
    _fillAlgorithmList(ui)
    _initiatePlot(ui)
    #initiate training film samples
    ui.filmSamples={}
    ui.resultDict={}
    for i in range(ui.listWidget_2.count()):
        environmentName = str(ui.listWidget_2.item(i).text())
        ui.filmSamples[environmentName]=[]
    QtCore.QObject.connect(ui.toolButton,QtCore.SIGNAL('clicked()'), partial(chooseFile, ui))

    QtCore.QObject.connect(ui.pushButton_5,QtCore.SIGNAL('clicked()'), partial(saveToFile, ui))
    QtCore.QObject.connect(ui.pushButton,QtCore.SIGNAL('clicked()'), partial(runAnalysis, ui))
    QtCore.QObject.connect(ui.toolButton_2,QtCore.SIGNAL('clicked()'), partial(addSample, ui))
    QtCore.QObject.connect(ui.listWidget_2,QtCore.SIGNAL('currentRowChanged(int)'), partial(fillSamplesList, ui))
    QtCore.QObject.connect(ui.pushButton_6,QtCore.SIGNAL('clicked()'), partial(addToTraining, ui))
    QtCore.QObject.connect(ui.pushButton_3,QtCore.SIGNAL('clicked()'), partial(generateTrainingSamples, ui))
    QtCore.QObject.connect(ui.pushButton_2,QtCore.SIGNAL('clicked()'), partial(loadTrainingData, ui))
    QtCore.QObject.connect(ui.pushButton_7,QtCore.SIGNAL('clicked()'), partial(clearFileList, ui.listWidget_3))
    QtCore.QObject.connect(ui.pushButton_9,QtCore.SIGNAL('clicked()'), partial(clearFileList, ui.listWidget_6))
    QtCore.QObject.connect(ui.pushButton_10,QtCore.SIGNAL('clicked()'), partial(clearTrainingFilmSamples, ui))
    QtCore.QObject.connect(ui.pushButton_11,QtCore.SIGNAL('clicked()'), partial(addToClassifiersList, ui))

    QtCore.QObject.connect(ui.toolButton_3,QtCore.SIGNAL('clicked()'), partial(chooseTestFile, ui))
    QtCore.QObject.connect(ui.pushButton_4,QtCore.SIGNAL('clicked()'), partial(runTest, ui))
    QtCore.QObject.connect(ui.listWidget_7,QtCore.SIGNAL('currentRowChanged(int)') , partial(printAlgorithmResults, ui))

def chooseFile(ui):
    fileName= QtGui.QFileDialog.getOpenFileName(None,'Open file','data/input')
    ui.toolButton.setText(fileName)
    ui.toolButton.setStyleSheet('QPushButton {font: bold 14px;}')
    ui.listWidget_3.insertItem(0, fileName)

def addToClassifiersList(ui):
    print "Dodaje ", ui.listWidget_4.currentItem().text()
    param = ""
    if ui.checkBox_3.isChecked():
        param = str(ui.spinBox_4.value())
        ui.checkBox_3.setChecked(False)
    ui.listWidget_8.insertItem(0, ui.listWidget_4.currentItem().text()+","+param)

def clearFileList(qlistWidget):
    qlistWidget.clear()

def clearTrainingFilmSamples(ui):
    for i in range(ui.listWidget_2.count()):
        environmentName = str(ui.listWidget_2.item(i).text())
        ui.filmSamples[environmentName]=[]
    ui.listWidget_5.clear()

def saveToFile(ui):
    global output
    tempOutput = []
    keyName = str(ui.listWidget_7.currentItem().text())
    for measurement in output:
        tempOutput.append(measurement[keyName])

    fileName = QtGui.QFileDialog.getSaveFileName(None,'Save to file','data/output')
    with open(fileName, "w") as f:
        #f.write(str(output))
        pickle.dump(output, f)

def runAnalysis(ui):
    global output
    output =[]
    fileUrls=[]
    for n in range(ui.listWidget_3.count()):
        fileUrls.append(str(ui.listWidget_3.item(n).text()))

    if ui.checkBox.isChecked():
        for i in range(ui.listWidget.count()):
            algorithm = str(ui.listWidget.item(i).text().replace('.py',''))
            print algorithm
            _analyse(ui, fileUrls, algorithm)
            #_writeDistribution(algorithm, distX, distY)
    else:
        algorithm = str(ui.listWidget.currentItem().text()).replace('.py','')
        _analyse(ui, fileUrls, algorithm)
        #_writeDistribution(algorithm, distX, distY)

def generateTrainingSamples(ui):
    if ui.listWidget_6.count() >0:
        display1 = Display(ui.label_17)
        #load the selected algorithms
        algorithms={}#, samples ={},{}
        outputTrainingDataName =''
        for i in range(ui.listWidget_6.count()):
            algorithmName = str(ui.listWidget_6.item(i).text().replace('.py',''))
            algorithm = _loadAlgorithmClass(algorithmsFolder, algorithmName)()
            algorithms[algorithmName] = algorithm
            #outputTrainingDataName+=str(algorithmName+'#')

        period = ui.spinBox.value()
        #print algorithms
        trainingSamples, responses =[], []
        for key in algorithms.keys():
            #iterate through environments
            for n in range(ui.listWidget_2.count()):
                environment = str(ui.listWidget_2.item(n).text())
                #iterate through environment sample videos
                filmUrlList = ui.filmSamples[environment]
                for filmUrl in filmUrlList:
                    print filmUrl
                    video, i = cv2.VideoCapture(filmUrl), 0
                    ret, frame = video.read()
                    #initiate the analysis algorithms
                    for algorithmName in algorithms.keys():
                        algorithms[algorithmName]._initiateRegisters(frame)
                    #iterate via frames in a film sample
                    while ret:
                        i += 1
                        display1.display(frame)
                        #for each algorithm
                        sample =[]
                        #for key in algorithms.keys():
                        algorithms[key].analyze(frame, period)
                        if i % period==0:
                            #distX, distY = algorithms[key].getDistribution(period)
                            #weightCenter = algorithms[key].getWeightCenter(distX[1], distY[1])
                            #sample = algorithms[key].getCharacteristicPointsVector()
                            #sample.append(weightCenter[0])
                            #sample.append(weightCenter[1])
                            if ui.radioButton_2.isChecked():
                                sample = np.array(algorithms[key].keyPointsVector).astype(np.float32)
                                #samples[environment].append(sample)
                            else:
                                sample=algorithms[key].keyPointsVector
                            trainingSamples.append(sample)
                            responses.append(n)


                        ret, frame = video.read()
            if ui.radioButton_2.isChecked():
                fileName = 'data/trainingData/'+ key +'.dat'
                with open(fileName, "w") as f:
                    print "Saved to ", fileName
                    pickle.dump([trainingSamples,responses],f)
            else:
                fileName = 'data/trainingData/'+key+ '.txt'
                with open(fileName, "w") as f:
                    print "Saved to ", fileName
                    f.write(str(trainingSamples)+"\nRESPONSES\n"+str(responses))


def loadTrainingData(ui):
    fileName = str(QtGui.QFileDialog.getOpenFileName(None,'Select file','data/trainingData/'))
    if len(fileName)>0:
        data =''
        with open(fileName, "r") as f:
            [ui.trainingSamples, ui.responses] = pickle.load(f)

def train(ui):
    ui.trainedClassifiers , ui.trainedClassifiersNames = [], []
    for i in range(ui.listWidget_8.count()):
        try :
            samps = np.array(ui.trainingSamples).astype(np.float32)
            resps = np.array(ui.responses).astype(np.float32)
        except:
            samps, resps = _removeZeroLenSamples(ui.trainingSamples, ui.responses)
            samps, resps = np.array(samps).astype(np.float32), np.array(resps).astype(np.float32)
        csfr, csfrName = _getClassifier(ui, i, samps, resps)
        ui.trainedClassifiers.append(csfr)
        ui.trainedClassifiersNames.append(csfrName)

def _removeZeroLenSamples(samples, responses):
    sampList, respList = [], []
    for samp, resp in zip(samples, responses):
        if len(samp)!=0:
            sampList.append(samp)
            respList.append(resp)
    return sampList, respList

def runTest(ui):
    train(ui)
    resultText=""
    finalResultText=""
    print "Poszlo do testow"

    for classifier, classifierName in zip(ui.trainedClassifiers, ui.trainedClassifiersNames):
        i, j, k, correctAnswers = 0,0,ui.listWidget_9.count(), 0
        while k>0:
            k=k-1
            fileUrl, label = str(ui.listWidget_9.item(k).text()).split(',')
            print fileUrl

            label = int(label)
            video  = cv2.VideoCapture(fileUrl)
            ret, frame = video.read()
            algorithmName = str(ui.comboBox.currentText()).replace('.py','')
            algorithmClass = _loadAlgorithmClass(algorithmsFolder,algorithmName)()
            algorithmClass._initiateRegisters(frame)
            avaragePeriod = ui.spinBox.value()
            outlierDistance = 4

            while ret:
                algorithmClass.analyze(frame, avaragePeriod)
                if 0xFF & cv2.waitKey(5) == 27:
                    break
                ret, frame = video.read()
                i+=1
                if i%avaragePeriod==0:
                    j+=1
                    resultDict = algorithmClass.getResultDict(i, outlierDistance)
                    if len(resultDict['keyPointsVector']):
                        if classifier.predict(np.array(resultDict['keyPointsVector'], dtype=np.float32)) == label:
                            correctAnswers+=1
                        resultText= str(correctAnswers) + " / " + str(j) + "  " + str(100*correctAnswers/float(j)) + " %\n"
                        ui.plainTextEdit.setPlainText(finalResultText+ resultText)
        finalResultText += classifierName + " " + resultText
        ui.plainTextEdit.setPlainText(finalResultText)

def _getClassifier(ui, rowNumber, trainingSamples, responses):
    rowString = str(ui.listWidget_8.item(rowNumber).text())
    classifierName, param = rowString.split(",")
    classifierName = str(classifierName.replace('.py',''))
    if param:
        param = int(param)
        classifier = _loadAlgorithmClass(classifiersFolder, classifierName)(param)
    else:
        classifier = _loadAlgorithmClass(classifiersFolder, classifierName)()
    classifier.train(trainingSamples, responses)
    return classifier, rowString

def addSample(ui):
    fileName= QtGui.QFileDialog.getOpenFileName(None,'Open file','data/input')
    if len(fileName)>0:
        currentEnviroment = str(ui.listWidget_2.currentItem().text())
        ui.filmSamples[currentEnviroment].append(str(fileName))
        ui.listWidget_5.clear()
        ui.listWidget_5.insertItems(0,ui.filmSamples[currentEnviroment])

def chooseTestFile(ui):
    fileName= QtGui.QFileDialog.getOpenFileName(None,'Open file','data/input')
    ui.toolButton_3.setText(fileName)
    ui.toolButton_3.setStyleSheet('QPushButton {font: bold 14px;}')
    ui.listWidget_9.insertItem(0, fileName+','+str(ui.spinBox_3.value()))

"""
def _writeDistribution(algorithm, distX, distY):
    global output
    output += "\n{}\n".format(algorithm)
    _writeCSVrow("Val(x)", distX[1])
    _writeCSVrow("x", distX[0])
    _writeCSVrow("Val(y)", distY[1])
    _writeCSVrow("y", distY[0])

def _writeCSVrow(label, data):
    global output
    output+="{}".format(label)
    for value in data:
        output+=",{}".format(value)
    output+="\n"

"""

def _analyse(ui, fileUrls, algorithm):
    global output
    if len(fileUrls)!=0:
        algorithmClass= _loadAlgorithmClass(algorithmsFolder,algorithm)()
        if len(str(ui.qwtPlot.title().text()))==0:
            ui.curveX, ui.curveY = preparePlot(ui, algorithm)

        for fileUrl in fileUrls:
            print fileUrl
            video, i,j = cv2.VideoCapture(fileUrl), 0,0
            ret, frame = video.read()
            algorithmClass._initiateRegisters(frame)
            display1 = Display(ui.label_17)
            display2 = Display(ui.label_18)
            display3 = Display(ui.label_7)
            avaragePeriod, outlierDistance = ui.spinBox.value(), ui.spinBox_2.value()
            opencvPreview = ui.checkBox_2.isChecked()
            ui.resultDict = {}
            while ret:
                display1.display(frame)
                algorithmClass.analyze(frame, avaragePeriod)
                display2.display(algorithmClass.processedFrame)
                frame2= algorithmClass.processedFrame
                if not opencvPreview:
                    frame2 = np.zeros((100,100,3), dtype=np.uint8)
                cv2.imshow("raw preview ", frame2)
                if 0xFF & cv2.waitKey(5) == 27:
                   break
                ret, frame = video.read()
                if i%ui.spinBox.value()==0 and i>0:
                    ui.resultDict = algorithmClass.getResultDict(i, outlierDistance)
                    output.append(ui.resultDict)
                    #print "output len ", sys.getsizeof(output)

                    if j==0:
                        printAlgorithmResultsKeys(ui, j)
                        j+=1
                i+=1
            video.release()

def _loadAlgorithmClass(algorithmsFolder, algorithmName, param = None):
    moduleName = algorithmsFolder +'.'+ algorithmName
    mod = __import__(moduleName, fromlist=[algorithmName])
    klass = getattr(mod, algorithmName)
    return klass

def drawAlgorithmResults(ui, resultDict):
    ui.listWidget_7.currentItem().text()

def preparePlot(ui, title):
    ui.qwtPlot.setTitle(title)
    curveX, curveY = QwtPlotCurve("X distribution"), QwtPlotCurve("Y distribution")
    curveX.attach(ui.qwtPlot)
    curveY.attach(ui.qwtPlot)
    curveX.setSymbol(QwtSymbol(QwtSymbol.Ellipse, Qt.QBrush(), Qt.QPen(Qt.Qt.yellow), Qt.QSize(7, 7)))
    curveX.setPen(Qt.QPen(Qt.Qt.red))
    curveY.setPen(Qt.QPen(Qt.Qt.blue))
    return curveX, curveY

def prepareDistributionImage(image):
    valmax = np.amax(image)
    if valmax>0:
        scale = valmax/255.0
    else:
        scale = 1
    img = stats.threshold(image, np.median(image))
    return img/scale

def replotResults(ui, res0, res1, curveX, curveY):
    curveX.setData(res0[0],res0[1])
    curveY.setData(res1[0],res1[1])
    ui.qwtPlot.replot()

def printAlgorithmResultsKeys(ui, i=-1):
    if i==0:
        ui.listWidget_7.clear()
        for key, value in ui.resultDict.iteritems():
            ui.listWidget_7.insertItem(0, key)

def printAlgorithmResults(ui):
    curItem = ui.listWidget_7.currentItem()
    if curItem:
        keyName = str(curItem.text())
        ui.plainTextEdit_2.setPlainText(str(ui.resultDict[keyName]))

def _fillAlgorithmList(ui):
    algorithms = os.listdir(algorithmsFolder)
    algorithms = [k for k in algorithms if ((not '.pyc' in k) and (not '__' in k))]
    ui.listWidget.insertItems(0,algorithms)
    ui.listWidget.setCurrentRow(0)

    classifiers = os.listdir(classifiersFolder)
    classifiers = [k for k in classifiers if ((not '.pyc' in k) and (not '__' in k))]
    ui.listWidget_4.insertItems(0,classifiers)
    ui.listWidget_4.setCurrentRow(0)
    ui.listWidget_2.setCurrentRow(0)

    ui.comboBox.addItems(algorithms)

def addToTraining(ui):
    item = QtGui.QListWidgetItem(ui.listWidget.currentItem().text())
    ui.listWidget_6.addItem(item)
    print ui.listWidget.currentItem().text()

def fillSamplesList(ui):
    ui.listWidget_5.clear()
    currentEnviroment = str(ui.listWidget_2.currentItem().text())
    ui.listWidget_5.insertItems(0,ui.filmSamples[currentEnviroment])

def _initiatePlot(ui):
    mY = QwtPlotMarker()
    mY.setLabelAlignment(Qt.Qt.AlignRight | Qt.Qt.AlignTop)
    mY.setLineStyle(QwtPlotMarker.HLine)
    mY.setYValue(0.0)
    mY.attach(ui.qwtPlot)
    ui.qwtPlot.setAxisTitle(QwtPlot.xBottom, "Pixel position")
    ui.qwtPlot.setAxisTitle(QwtPlot.yLeft, "Values")
    legend = Qwt.QwtLegend()
    legend.setItemMode(Qwt.QwtLegend.CheckableItem)
    ui.qwtPlot.insertLegend(legend, Qwt.QwtPlot.RightLegend)
    ui.qwtPlot.replot()
    ui.qwtPlot.show()

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    connect(ui)
    MainWindow.show()
    sys.exit(app.exec_())
import datetime
import sys
import random
from PyQt5.QtCore import QTimer
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QDialog
from DecisionTree.DecisionTreeFunctions import predict_Patient_priority
from PyQt5 import QtWidgets
from PyQt5.QtGui import QFont, QColor, QBrush
from PyQt5.QtWidgets import QApplication, QMainWindow
import pandas as pd
from Utils.csvFileFunctions import add_column_in_csv
from design_data import logo_rc, image2_rc, imageMarker_rc



# Data structures declaration.
patientesTableData = {}
patientesTableDataByName_BeforeTreatment = {}
patientesTableDataByName_DuringTreatment = {}
patientesTableScoresByName_DuringTreatment = {}
patientesTableDataByName_AfterTreatment = {}
patientesDataAndRating_SaveToDB = [["firstName", "lastName", "patientID", "predictedPriorityRate", "doctorEvaluation"]]
patientesTableScoresByName_AfterTreatment = {}
seconds_error_message_rating_table = 0
errorMessageRatingTable = " "
seconds_error_message_treatment_window = 0
errorMessageTreatmentWindow = " "


# Class for the emergency button simulation window.
class EmergencyButtonWindow(QMainWindow):
    def __init__(self):
        super(EmergencyButtonWindow, self).__init__()
        self.initUI()

    # Function that runs when the emergency button was clicked
    def emergency_button_clicked(self):
        mainwindow.perdict_patient_urgency_score()
        mainwindow.loaddata()

    # The function that initializes the UI
    def initUI(self):
        self.setGeometry(400, 400, 500, 440)
        self.setWindowTitle("Patient Emergency Button Simulation App")

        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText("Push Emergency Button")
        self.b1.clicked.connect(self.emergency_button_clicked)
        self.b1.setGeometry(120, 350, 250, 70)
        font = QFont('Agency FB', 20, weight=100)
        self.b1.setFont(font)
        self.setStyleSheet("background-image: url(EMERGENCY_BUTTON.jpg); background-attachment: fixed")
        self.setFixedSize(self.size())


# Class for the simulation window of the button that is outside the patient room.
# The medical stuff use this button when they start the patient treatment and when they're done treating him.
class TreatmentWindow(QMainWindow):
    def __init__(self):
        super(TreatmentWindow, self).__init__()
        self.initUI()

    # Function for starting a timer that shows an error message
    def showMessageErrorTimer(self):
        global seconds_error_message_treatment_window
        self.timer = QTimer()
        seconds_error_message_treatment_window = 0
        self.timer.timeout.connect(self.showErrorMessageForTwoSeconds)
        self.timer.start(1000)  # The timer execute the function every 1 second

    # Function that the timer runs that shows the error message
    def showErrorMessageForTwoSeconds(self):
        global seconds_error_message_treatment_window, errorMessageTreatmentWindow
        self.label_Error.setText(errorMessageTreatmentWindow)
        seconds_error_message_treatment_window += 1
        # Because the timer execute this function every 1 second,at the 4th execution,
        # after 4 seconds has passed the error message will disappear.
        if (seconds_error_message_treatment_window == 4):
            seconds_error_message_treatment_window = 0
            errorMessageTreatmentWindow = " "
            self.label_Error.setText(" ")
            self.timer.stop()

    # Function that runs when the start treatment button was clicked
    def start_treatment_button_clicked(self):
        global errorMessageTreatmentWindow
        print("'start_treatment_button_clicked' clicked")
        # Check if the first name or the last name is an empty string.
        if self.textboxFirstName.text() == "" or self.textboxLastName.text() == "":
            errorMessageTreatmentWindow = "Please enter valid patient name"
            self.showMessageErrorTimer()
            return
        first_name = self.textboxFirstName.text()
        last_name = self.textboxLastName.text()
        full_name = first_name + " " + last_name
        # Check if the patient ,whose name entered, pressed the emergency button and require a treatment.
        if full_name not in patientesTableDataByName_BeforeTreatment.keys():
            errorMessageTreatmentWindow = "The entered name is not a name of a \n patient who pressed the emergency button"
            self.showMessageErrorTimer()
            return
        # Call the function that updates that the patient started getting the treatment
        mainwindow.start_patient_treatment(first_name, last_name)

    # Function that runs when the end treatment button was clicked
    def end_treatment_button_clicked(self):
        global errorMessageTreatmentWindow
        print("'end_treatment_button_clicked' clicked")
        # Check if the first name or the last name is an empty string.
        if self.textboxFirstName.text() == "" or self.textboxLastName.text() == "":
            errorMessageTreatmentWindow = "Please enter valid patient name"
            self.showMessageErrorTimer()
            return
        first_name = self.textboxFirstName.text()
        last_name = self.textboxLastName.text()
        full_name = first_name + " " + last_name
        # Check if the patient ,whose name entered, pressed the emergency button a medical staff member treated him.
        if full_name not in patientesTableDataByName_DuringTreatment.keys():
            errorMessageTreatmentWindow = "The entered name is not a name of a \n patient who got treatment"
            self.showMessageErrorTimer()
            return
        # Call the function that updates that the patient treatment ended
        mainwindow.end_patient_treatment(first_name, last_name)

    # The function that initializes the UI
    def initUI(self):
        self.setGeometry(400, 400, 500, 440)
        self.setWindowTitle("Medical staff Treatment Button Simulation App")
        self.btn_Start_Treatment = QtWidgets.QPushButton(self)
        self.btn_End_Treatment = QtWidgets.QPushButton(self)
        self.labelFirstName = QtWidgets.QLabel('Patient First Name :', self)
        self.labelLastName = QtWidgets.QLabel('Patient Last Name :', self)
        self.label_Error = QtWidgets.QLabel(' ', self)
        self.textboxFirstName = QtWidgets.QLineEdit(self)
        self.textboxLastName = QtWidgets.QLineEdit(self)

        self.btn_Start_Treatment.setText("Start Treatment")
        self.btn_Start_Treatment.clicked.connect(self.start_treatment_button_clicked)
        self.btn_Start_Treatment.setGeometry(50, 350, 180, 70)
        font = QFont('Agency FB', 20, weight=100)
        self.btn_Start_Treatment.setFont(font)

        self.btn_End_Treatment.setText("End Treatment")
        self.btn_End_Treatment.clicked.connect(self.end_treatment_button_clicked)
        self.btn_End_Treatment.setGeometry(270, 350, 180, 70)
        self.btn_End_Treatment.setFont(font)

        font = QFont('Agency FB', 20, weight=100)
        self.labelFirstName.setFont(font)
        self.labelFirstName.setGeometry(50, 20, 170, 40)

        self.labelLastName.setFont(font)
        self.labelLastName.setGeometry(50, 70, 170, 40)

        font = QFont('Agency FB', 18, weight=100)
        self.label_Error.setFont(font)
        self.label_Error.setGeometry(70, 270, 350, 80)

        self.textboxFirstName.setGeometry(240, 20, 170, 40)
        self.textboxLastName.setGeometry(240, 70, 170, 40)
        self.setStyleSheet("background-image: url(TreatmentButton.jpg); background-attachment: fixed")
        self.label_Error.setStyleSheet("color: rgb(223, 0, 0);background: transparent;")
        self.labelFirstName.setStyleSheet("background: transparent;")
        self.labelLastName.setStyleSheet("background: transparent;")
        self.setFixedSize(self.size())


# Class for the main window that is displayed to the medical staff.
# It contains 2 table in 2 different tabs:
# The first tab is the 'Distress Calls Table' tab.
# It displays a table of all the distress calls sorted by the predicted
# priority rating (the most urgent call will appear first),each row represents a distress call. Each row contains the
# patient's first and last name, gender, room number, bed number, ICD-9 diagnosis ( International Classification of
# Diseases), BMI, pulse and priority rating (predicted by the software).The priority rating is between 1 and 100 when
# 1 is "Not Urgent" and 100 is an "Emergency".
# The second tab is the 'Distress Calls Rating' tab.
# It displays a table called "Distress Calls Rating" table of all the treated distress calls.
# The staff member that treated a call can rate the distress calls in the "Distress Calls Rating" table in a range of
# 1 to 100 according to their medical knowledge and evaluation.

class MainWindow(QDialog):
    # The function that initializes the UI
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('HospitalResponseSystemUI.ui', self)
        self.updateRatingButton_2.clicked.connect(self.updateRating)
        self.DistressCallsTable.setColumnWidth(0, 120)
        self.DistressCallsTable.setColumnWidth(1, 124)
        self.DistressCallsTable.setColumnWidth(2, 120)
        self.DistressCallsTable.setColumnWidth(3, 120)
        self.DistressCallsTable.setColumnWidth(4, 120)
        self.DistressCallsTable.setColumnWidth(5, 310)
        self.DistressCallsTable.setColumnWidth(6, 120)
        self.DistressCallsTable.setColumnWidth(7, 120)
        self.DistressCallsTable.setColumnWidth(8, 120)
        self.DistressCallsTable.setHorizontalHeaderLabels(
            ["First Name", "Last Name", "Gender", "Room Number", "Bed Number",
             "ICD-9 diagnosis\n(International Classification of Diseases)", "BMI", "Pulse",
             "Priority Rate"])
        self.DistressCallsRatingTable_2.setColumnWidth(0, 120)
        self.DistressCallsRatingTable_2.setColumnWidth(1, 124)
        self.DistressCallsRatingTable_2.setColumnWidth(2, 100)
        self.DistressCallsRatingTable_2.setColumnWidth(3, 100)
        self.DistressCallsRatingTable_2.setColumnWidth(4, 100)
        self.DistressCallsRatingTable_2.setColumnWidth(5, 310)
        self.DistressCallsRatingTable_2.setColumnWidth(6, 120)
        self.DistressCallsRatingTable_2.setColumnWidth(7, 100)
        self.DistressCallsRatingTable_2.setColumnWidth(8, 100)
        self.DistressCallsRatingTable_2.setColumnWidth(9, 100)
        self.DistressCallsRatingTable_2.setHorizontalHeaderLabels(
            ["First Name", "Last Name", "Gender", "Room Number", "Bed Number",
             "ICD-9 diagnosis\n(International Classification of Diseases)", "BMI", "Pulse",
             "Priority Rate", "Doctor\nEvaluation"])
        self.setFixedSize(self.size())

    # The function that is called when the medical staff member pressed the update rating button.
    def updateRating(self):
        global patientesDataAndRating_SaveToDB
        global errorMessageRatingTable
        patientDataAndRating_SaveToDB = []
        # Check if there are patients who already got treated and the medical staff member need to fill the doctor
        # evaluation section .
        if (patientesTableDataByName_AfterTreatment):
            if (patientesTableDataByName_AfterTreatment.keys()):
                numberOfRows = len(patientesTableDataByName_AfterTreatment.keys())
                # Get doctor evaluation of each cell at the doctor evaluation column.
                # And check if the doctor evaluation is a valid number - an integer between 1-100.
                # If not - it call a function that shows a suitable error message for 3 seconds.
                for rowIndex in range(numberOfRows):
                    doctorEvItem = self.DistressCallsRatingTable_2.item(rowIndex, 9)
                    doctorEv = doctorEvItem.text()
                    if (not doctorEv):
                        errorMessageRatingTable = "Please fill the doctor evaluation section"
                        self.showMessageErrorTimerRatingTable()
                        continue
                    if (not doctorEv.isnumeric() or int(doctorEv) > 100 or int(doctorEv) < 1):
                        errorMessageRatingTable = "Doctor evaluation value is invalid"
                        self.showMessageErrorTimerRatingTable()
                        continue
                    # Get the patient data from rating table
                    patientFirstName = self.DistressCallsRatingTable_2.item(rowIndex, 0).text()
                    patientLastName = self.DistressCallsRatingTable_2.item(rowIndex, 1).text()
                    patientName = patientFirstName + " " + patientLastName
                    currData = patientesTableDataByName_AfterTreatment.pop(patientName)
                    patientID = currData[['eid']].values[0][0]
                    patientesTableScoresByName_AfterTreatment.pop(patientName)
                    patientDataAndRating_SaveToDB.append(patientFirstName)
                    patientDataAndRating_SaveToDB.append(patientLastName)
                    patientDataAndRating_SaveToDB.append(patientID)
                    predictedPriorityRateItem = self.DistressCallsRatingTable_2.item(rowIndex, 8)
                    predictedPriorityRate = predictedPriorityRateItem.text()
                    patientDataAndRating_SaveToDB.append(predictedPriorityRate)
                    patientDataAndRating_SaveToDB.append(doctorEv)
                    patientesDataAndRating_SaveToDB.append(patientDataAndRating_SaveToDB)
                    patientDataAndRating_SaveToDB = []
                    # Get the current date and save the patient data with the doctor's evaluation score to an .csv
                    # file with of the same day (each day a new .csv file is created for organization purposes and to
                    # prevent overflow). The data is saved to a csv file so later on we could use the data to improve
                    # our model.
                    x = datetime.datetime.now()
                    dateAndTime = x.strftime("%c")
                    dateComponents = dateAndTime.split()
                    currDate = "_" + dateComponents[0] + "_" + dateComponents[1] + "_" + dateComponents[2] + "_" + \
                               dateComponents[4]
                    path = '..\\Data\\RatingData' + currDate + '.csv'
                    add_column_in_csv(path, patientesDataAndRating_SaveToDB)
                self.loadRatingData()
        else:
            errorMessageRatingTable = "There is no rating to update"
            self.showMessageErrorTimerRatingTable()

    # Function for starting a timer that shows an error message
    def showMessageErrorTimerRatingTable(self):
        global seconds_error_message_rating_table
        self.timer = QTimer()
        seconds_error_message_rating_table = 0
        self.timer.timeout.connect(self.showRatingTableErrorMessageForTwoSeconds)
        self.timer.start(1000)  # The timer execute the function every 1 second

    # Function that the timer runs that shows the error message
    def showRatingTableErrorMessageForTwoSeconds(self):
        global seconds_error_message_rating_table, errorMessageRatingTable
        self.label_ERROR.setText(errorMessageRatingTable)
        seconds_error_message_rating_table += 1
        # Because the timer execute this function every 1 second,at the 3rd execution,
        # after 3 seconds has passed the error message will disappear.
        if (seconds_error_message_rating_table == 3):
            seconds_error_message_rating_table = 0
            errorMessageRatingTable = " "
            self.label_ERROR.setText(" ")
            self.timer.stop()

    # The function that runs the urgency score algorithm of the model to predict a patient urgency rate.
    def perdict_patient_urgency_score(self):
        # The last 20,000 entries of the dataset is used for testing.
        # Therefore, we randomly select 1 person from the testing section for simulation purposes.
        data = pd.read_csv('..\\Data\\patientsList.csv')
        index = random.randrange(80000, 100000)
        data = data[index:index + 1]
        emergencyScore = predict_Patient_priority(data)
        name = data[['firstName']].values[0][0] + " " + data[['lastName']].values[0][0]
        if (patientesTableDataByName_BeforeTreatment):
            # If he already in the distress call table, take into consideration his first call.
            # Don't run it over, so he'll need to wait all over again.
            if name in patientesTableDataByName_BeforeTreatment.keys():
                return
        patientesTableDataByName_BeforeTreatment[name] = data
        # If there's another patient with the same score, the one who made the call first will be treated first.
        if (patientesTableData):
            if emergencyScore in patientesTableData.keys():
                patientesTableData[emergencyScore].append(data)
            else:
                patientesTableData[emergencyScore] = [data]
        else:
            patientesTableData[emergencyScore] = [data]

    # The function that initialized that loads the patients data to the distress call table
    def loaddata(self):
        tablerow = 0
        # Sort the patients from the most urgent call to the least urgent call
        if (patientesTableData.keys()):
            keys = sorted(patientesTableData.keys())
            keys.reverse()
            for key in keys:
                givenScoreDataList = patientesTableData[key]
                for currData in givenScoreDataList:
                    tablerow += 1
                    self.DistressCallsTable.setRowCount(tablerow)
                    self.DistressCallsTable.setItem(tablerow - 1, 0,
                                                    QtWidgets.QTableWidgetItem(currData[['firstName']].values[0][0]))
                    self.DistressCallsTable.setItem(tablerow - 1, 1,
                                                    QtWidgets.QTableWidgetItem(currData[['lastName']].values[0][0]))
                    self.DistressCallsTable.setItem(tablerow - 1, 2,
                                                    QtWidgets.QTableWidgetItem(str(currData[['gender']].values[0][0])))
                    self.DistressCallsTable.setItem(tablerow - 1, 3, QtWidgets.QTableWidgetItem(
                        str(currData[['room Number']].values[0][0])))
                    self.DistressCallsTable.setItem(tablerow - 1, 4, QtWidgets.QTableWidgetItem(
                        str(currData[['bed Number']].values[0][0])))
                    self.DistressCallsTable.setItem(tablerow - 1, 5,
                                                    QtWidgets.QTableWidgetItem(str(self.get_diagnosis(currData))))
                    self.DistressCallsTable.setItem(tablerow - 1, 6, QtWidgets.QTableWidgetItem(
                        str(round(currData[['bmi']].values[0][0], 2))))
                    self.DistressCallsTable.setItem(tablerow - 1, 7,
                                                    QtWidgets.QTableWidgetItem(str(currData[['pulse']].values[0][0])))
                    emergencyScoreTableItem = QtWidgets.QTableWidgetItem(str(key))
                    if (key >= 75):
                        emergencyScoreTableItem.setForeground(QBrush(QColor(204, 0, 0)))
                    if (key < 75 and key >= 45):
                        emergencyScoreTableItem.setForeground(QBrush(QColor(255, 145, 34)))
                    if (key < 45):
                        emergencyScoreTableItem.setForeground(QBrush(QColor(0, 153, 0)))
                    self.DistressCallsTable.setItem(tablerow - 1, 8, emergencyScoreTableItem)
        else:
            self.DistressCallsTable.setRowCount(0)

    # The function that initialized that loads the patients data to the distress call rating table
    def loadRatingData(self):
        tablerow = 0
        if (patientesTableScoresByName_AfterTreatment and patientesTableDataByName_AfterTreatment):
            for name in patientesTableDataByName_AfterTreatment:
                currData = patientesTableDataByName_AfterTreatment[name]
                tablerow += 1
                self.DistressCallsRatingTable_2.setRowCount(tablerow)
                self.DistressCallsRatingTable_2.setItem(tablerow - 1, 0,
                                                        QtWidgets.QTableWidgetItem(
                                                            currData[['firstName']].values[0][0]))
                self.DistressCallsRatingTable_2.setItem(tablerow - 1, 1,
                                                        QtWidgets.QTableWidgetItem(currData[['lastName']].values[0][0]))
                self.DistressCallsRatingTable_2.setItem(tablerow - 1, 2,
                                                        QtWidgets.QTableWidgetItem(
                                                            str(currData[['gender']].values[0][0])))
                self.DistressCallsRatingTable_2.setItem(tablerow - 1, 3, QtWidgets.QTableWidgetItem(
                    str(currData[['room Number']].values[0][0])))
                self.DistressCallsRatingTable_2.setItem(tablerow - 1, 4, QtWidgets.QTableWidgetItem(
                    str(currData[['bed Number']].values[0][0])))
                self.DistressCallsRatingTable_2.setItem(tablerow - 1, 5,
                                                        QtWidgets.QTableWidgetItem(str(self.get_diagnosis(currData))))
                self.DistressCallsRatingTable_2.setItem(tablerow - 1, 6, QtWidgets.QTableWidgetItem(
                    str(round(currData[['bmi']].values[0][0], 2))))
                self.DistressCallsRatingTable_2.setItem(tablerow - 1, 7,
                                                        QtWidgets.QTableWidgetItem(
                                                            str(currData[['pulse']].values[0][0])))
                emergencyScore = patientesTableScoresByName_AfterTreatment[name]
                emergencyScoreTableItem = QtWidgets.QTableWidgetItem(str(emergencyScore))
                if (emergencyScore >= 75):
                    emergencyScoreTableItem.setForeground(QBrush(QColor(204, 0, 0)))
                if (emergencyScore < 75 and emergencyScore >= 45):
                    emergencyScoreTableItem.setForeground(QBrush(QColor(255, 145, 34)))
                if (emergencyScore < 45):
                    emergencyScoreTableItem.setForeground(QBrush(QColor(0, 153, 0)))
                self.DistressCallsRatingTable_2.setItem(tablerow - 1, 8, emergencyScoreTableItem)
                self.DistressCallsRatingTable_2.setItem(tablerow - 1, 9, QtWidgets.QTableWidgetItem(""))
        else:
            self.DistressCallsRatingTable_2.setRowCount(0)

    # The function that updates that the patient started getting the treatment
    def start_patient_treatment(self, first_name, last_name):
        # Call the function that finds the emergency score saved according to the name of the patient
        name = first_name + " " + last_name
        emergencyScore = self.get_emergencyScore_from_name(first_name, last_name)
        if (emergencyScore == -1):
            return
        # Check if the patient pressed the emergency button.
        # If so remove the patient dictionary an update the distress calls table
        if (patientesTableDataByName_BeforeTreatment):
            if name in patientesTableDataByName_BeforeTreatment.keys():
                data = patientesTableDataByName_BeforeTreatment.pop(name)
                patientesTableDataByName_DuringTreatment[name] = data
                patientesTableScoresByName_DuringTreatment[name] = emergencyScore
                patientsList = patientesTableData[emergencyScore]
                if len(patientsList) == 1:
                    patientesTableData.pop(emergencyScore)
                else:
                    patientIndex = 0
                    for patient in patientsList:
                        if data.equals(patient):
                            break;
                        patientIndex += 1
                    patientsList.pop(patientIndex)
                self.loaddata()

    # The function that updates that the patient treatment was ended
    def end_patient_treatment(self, first_name, last_name):
        name = first_name + " " + last_name
        # Check if the patient got treated.
        # If so add the patient to the distress calls rating table
        if (patientesTableDataByName_DuringTreatment):
            if name in patientesTableDataByName_DuringTreatment.keys():
                if (patientesTableScoresByName_DuringTreatment):
                    if name in patientesTableScoresByName_DuringTreatment.keys():
                        data = patientesTableDataByName_DuringTreatment.pop(name)
                        emergencyScore = patientesTableScoresByName_DuringTreatment.pop(name)
                        patientesTableDataByName_AfterTreatment[name] = data
                        patientesTableScoresByName_AfterTreatment[name] = emergencyScore
                        self.loadRatingData()
        return

    # The function that finds the emergency score saved according to the name of the patient
    def get_emergencyScore_from_name(self, first_name, last_name):
        if (patientesTableData and patientesTableData.keys()):
            for key in patientesTableData.keys():
                for patient in patientesTableData[key]:
                    curr_first_name = patient[['firstName']].values[0][0]
                    curr_last_name = patient[['lastName']].values[0][0]
                    if ((curr_first_name == first_name) and (curr_last_name == last_name)):
                        return key
        return -1

    # The function that returns which ICD-9 diagnosis (International Classification of Diseases) diseases the patient
    # has and returns it as a string.
    def get_diagnosis(self, data):
        diagnosis = ""
        if data[['dialysisrenalendstage']].values[0][0] == 1:
            diagnosis += "renal disease"
        if data[['asthma']].values[0][0] == 1:
            if (diagnosis):
                diagnosis += ","
            diagnosis += "asthma"
        if data[['irondef']].values[0][0] == 1:
            if (diagnosis):
                diagnosis += ","
            diagnosis += "iron deficiency"
        if data[['pneum']].values[0][0] == 1:
            if (diagnosis):
                diagnosis += ","
            diagnosis += "pneumonia"
        if data[['substancedependence']].values[0][0] == 1:
            if (diagnosis):
                diagnosis += ","
            diagnosis += "substance dependence"
        if data[['psychologicaldisordermajor']].values[0][0] == 1:
            if (diagnosis):
                diagnosis += ","
            diagnosis += "major psychological disorder"
        if data[['depress']].values[0][0] == 1:
            if (diagnosis):
                diagnosis += ","
            diagnosis += "depression"
        if data[['psychother']].values[0][0] == 1:
            if (diagnosis):
                diagnosis += ","
            diagnosis += " psychological disorder"
        if data[['fibrosisandother']].values[0][0] == 1:
            if (diagnosis):
                diagnosis += ","
            diagnosis += "fibrosis"
        if data[['malnutrition']].values[0][0] == 1:
            if (diagnosis):
                diagnosis += ","
            diagnosis += "malnutrituion"
        if data[['hemo']].values[0][0] == 1:
            if (diagnosis):
                diagnosis += ","
            diagnosis += "blood disorder"
        if (not diagnosis):
            if data[['secondarydiagnosisnonicd9']].values[0][0] == 1:
                diagnosis += "secondary diagnosis (non ICD-9)"
            else:
                diagnosis = "unknown"
        return diagnosis

# The simulation windows execution function
def windows():
    app = QApplication(sys.argv)
    win1 = EmergencyButtonWindow()
    win2 = TreatmentWindow()
    win1.show()
    win2.show()
    sys.exit(app.exec_())


# main
app = QApplication(sys.argv)
mainwindow = MainWindow()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedHeight(821)
widget.setFixedWidth(1400)
widget.setWindowTitle("Hospital Response System App")
widget.show()
windows()
try:
    sys.exit(app.exec_())
except:
    print("Exiting")

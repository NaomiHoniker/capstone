import PySimpleGUI as sg
from capture import *
import os

sg.theme('DarkGrey6')

home_window = sg.Window('Sign Language Interpreter', [[sg.Text('Input Capture Classifier:'), sg.Input(key='letter')],
                                                      [sg.Text('Amount of Consecutive Captures:'),
                                                       sg.Input(0, key='count')],
                                                      [sg.Text('Directory To Place Captured Image Folder'),
                                                       sg.FolderBrowse(key='path')],
                                                      [sg.Button('Begin Capturing'), sg.Button('Exit')]])

while True:
    event, values = home_window.read()
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elif event == 'Begin Capturing':
        if values['letter'].isalpha() and int(values['count']) > 0\
                and values['path'] is not "":
            print(values['path'])
            home_window.Hide()
            gather_data(values['letter'], int(values['count']), values['path'] + "/")
            home_window.UnHide()
        else:
            continue

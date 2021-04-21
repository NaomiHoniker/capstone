import PySimpleGUI as sg
from capture import capture

sg.theme('DarkGrey6')

models_list = ['Sign Language', 'Rock Paper Scissors']

home_window = sg.Window('Sign Language Interpreter', [[sg.Text('Select Model:'),
                                                       sg.Combo(values=models_list, key='dropdown')],
                                                      [sg.Button('Begin Capturing'), sg.Button('Exit')]])

while True:
    event, values = home_window.read()
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elif event == 'Begin Capturing':
        if values['dropdown'] != '':  # If the dropdown isn't empty..
            home_window.Hide()
            # Call the capture function
            capture(values['dropdown'])
            home_window.UnHide()
        else:
            continue

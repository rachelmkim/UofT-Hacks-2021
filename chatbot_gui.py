import tkinter as tk


def send_message():
    message = ENTRYTEXT.get('0.0', 'end-1c')
    # TODO: call the function in another module that processes the text
    ENTRYTEXT.delete('0.0', 'end')

    if message != '':
        CHATLOG.config()
        CHATLOG.insert('end', "You: " + message + '\n\n')

        bot_message = receive_message()
        CHATLOG.config()
        CHATLOG.insert('end', "Bot: " + bot_message + '\n\n')


def receive_message() -> str:
    # TODO: call the function in another module that gives back the chatbot's response
    # TODO: change the variable message below
    message = 'test'

    return message


# Main Part of the File
ROOT = tk.Tk()
# app = Application(master=root)

CANVAS = tk.Canvas(ROOT, width=500, height=500)
CANVAS.pack()

CHATLOG = tk.Text(ROOT, height=20, width=50)
CANVAS.create_window(250, 200, window=CHATLOG)
SCROLL_BAR_FOR_CHATLOG = tk.Scrollbar(ROOT, command=CHATLOG.yview)
SCROLL_BAR_FOR_CHATLOG.place(x=450, y=40, height=325)
CHATLOG['yscrollcommand'] = SCROLL_BAR_FOR_CHATLOG.set

ENTRYTEXT = tk.Text(ROOT, height=3, width=50)
CANVAS.create_window(250, 400, window=ENTRYTEXT)
SCROLL_BAR_FOR_ENTRY = tk.Scrollbar(ROOT, command=ENTRYTEXT.yview)
SCROLL_BAR_FOR_ENTRY.place(x=450, y=375, height=50)
ENTRYTEXT['yscrollcommand'] = SCROLL_BAR_FOR_ENTRY.set

SEND_BUTTON = tk.Button(ROOT, text='Send', command=send_message)
SEND_BUTTON.pack(side="right")

ROOT.mainloop()

import regex

def conversation_list() -> list:
    """ read movie_conversations and make a list of ids of lines
    """
    lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
    conv_id = []
    for line in lines[:-1]:
        line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        conv_id.append(line.split(','))
    return conv_id
 
    
def line_dict() -> dict:
    """ read movie_lines and return id: line dict
    """
    lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
    id_line = {}
    for line in lines:
        line = line.split(' +++$+++ ')
        if len(line) == 5:
            id_line[line[0]] = line[4]
    return id_line

 
def q_and_a() -> tuple:
    """
    return lists of q and a lines
    """
    conv_id = conversation_list()
    id_line = line_dict()
    questions = []
    answers = []
    for conv in conv_id:
        for i in range(len(conv) - 1):
            questions.append(id_line[conv[i]])
            answers.append(id_line[conv[i + 1]])
    return questions, answers


def clean_text(text) -> str:
    text = text.lower()
    text = regex.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text


# clean the Qs and As
questions = q_and_a()[0]
answers = q_and_a()[1]

clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
clean_answers = []    
for answer in answers:
    clean_answers.append(clean_text(answer))
    
    
    
# uses code from https://github.com/suriyadeepan/practical_seq2seq/blob/master/datasets/cornell_corpus/data.py

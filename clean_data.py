def conversation_list() -> list:
    """ read movie_conversations and make an id list
    """
    lines = open('movie_conversations.txt').read().split('\n')
    id_list = []
    for line in lines[:-1]:
        line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        id_list.append(line.split(','))
    return id_list
 

def line_dict() -> dict:
    """ read movie_lines and return id: line dict.
    """
    lines = open('movie_lines.txt').read().split('\n')
    id_line = {}
    for line in lines:
        line = line.split(' +++$+++ ')
        id_line[line[0]] = line[4]
    return id_line
    # TODO: fix UnicodeDecodeError
    
    
def conversation_file(conv_id, id_line):
    """ input: the output of the above functions
    make conversation file
    """
    id = 0
    for conversation in conv_id:
        f = open(str(id) + '.txt', 'w')
        for line_id in conversation:
            f.write(id_line[line_id])
            f.write('\n')
        f.close()
        id += 1

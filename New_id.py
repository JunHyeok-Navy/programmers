from string import ascii_lowercase

def solution(new_id):
    alphabet_list = list(ascii_lowercase)
    list_1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '_', '.']
    list_2 = []

    #1단계
    new_id = new_id.lower()

    #2단계
    for i in range(len(new_id)):
        if new_id[i] in alphabet_list:
            list_2.append(new_id[i])
        elif new_id[i] in list_1:
            list_2.append(new_id[i])

    new_id = ''.join(list_2)


    #3단계
    dots = []
    for i in range(len(new_id), 0, -1):
        dots.append('.'*i)
    for dot in dots:
        if dot in new_id:
            new_id = new_id.replace(dot,'.')

    #4단계
    if new_id[0] == '.':
        new_id = new_id[1:]
    elif new_id[-1] == '.':
        new_id = new_id[:-1]

    #5단계
    if len(new_id) == 0:
        new_id = 'a'

    #6단계
    if len(new_id) >= 16:

        new_id = list(new_id)
        new_id = new_id[0:15]
        new_id = ''.join(new_id)

    if new_id[0] == '.':
        new_id = list(new_id)
        del new_id[0]
        new_id = ''.join(new_id)

    if new_id[-1] == '.':
        new_id = list(new_id)
        del new_id[-1]
        new_id = ''.join(new_id)

    #7단계
    while len(new_id) <= 2:
        new_id = list(new_id)
        new_id.append(new_id[-1])
        new_id = ''.join(new_id)

    answer = new_id
    return answer

def solution(answers):
    fail_1 = [1,2,3,4,5]
    fail_2 = [2,1,2,3,2,4,2,5]
    fail_3 = [3,3,1,1,2,2,4,4,5,5]
    fail_1_c = 0
    fail_2_c = 0
    fail_3_c = 0
    if len(answers)>len(fail_1):
        for i in range(len(answers)-len(fail_1)):
            fail_1.append(fail_1[i])
    if len(answers)>len(fail_2):
        for i in range(len(answers) - len(fail_2)):
            fail_2.append(fail_2[i])
    if len(answers)>len(fail_3):
        for i in range(len(answers) - len(fail_3)):
            fail_3.append(fail_3[i])
    for i in range(len(answers)):
        if fail_1[i]==answers[i]:
            fail_1_c += 1
        if fail_2[i] == answers[i]:
            fail_2_c += 1
        if fail_3[i] == answers[i]:
            fail_3_c += 1
        else:
            pass
    ww = {1 : fail_1_c, 2:fail_2_c, 3:fail_3_c}
    print(ww)
    answer = []
    for key, value in ww.items():
        if value == max(ww.values()):
            answer.append(key)
    return answer

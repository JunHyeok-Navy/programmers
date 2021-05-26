def solution(participant, completion):
    participant.sort()
    completion.sort()
    for i in range(0, len(completion)):
        if not participant[i] == completion[i]:
            answer = participant[i]
            break
        else:
            answer = participant[-1]
    return answer

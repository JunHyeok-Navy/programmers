def change(n):
    p = 0
    while True:
        p += n % 2
        if n == 1:
            break
        n = n // 2
    return p

def solution(n):
    ans = n
    while True:
        ans += 1
        if change(n) == change(ans):
            break
    answer = ans
    return answer

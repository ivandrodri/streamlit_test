import time

TIMES = [1.0, 1.0, 1.0]
NUM_LOOPS = 10


def task_3(iteration: int):
    print(f"Task 3 a subtask of Task 2 RUNNING at iter : {iteration}")
    time.sleep(TIMES[2])
    print(f"Task 3 a subtask of Task 2 DONE at iter : {iteration}")


def task_1(iteration: int):
    task_3(iteration=iteration)
    print(f"Task 1 RUNNING at iter: {iteration}")
    time.sleep(TIMES[0])
    return f"Task 1 DONE at iter : {iteration}"


def task_2(message: str, iteration: int):
    print(f"Task 2 showing Task 1 message: {message} at iter: {iteration}")
    time.sleep(TIMES[1])
    print(f"Task 2 DONE at iter: {iteration}")


def main():

    for i in range(NUM_LOOPS):
        message = task_1(iteration=i)
        task_2(message=message, iteration=i)
        print(f"--------------------------")


if __name__ == '__main__':
    main()



